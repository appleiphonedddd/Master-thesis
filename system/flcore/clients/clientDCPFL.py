import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client

from sklearn.preprocessing import label_binarize
from sklearn import metrics

def _safe_logits(model, x):
    """Supports both (base/head) and single forward models."""
    if hasattr(model, "base") and hasattr(model, "head"):
        h = model.base(x)
        return model.head(h)
    return model(x)

def _stable_logits(z):
    """Row-wise max-shift to avoid overflow in softmax/KL."""
    return z - z.max(dim=1, keepdim=True).values

class clientDCPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(
            self.model_per.parameters(), lr=self.learning_rate)
        self.scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.model_ex = copy.deepcopy(self.model_per)                              
        self.optimizer_ex = torch.optim.SGD(self.model_ex.parameters(),            
                                            lr=self.learning_rate, momentum=0.9)   
        self.has_exchange = False

        num_layers = len(list(self.model.parameters()))
        self.alpha = [getattr(args, "alpha", 1.0)] * num_layers
        self.alpha_lr = float(getattr(args, "alpha_lr", self.learning_rate))
        self.alpha_update_gap = int(getattr(args, "alpha_update_gap", 0))
        self._step = 0  

        self.kd_T = float(getattr(args, "kd_temperature", 1.0))

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for _, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        for m in (self.model, self.model_per, self.model_ex):
            m.to(self.device)

    def set_exchange_model(self, state_dict, exchanged_round=None):  
        """Receive partner's personalized weights from server (exchange model)."""
        try:
            self.model_ex.load_state_dict(state_dict, strict=True)
        except Exception:
            self.model_ex.load_state_dict(state_dict, strict=False)
        self.has_exchange = True 
        if exchanged_round is not None:
            self._exchanged_round = exchanged_round  
        self.model_ex.to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()
        self.model_per.train()                                                      
        if hasattr(self, 'model_ex'):                                               
            self.model_ex.train()                                                   
                                                                                    
        epochs = self.local_epochs                                                  
        if self.train_slow:                                                         
            epochs = np.random.randint(1, max(1, epochs // 2))                      
                                                                                    
        for _ in range(epochs):                                                     
            for x, y in trainloader:                                                
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)  
                y = y.to(self.device)                                               
                if self.train_slow:                                                 
                    time.sleep(0.1 * np.abs(np.random.rand()))                      
                                                                                                                             
                self.optimizer.zero_grad()                                          
                out_g = self.model(x)                                               
                loss_g = self.loss(out_g, y)                                        
                if torch.isfinite(loss_g):
                    loss_g.backward()                                               
                    self.optimizer.step()                                           
                                                                                    
                if getattr(self, "has_exchange", False):
                    out_p = _safe_logits(self.model_per, x)
                    out_e = _safe_logits(self.model_ex,  x)

                    logits_p = _stable_logits(out_p)
                    logits_e = _stable_logits(out_e)

                    ce_p = self.loss(logits_p, y)
                    ce_e = self.loss(logits_e, y)

                    T = self.kd_T if self.kd_T > 0 else 1.0
                    logp = F.log_softmax(logits_p / T, dim=1)
                    te   = F.softmax    (logits_e.detach() / T, dim=1)
                    loge = F.log_softmax(logits_e / T, dim=1)
                    tp   = F.softmax    (logits_p.detach() / T, dim=1)

                    loss_p = ce_p + (T * T) * F.kl_div(logp, te, reduction='batchmean')
                    loss_e = ce_e + (T * T) * F.kl_div(loge, tp, reduction='batchmean')

                    if torch.isfinite(loss_p):
                        self.optimizer_per.zero_grad()
                        loss_p.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), 5.0)
                        self.optimizer_per.step()

                    if torch.isfinite(loss_e):
                        self.optimizer_ex.zero_grad()
                        loss_e.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_ex.parameters(), 5.0)
                        self.optimizer_ex.step()
                else:
                    self.optimizer_per.zero_grad()
                    out_p = _safe_logits(self.model_per, x)
                    logits_p = _stable_logits(out_p)
                    loss_p = self.loss(logits_p, y)
                    if torch.isfinite(loss_p):
                        loss_p.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), 5.0)
                        self.optimizer_per.step()

                self._step += 1
                if self.alpha_update_gap > 0 and (self._step % self.alpha_update_gap == 0):
                    self.alpha_update()

        for idx, (p_per, p_glob) in enumerate(zip(self.model_per.parameters(), self.model.parameters())):
            p_per.data = (1 - self.alpha[idx]) * p_glob.data + self.alpha[idx] * p_per.data
                                                                                    
        if getattr(self, "learning_rate_decay", False) and hasattr(self, "learning_rate_scheduler"):
            self.learning_rate_scheduler.step()
        if hasattr(self, 'scheduler_per') and self.scheduler_per is not None:
            self.scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        model = self.model_per if model is None else model
        model.eval()

        correct = 0
        total = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                out = _safe_logits(model, x)

                correct += (torch.argmax(out, dim=1) == y).sum().item()
                total += y.size(0)

                prob = F.softmax(out, dim=1).cpu().numpy()
                y_prob.append(prob)
                nc = self.num_classes
                lb = label_binarize(y.cpu().numpy(), classes=np.arange(nc))
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return correct, total, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        total = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                out = _safe_logits(self.model_per, x)
                loss = self.loss(out, y)
                total += y.size(0)
                losses += loss.item() * y.size(0)

        return losses, total

    def alpha_update(self):
        # requires gradients computed in the same iteration
        for idx, (gl, gp) in enumerate(zip(self.model.parameters(), self.model_per.parameters())):
            if gp.grad is None or gl.grad is None:
                continue
            diff = (gp.detach() - gl.detach()).reshape(-1)
            grad = (self.alpha[idx] * gp.grad.detach() +
                    (1.0 - self.alpha[idx]) * gl.grad.detach()).reshape(-1)
            grad_alpha = float((diff * grad).sum().item())
            # tiny L2 regularization on alpha to keep it bounded
            grad_alpha += 0.01 * float(self.alpha[idx])
            new_alpha = float(self.alpha[idx]) - float(self.alpha_lr) * grad_alpha
            self.alpha[idx] = max(0.0, min(1.0, new_alpha))

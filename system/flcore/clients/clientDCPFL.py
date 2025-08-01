import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientDCPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # initialize per-client personalized model
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(
            self.model_per.parameters(), lr=self.learning_rate)
        self.scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        # one alpha parameter per model layer for interpolation
        num_layers = len(list(self.model.parameters()))
        self.alpha = [args.alpha] * num_layers

        # record number of samples per class for weighting
        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(self.sample_per_class.int())

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()
        self.model_per.train()

        epochs = self.local_epochs
        if self.train_slow:
            epochs = np.random.randint(1, max(1, epochs // 2))

        for _ in range(epochs):
            for x, y in trainloader:
                # move to device
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # train global branch
                self.optimizer.zero_grad()
                out_g = self.model(x)
                loss_g = self.loss(out_g, y)
                loss_g.backward()
                self.optimizer.step()

                # train personalized branch via deep mutual learning
                rep = self.model_per.base(x)
                out_p = self.model_per.head(rep)
                loss_p = self.loss(out_p, y)
                self.optimizer_per.zero_grad()
                loss_p.backward()
                self.optimizer_per.step()

                # update interpolation weights
                self.alpha_update()

        # interpolate parameters between global and personalized
        for idx, (p_per, p_glob) in enumerate(
            zip(self.model_per.parameters(), self.model.parameters())):
            p_per.data = (1 - self.alpha[idx]) * p_glob.data + self.alpha[idx] * p_per.data

        # update timing stats
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
                rep = model.base(x)
                out = model.head(rep)

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
                rep = self.model_per.base(x)
                out = self.model_per.head(rep)
                loss = self.loss(out, y)
                total += y.size(0)
                losses += loss.item() * y.size(0)

        return losses, total

    def alpha_update(self):
        for idx, (gl, gp) in enumerate(zip(self.model.parameters(),
                                       self.model_per.parameters())):
        
            if gp.grad is None or gl.grad is None:
                continue

        
            diff = (gp.detach() - gl.detach()).reshape(-1)
            grad = (self.alpha[idx] * gp.grad.detach() +
                    (1.0 - self.alpha[idx]) * gl.grad.detach()).reshape(-1)

       
            grad_alpha = float((diff * grad).sum().item())
        
            grad_alpha += 0.01 * float(self.alpha[idx])

        
            new_alpha = float(self.alpha[idx]) - float(self.learning_rate) * grad_alpha

        
            self.alpha[idx] = max(0.0, min(1.0, new_alpha))


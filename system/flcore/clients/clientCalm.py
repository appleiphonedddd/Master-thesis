import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientCalm(Client):
    """
    FedCALM client.
    Pattern mirrors clientDodm: keep a local personalized model (model_per)
    for evaluation, while uploading self.model back to server.
    - Step 1: update personalized head (freeze base) on model_per
    - Step 2: update shared/base parameters (freeze head) on self.model
    After local epochs, copy the learned shared/base from self.model into model_per
    so evaluation uses (shared base + personalized head).
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Personalized copy for evaluation/inference
        self.model_per = copy.deepcopy(self.model)

        # Optimizer for personalized head only
        if hasattr(self.model_per, "head"):
            self.optimizer_head = torch.optim.SGD(self.model_per.head.parameters(), lr=self.learning_rate)
        else:
            # If the model has no "head", fall back to optimizing all params in model_per for the head step
            self.optimizer_head = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

        # Optimizer for shared/base only (uploaded to server)
        if hasattr(self.model, "base"):
            self.optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        else:
            # If the model has no "base", fall back to optimizing the whole model
            self.optimizer_base = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Optional schedulers (keep semantics aligned with base class)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_head,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_scheduler_base = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_base,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def _freeze(self, module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()
        self.model_per.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2 + 1)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x_in = x[0]
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # ---------------------
                # Step 1: personalize the head on model_per (freeze base)
                # ---------------------
                if hasattr(self.model_per, "base"):
                    self._freeze(self.model_per.base, False)
                    with torch.no_grad():
                        rep = self.model_per.base(x_in)
                    # Only head gets gradients
                    self.model_per.head.train()
                    out_personal = self.model_per.head(rep)
                else:
                    # No explicit base/head split
                    out_personal = self.model_per(x_in)

                loss_head = self.loss(out_personal, y)
                self.optimizer_head.zero_grad()
                loss_head.backward()
                self.optimizer_head.step()

                # ---------------------
                # Step 2: update shared/base on self.model (freeze head)
                # ---------------------
                if hasattr(self.model, "head"):
                    self._freeze(self.model.head, False)  # keep head frozen
                # Only base optimizer will step
                out_global = self.model(x_in)
                loss_base = self.loss(out_global, y)
                self.optimizer_base.zero_grad()
                loss_base.backward()
                self.optimizer_base.step()

        # After local training, align shared/base in model_per with self.model
        if hasattr(self.model, "base") and hasattr(self.model_per, "base"):
            for p_src, p_tgt in zip(self.model.base.parameters(), self.model_per.base.parameters()):
                p_tgt.data = p_src.data.clone()
        else:
            # If no explicit split, keep full model_per = model for evaluation
            self.clone_model(self.model, self.model_per)

        if self.learning_rate_decay:
            self.learning_rate_scheduler_per.step()
            self.learning_rate_scheduler_base.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, model=None):
        # Evaluate using personalized model by default
        if model is None:
            model = self.model_per
        testloader = self.load_test_data()
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x_in = x[0]
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                if hasattr(self.model_per, "base"):
                    rep = self.model_per.base(x_in)
                    output = self.model_per.head(rep)
                else:
                    output = self.model_per(x_in)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                nc = self.num_classes + (0 if self.num_classes != 2 else 1)
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)


        y_prob = np.concatenate(y_prob, axis=0) if len(y_prob) else np.zeros((0, self.num_classes))
        y_true = np.concatenate(y_true, axis=0) if len(y_true) else np.zeros((0, self.num_classes))
        if len(y_true) and len(y_prob):
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        else:
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        # Report loss using personalized model_per
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x_in = x[0]
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                if hasattr(self.model_per, "base"):
                    rep = self.model_per.base(x_in)
                    output = self.model_per.head(rep)
                else:
                    output = self.model_per(x_in)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

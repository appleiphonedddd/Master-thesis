import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad

class clientCPD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.alpha = getattr(args, "cpd_alpha", 1.0)
        self.beta = getattr(args, "cpd_beta", 1.0)
        self.gamma = getattr(args, "cpd_gamma", 1.0)
        self.tau = getattr(args, "cpd_tau", 0.1)

        self.teacher_model = None
        self.global_protos = None
        self.local_protos = None

    def train(self, is_selected):
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()

            start_time = time.time()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)

                    if hasattr(self.model, "base"):
                        z_student = self.model.base(x)
                    else:
                        z_student = output

                    fd_loss = torch.tensor(0.0, device=self.device)
                    if self.teacher_model is not None:
                        self.teacher_model.eval()
                        with torch.no_grad():
                            if hasattr(self.teacher_model, "base"):
                                z_teacher = self.teacher_model.base(x)
                            else:
                                z_teacher = output.detach()
                        fd_loss = F.mse_loss(z_student, z_teacher)
                    
                    align_loss = torch.tensor(0.0, device=self.device)
                    pcl_loss = torch.tensor(0.0, device=self.device)
                    if self.global_protos is not None:
                        global_protos = self.global_protos.to(self.device)
                        proto_valid_mask = (global_protos.abs().sum(dim=1) > 0)

                        valid_idx = proto_valid_mask[y]
                        if valid_idx.any():
                            z_valid = z_student[valid_idx]
                            y_valid = y[valid_idx]
                    
                            proto_y = global_protos[y_valid]  # (B_valid, D)
                            align_loss = ((z_valid - proto_y) ** 2).mean()

                            sim = F.cosine_similarity(
                            z_valid.unsqueeze(1),  # (B_valid,1,D)
                            global_protos.unsqueeze(0),  # (1,C,D)
                            dim=-1)

                            pos = sim[torch.arange(sim.size(0), device=self.device), y_valid]
                            neg_mask = proto_valid_mask.unsqueeze(0).expand_as(sim)  # (B_valid, C)
                            neg_mask[torch.arange(sim.size(0), device=self.device), y_valid] = False
                            neg = sim[neg_mask].view(sim.size(0), -1)  # (B_valid, C-1)

                            pos_exp = torch.exp(pos / self.tau)  # (B_valid,)
                            neg_exp = torch.exp(neg / self.tau).sum(dim=1)  # (B_valid,)

                            pcl_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

                            loss = (loss+ self.alpha * align_loss+ self.beta * pcl_loss+ self.gamma * fd_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

            self.model.eval()
            self.compute_local_prototypes()
    
    def compute_local_prototypes(self):

        if not hasattr(self.model, "base"):
            self.local_protos = None
            return
        
        trainloader = self.load_train_data()
        feats_by_class = [[] for _ in range(self.num_classes)]

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in trainloader:
                if type(x_batch) == type([]):
                    x_batch[0] = x_batch[0].to(self.device)
                    feats = self.model.base(x_batch[0])
                else:
                    x_batch = x_batch.to(self.device)
                    feats = self.model.base(x_batch)
                y_batch = y_batch.to(self.device)

                for f, label in zip(feats, y_batch):
                    feats_by_class[label.item()].append(f.detach().clone())

        feat_dim = None
        for lst in feats_by_class:
            if len(lst) > 0:
                feat_dim = lst[0].numel()
                break

        if feat_dim is None:
            self.local_protos = None
            return

        protos = []
        for lst in feats_by_class:
            if len(lst) > 0:
                stack = torch.stack(lst, dim=0)
                protos.append(stack.mean(dim=0))
            else:
                protos.append(torch.zeros(feat_dim, device=self.device))

        self.local_protos = torch.stack(protos, dim=0)
    
    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
    
    def set_parameters(self, model, progress, global_protos):
        
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(self.model)
        else:
            
            self.clone_model(self.model, self.teacher_model)

        if hasattr(self.model, "base") and hasattr(model, "base"):
            for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
        else:
            
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

        self.global_protos = global_protos
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad

class clientFIP(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.alpha = args.alpha
        self.fim_trace_history = []

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
                    
                    if self.global_protos is not None and hasattr(self.model, "base"):
                        z = self.model.base(x)
                        global_protos = self.global_protos.to(self.device)
                        proto_valid_mask = (global_protos.abs().sum(dim=1) > 0)
                    
                        valid_idx = proto_valid_mask[y]
                        if valid_idx.any():
                            z_valid = z[valid_idx]
                            v_valid = y[valid_idx]
                            proto_y = global_protos[v_valid]
                            align_loss = align_loss = ((z_valid - proto_y) ** 2).mean()
                            loss = loss + self.alpha * align_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

            # set model to eval mode
            self.model.eval()

            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(self.load_train_data()):
                
                # Forward pass
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())
            self.compute_local_prototypes()

        else:
            trainloader = self.load_train_data()

            self.model.eval()

            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                
                # Forward pass
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())
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

    def set_parameters(self, model, progress,global_protos):

        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        batch_size = 16
        trainloader = self.load_train_data(batch_size=batch_size)

        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        # Align global model's prototype with the local prototype
        alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
        alignment_loss_fn = torch.nn.MSELoss()

        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = model.base(x_batch)
                loss = 0
                for label in y_batch.unique():
                    if mean_prototypes[label.item()] is not None:
                        loss += alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label.item()])
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

        self.global_protos = global_protos

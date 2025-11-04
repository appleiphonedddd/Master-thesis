import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad
from utils.INF import mark_only_lora_as_trainable

class clientINF(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []

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
                grads = grad(nll, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g, p in zip(grads, [p for p in self.model.parameters() if p.requires_grad]):
                    if g is not None:
                        fim_trace_sum += g.pow(2).sum().item()
            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum)

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
                grads = grad(nll, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g, p in zip(grads, [p for p in self.model.parameters() if p.requires_grad]):
                    if g is not None:
                        fim_trace_sum += g.pow(2).sum().item()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum)

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
    
    def set_parameters(self, model, progress):

        with torch.no_grad():
            for new_p, old_p in zip(model.base.parameters(), self.model.base.parameters()):
                old_p.data.copy_(new_p.data)

        self.model.train()
        align_loader = self.load_train_data(batch_size=16)
        mse = nn.MSELoss()

        align_opt = torch.optim.SGD(self.model.base.parameters(), lr=0.01)

        teacher_base = copy.deepcopy(self.model.base).to(self.device).eval()

        for _ in range(1):
            for xb, _ in align_loader:
                xb = xb.to(self.device)
                with torch.no_grad():
                    h_old = teacher_base(xb)
                h_new = self.model.base(xb)
                loss = mse(h_new, h_old)
                align_opt.zero_grad()
                loss.backward()
                align_opt.step()

        for p in self.model.head.parameters():
            p.requires_grad = True
        mark_only_lora_as_trainable(self.model.base, bias='lora_only')
        
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
    
        self.optimizer = torch.optim.SGD(optim_params, lr=self.learning_rate, momentum=0.9)

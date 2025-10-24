import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad
from utils.INF import INF

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
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

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
    
    def Linear(self, model, progress):

        # Apply LoRA to the model
        self.INF = INF(r = 4, lora_alpha = 16, lora_dropout = 0.1, merge_weights = True)

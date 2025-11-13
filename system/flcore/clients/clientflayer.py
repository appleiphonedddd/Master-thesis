import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import math
#import os
from flcore.clients.clientbase import Client
from utils.FLAYER import LocalAggregation

class clientFLayer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.local_learning_rate = args.local_learning_rate
        self.layer_idx = args.layer_idx
        self.model_before = None
        self.model = copy.deepcopy(args.model)

        self.model_str = getattr(args, "model_str", None)
        self.aggregate_params = []
        self.args = args

        base_params = [{'params': self.model.parameters(), 'lr': self.local_learning_rate}]

        params = base_params
        self.model_name = self.model.__class__.__name__
        
        if self.model_name == "FedAvgCNN":
            params = [
                {'params': list(self.model.parameters())[:2], 'lr': self.local_learning_rate},
                {'params': list(self.model.parameters())[2:4], 'lr': self.local_learning_rate * 30},
                {'params': list(self.model.parameters())[4:6], 'lr': self.local_learning_rate * 40},
                {'params': list(self.model.parameters())[6:8], 'lr': self.local_learning_rate / 5},
            ]
        
        self.optimizer = torch.optim.SGD(params)
        self.local_aggregation = LocalAggregation(self.layer_idx)

    def train(self, is_selected):
        self.model_before = copy.deepcopy(self.model)
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

                    if i == 0:
                        adaptive_lr = []
                        idx = 0
                        model_name = self.model.__class__.__name__
                        if model_name == "FedAvgCNN":
                            for name, layer in self.model.named_children():
                                grads = []
                                for param in layer.parameters():
                                    if param.grad is not None:
                                        grad_norm = param.grad.data.norm(2).item()
                                   
                                        grads.append(20 * self.local_learning_rate *
                                                 (1 + (idx / 3) * math.log(1 + 1 / grad_norm)))

                                if grads:
                                    idx += 1
                                    adaptive_lr.append(sum(grads) / len(grads))
            
                            if len(adaptive_lr) >= 3:
                                params = [
                            {'params': list(self.model.parameters())[:2], 'lr': self.learning_rate},
                            {'params': list(self.model.parameters())[2:4], 'lr': adaptive_lr[1]},
                            {'params': list(self.model.parameters())[4:6], 'lr': adaptive_lr[2]},
                            {'params': list(self.model.parameters())[6:8], 'lr': self.learning_rate / 5},
                            ]
                        #self.optimizer = torch.optim.SGD(params)
                        self.optimizer.step()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

            self.model.eval()
    
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

    def get_parameters_sparse(self, model_before, model):
        model_param = [val.data.cpu().numpy() for val in model.parameters()]
        model_change = [np.abs(val1.data.detach().cpu().numpy() - val2.data.detach().cpu().numpy())
                        for val1, val2 in zip(model.parameters(), model_before.parameters())]

        layer_len = len(model_param)
        for layer_id, (change_layer, param_layer) in enumerate(zip(model_change, model_param)):
            mask_ratio = 1 - max(min(layer_id / layer_len, 0.9), 0.1)
            if ((layer_len - 2) <= layer_id < layer_len):
                mask_ratio = 0.0

            # Check if the layer should be processed
            if mask_ratio != 0.0:
                if change_layer.ndim == 4:
                    kernel_num = change_layer.shape[2] ** 2
                    if mask_ratio >= (1 / kernel_num) and mask_ratio <= ((kernel_num - 1) / kernel_num):
                        prune = np.round(kernel_num * mask_ratio).astype(int)
                    
                    elif mask_ratio < (1 / kernel_num):
                        prune = 1
                    else:
                        prune = kernel_num - 1
                    
                    reshaped_layer = change_layer.reshape(change_layer.shape[0], change_layer.shape[1], -1)
                    sorted_indices = np.argsort(reshaped_layer, axis=-1)
                
                    # Determine the threshold index for each filter of each output channel
                    threshold_indices = sorted_indices[:, :, :prune]

                    # Create a mask for elements to zero out
                    mask = np.ones_like(reshaped_layer, dtype=bool)
                    np.put_along_axis(mask, threshold_indices, False, axis=-1)

                    # Apply mask to the original parameter layer, after reshaping the mask back
                    # param_layer[mask.reshape(param_layer.shape)] = 0
                    param_layer *= (mask.reshape(param_layer.shape))           

            elif change_layer.ndim == 1:
                element_num = change_layer.shape[0]

                if mask_ratio >= (1 / element_num) and mask_ratio <= ((element_num - 1) / element_num):
                    prune = np.round(element_num * mask_ratio).astype(int)
                elif mask_ratio < (1 / element_num):
                    prune = 1
                else:
                    prune = element_num - 1

                sorted_indices = np.argsort(change_layer, axis=-1)
                threshold_indices = sorted_indices[:prune]
                param_layer[threshold_indices] = 0
        
        return model_param
    
    def local_initialization(self, received_global_model, acc):
        self.local_aggregation.adaptive_local_aggregation(received_global_model, self.model, acc)

        trainloader = self.load_train_data()
        self.model.train()
        params = list(self.model.parameters())
        for param in params[:-self.layer_idx]:
            param.requires_grad = False

        for step in range(1):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        for param in params[:-self.layer_idx]:
            param.requires_grad = True
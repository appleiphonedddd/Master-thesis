import time
import copy
import numpy as np
import torch
from flcore.clients.clientflayer import clientFLayer
from flcore.servers.serverbase import Server

class FedFlayer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFLayer)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
    
    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_models(self, accs):
        assert (len(self.clients) > 0)

        for client, acc in zip(self.clients, accs):
            client.local_initialization(self.global_model, acc)

    def receive_models(self, accs):
        assert (len(self.selected_clients) > 0)

        self.aggregate_params = []
        self.uploaded_ids = []

        s_t = time.time()
        for client, acc in zip(self.selected_clients, accs):
            self.uploaded_ids.append(client.id)
            self.aggregate_params.append((client.get_parameters_sparse(client.model_before, client.model),
                                          client.train_samples))
    
        print("mask:")
        print('-' * 50, time.time() - s_t)
    
    def train(self):
        accs = [0.0] * self.num_clients
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(accs)
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                self.evaluate()

            self.send_selected_models(selected_ids, i)

            for client in self.alled_clients:
                client.train(client.id in selected_ids)
            
            self.receive_models(accs)

            self.aggregate_parameters()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        print(f'+++++++++++++++++++++++++++++++++++++++++')

        self.save_results()
        self.save_global_model()
    
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFLayer)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def aggregate_sparse(self, results):
        num_examples_total = sum([num_examples for _, num_examples in results]) 

        weighted_weights = [
            [layer * num_examples for layer in weights] 
            for weights, num_examples in results
        ]

        client_num_examples = np.array([num_examples for _, num_examples in results])
    
        weights_prime = []

        for layer_updates in zip(*weighted_weights):
            if layer_updates[0].ndim == 4 or layer_updates[0].ndim == 1:
                weight_matrix = np.add.reduce(layer_updates)
                num_total_matrix = np.full_like(layer_updates[0], num_examples_total)
                
                for client_id, layer in enumerate(layer_updates):
                    num_total_matrix[layer == 0] -= client_num_examples[client_id]
    
                weights_prime.append(np.divide(weight_matrix, num_total_matrix,
                    out=np.zeros_like(weight_matrix), where=num_total_matrix != 0))
            else:
                weight_matrix = np.add.reduce(layer_updates)
                weights_prime.append(weight_matrix / num_examples_total)
        
        return weights_prime
    
    def set_parameters(self, model, parameters):
        for new_param, old_param in zip(parameters, model.parameters()):
            old_param.data = torch.tensor(new_param, dtype=torch.float).to(self.device)
    
    def get_parameters(self, model):
        return [val.data.cpu().numpy() for val in model.parameters()]

    def aggregate_parameters(self):

        parameters_lastround = self.get_parameters(self.global_model)
        s_t = time.time()
        s_t1 = time.time()

        parameters_thisround_sparse_np = self.aggregate_sparse(self.aggregate_params)

        print("aggregate: ")
        print('-' * 50, time.time() - s_t)

        parameters_thisround = [np.where(layer == 0, Layer, layer)
            for layer, Layer in zip(parameters_thisround_sparse_np, parameters_lastround)]

        print("fill zero:")
        print('-' * 50, time.time() - s_t1)

        self.set_parameters(self.global_model, parameters_thisround)

        del parameters_lastround, parameters_thisround, parameters_thisround_sparse_np
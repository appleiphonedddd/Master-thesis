import time
import copy
import numpy as np
import torch
from flcore.clients.clientcalm import clientCALM
from flcore.servers.serverbase import Server

class FedCALM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.calm_epsilon = args.calm_epsilon
        self.calm_C = args.calm_C
        self.calm_lr = args.calm_lr
        self.calm_steps = args.calm_steps

        self.set_slow_clients()
        self.set_clients(clientCALM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
    
    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = epoch / self.global_rounds
            
            client.set_parameters(copy.deepcopy(self.global_model), progress)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_calm(self):
        if len(self.uploaded_models) == 0:
            return
        
        old_global = copy.deepcopy(self.global_model)
        with torch.no_grad():
            global_params = [p for p in self.global_model.parameters()]
            old_global_params = [p for p in old_global.parameters()]
            client_params_list = [[p for p in m.parameters()] for m in self.uploaded_models]

            num_clients = len(client_params_list)
            eps = 1e-12

            for layer_idx, (g_param, g_old_param) in enumerate(zip(global_params, old_global_params)):
                g_old = g_old_param.data
                updates = []
                for c in range(num_clients):
                    local_param = client_params_list[c][layer_idx].data
                    updates.append((local_param - g_old).view(-1))
                updates = torch.stack(updates)

                if torch.allclose(updates, torch.zeros_like(updates)):
                    continue
                
                v = updates.clone()
                for i in range(num_clients):
                    vi = v[i]
                    for j in range(num_clients):
                        if i == j:
                            continue
                        vj = updates[j]

                        dot_ij = torch.dot(vi, vj)
                        norm_i = torch.norm(vi) + eps
                        norm_j = torch.norm(vj) + eps
                        cos_ij = dot_ij / (norm_i * norm_j)

                        if cos_ij < 0:
                            proj_coeff = dot_ij / (norm_j ** 2 + eps)
                            vi = vi - proj_coeff * vj

                    v[i] = vi

                delta_perp = v.mean(dim=0)

                norms = torch.norm(updates, dim=1) + eps
                dot_avg = torch.mv(updates, delta_perp)
                G = updates @ updates.t()

                lambda_vec = torch.zeros(num_clients, device=updates.device)

                for _ in range(self.calm_steps):
                    grad = self.calm_epsilon * norms - dot_avg - torch.mv(G, lambda_vec)
                    lambda_vec = lambda_vec + self.calm_lr * grad
                    lambda_vec = torch.clamp(lambda_vec, 0.0, self.calm_C)

                d_star = delta_perp + torch.mv(updates.t(), lambda_vec)

                g_param.data = g_old + d_star.view_as(g_param.data)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                self.evaluate()

            self.send_selected_models(selected_ids, i)

            for client in self.alled_clients:
                client.train(client.id in selected_ids)
            
            self.receive_models()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            self.aggregate_calm()
            
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
            self.set_new_clients(clientCALM)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
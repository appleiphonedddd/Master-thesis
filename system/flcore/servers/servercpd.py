import time
import copy
import numpy as np
import torch 
from flcore.clients.clientcpd import clientCPD
from flcore.servers.serverbase import Server

class FedCPD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientCPD)
        self.global_protos = None
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
            
            client.set_parameters(
                copy.deepcopy(self.global_model),
                progress,
                self.global_protos
            )

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

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

            if len(self.uploaded_models) > 0:
                self.aggregate_parameters()

            self.aggregate_prototypes()
            
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
            self.set_new_clients(clientCPD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def aggregate_prototypes(self):

        if len(self.uploaded_ids) == 0:
            return

        template = None
        for cid in self.uploaded_ids:
            lp = getattr(self.clients[cid], "local_protos", None)
            if lp is not None:
                template = lp
                break

        if template is None:
            # 這輪剛好沒有任何 client 回傳 prototype
            return

        device = template.device
        num_classes, feat_dim = template.size()

        sum_protos = torch.zeros(num_classes, feat_dim, device=device)
        cnt_protos = torch.zeros(num_classes, device=device)

        for cid in self.uploaded_ids:
            lp = getattr(self.clients[cid], "local_protos", None)
            if lp is None:
                continue
            lp = lp.to(device)
            
            mask = (lp.abs().sum(dim=1) > 0)
            sum_protos[mask] += lp[mask]
            cnt_protos[mask] += 1.0

        cnt_expand = cnt_protos.clamp_min(1.0).view(-1, 1)
        self.global_protos = sum_protos / cnt_expand

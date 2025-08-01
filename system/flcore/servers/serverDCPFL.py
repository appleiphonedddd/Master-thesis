import time
import random
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler  # ### NEW
from flcore.clients.clientDCPFL import clientDCPFL
from flcore.servers.serverbase import Server

# ===================== #
#  Update Embedding 工具 #
# ===================== #
# ### NEW: 省記憶體的嵌入做法（hybrid = layerstats + countsketch）
@torch.no_grad()
def _flatten_params_cpu_f32(model):
    vecs = []
    for p in model.parameters():
        v = p.detach().view(-1).cpu().to(torch.float32)
        if v.numel() > 0:
            vecs.append(v)
    if not vecs:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(vecs, dim=0)

@torch.no_grad()
def _delta_vec(global_model, client_model):
    g = _flatten_params_cpu_f32(global_model)
    c = _flatten_params_cpu_f32(client_model)
    # 假設參數拓撲一致
    return c - g  # torch.float32 CPU

def _hash32(x, seed):
    x = (np.uint64(x) ^ np.uint64(seed)) * np.uint64(0x9e3779b97f4a7c15)
    x ^= (x >> np.uint64(33))
    x *= np.uint64(0xff51afd7ed558ccd)
    x ^= (x >> np.uint64(33))
    x *= np.uint64(0xc4ceb9fe1a85ec53)
    x ^= (x >> np.uint64(33))
    return int(x & np.uint64(0xffffffff))

def _countsketch(delta: torch.Tensor, k=512, seed=42):
    y = np.zeros(k, dtype=np.float32)
    n = delta.numel()
    a1, b1 = 1103515245 + seed, 12345 + seed * 3
    a2, b2 = 1402946737 + seed * 5, 2463534242 + seed * 7
    # 流式處理，O(d) 時間、O(k) 記憶體
    for i in range(n):
        h = (_hash32(a1 * (i + 1) + b1, seed) % k)
        s = -1.0 if (_hash32(a2 * (i + 1) + b2, seed) & 1) else 1.0
        y[h] += s * float(delta[i])
    # 正規化
    norm = np.linalg.norm(y) + 1e-8
    y /= norm
    return y  # (k,)

def _layerwise_stats(global_model, client_model):
    feats = []
    for (pg, pc) in zip(global_model.parameters(), client_model.parameters()):
        dg = (pc.detach().cpu().to(torch.float32) - pg.detach().cpu().to(torch.float32)).view(-1)
        if dg.numel() == 0:
            continue
        n = float(dg.numel())
        l2 = float(torch.linalg.norm(dg) / (n ** 0.5))
        mean = float(dg.mean())
        std = float(dg.std(unbiased=False))
        maxabs = float(dg.abs().max())
        feats.extend([l2, mean, std, maxabs])
    return np.asarray(feats, dtype=np.float32) if feats else np.zeros(4, dtype=np.float32)

def _update_embedding(global_model, client_model, mode="hybrid", k=512, seed=42):
    d = _delta_vec(global_model, client_model)
    if mode == "layerstats":
        return _layerwise_stats(global_model, client_model)
    if mode == "countsketch":
        return _countsketch(d, k=k, seed=seed)
    if mode == "hybrid":
        a = _layerwise_stats(global_model, client_model)
        b = _countsketch(d, k=k, seed=seed)
        return np.concatenate([a, b], axis=0)
    raise ValueError(f"Unknown embedding mode: {mode}")

class DCPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients and initialize clientDCPFL instances
        self.set_slow_clients()
        self.set_clients(clientDCPFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # record historical loss for each client
        self.loss_history = {i: [] for i in range(self.num_clients)}
        # record time cost for each round
        self.Budget = []

        # ### NEW: 可由 argparse 傳入，否則用預設
        self.embed_mode = getattr(self.args, "embed_mode", "hybrid")
        self.embed_dim = int(getattr(self.args, "embed_dim", 512))
        self.embed_seed = int(getattr(self.args, "embed_seed", 42))

    def train(self):
        def get_dbscan_hparams(args):
            if hasattr(args, 'eps'):
                eps = args.eps
            elif hasattr(args, 'dbscan_eps'):
                eps = args.dbscan_eps
            else:
                eps = 0.75

            if hasattr(args, 'minpts'):
                minpts = args.minpts
            elif hasattr(args, 'dbscan_minpts'):
                minpts = args.dbscan_minpts
            else:
                minpts = 5

            try:
                eps = float(eps)
            except Exception:
                print(f"[WARN] Invalid eps={eps}, fallback to 0.75")
                eps = 0.75

            try:
                minpts = int(minpts)
            except Exception:
                print(f"[WARN] Invalid minpts={minpts}, fallback to 5")
                minpts = 5

            if eps <= 0:
                print(f"[WARN] eps <= 0 ({eps}), force to 0.75")
                eps = 0.75
            if minpts < 1:
                print(f"[WARN] minpts < 1 ({minpts}), force to 5")
                minpts = 5

            return eps, minpts
        
        for i in range(self.global_rounds + 1):
            start_time = time.time()

            # 1. select clients and distribute global model
            self.selected_clients = self.select_clients()
            self.send_models()

            # 2. periodic evaluation
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            # 3. local training on selected clients
            for client in self.selected_clients:
                client.train()

            # 4. collect local models
            self.receive_models()

            # 如果本輪沒有任何上傳，跳過聚合
            if len(getattr(self, "uploaded_models", [])) == 0:
                print("[WARN] No active clients uploaded this round. Skip clustering/aggregation.")
                # 記錄時間並進下一輪
                elapsed = time.time() - start_time
                self.Budget.append(elapsed)
                print('-' * 25, 'time cost', '-' * 25, elapsed)
                continue

            # 5. perform DBSCAN clustering on **compressed update embeddings**  ### NEW
            eps, minpts = get_dbscan_hparams(self.args)

            features = []
            for model in self.uploaded_models:
                z = _update_embedding(
                    global_model=self.global_model,
                    client_model=model,
                    mode=self.embed_mode,
                    k=self.embed_dim,
                    seed=self.embed_seed
                )
                features.append(z.astype(np.float32))
            X = np.vstack(features)
            # z-score 對 DBSCAN 很重要
            X = StandardScaler().fit_transform(X)

            # 太少點或 minpts 太大時，直接單群 fallback
            if len(self.uploaded_models) < max(2, minpts):
                labels = np.zeros(len(self.uploaded_models), dtype=int)
            else:
                clustering = DBSCAN(eps=eps, min_samples=minpts, metric="euclidean").fit(X)
                labels = clustering.labels_
                # 若全是 outlier 或只有 1 群，fallback
                if (labels == -1).all() or (len(set(labels)) <= 1):
                    print("[INFO] DBSCAN returned <=1 cluster or all outliers; fallback to single cluster.")
                    labels = np.zeros(len(self.uploaded_models), dtype=int)

            # 6. model exchange based on weighted moving average (WMA) of loss
            for idx, cid in enumerate(self.uploaded_ids):
                client = next(c for c in self.clients if c.id == cid)
                loss, _ = client.train_metrics()
                self.loss_history[cid].append(loss)

                history = self.loss_history[cid]
                if len(history) >= 3:
                    wma = (0.1 * history[-3] + 0.1 * history[-2] + 0.8 * history[-1])
                    if history[-1] < wma:
                        same_cluster = [
                            j for j, lab in enumerate(labels) if lab == labels[idx] and j != idx
                        ]
                        partner = random.choice(same_cluster) if same_cluster else random.choice(
                            [j for j in range(len(labels)) if j != idx]
                        )
                        # swap models with selected partner
                        self.uploaded_models[idx], self.uploaded_models[partner] = (
                            self.uploaded_models[partner],
                            self.uploaded_models[idx],
                        )
                        print(f"Client {cid} exchanged model with client {self.uploaded_ids[partner]}")

            # 7. aggregate parameters via FedAvg
            self.aggregate_parameters()

            # 8. DLG evaluation if configured
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # 9. record and print time cost for this round
            elapsed = time.time() - start_time
            self.Budget.append(elapsed)
            print('-' * 25, 'time cost', '-' * 25, elapsed)

            # 10. early stopping check
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # print best accuracy and average time cost
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # save results and final global model
        self.save_results()
        self.save_global_model()

        # fine-tune and evaluate new clients if any
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDCPFL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert len(self.selected_clients) > 0

        # randomly drop clients according to drop rate
        self.active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0

        per_class = []
        for client in self.active_clients:
            try:
                avg_train_time = (
                    client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds']
                    + client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                )
            except ZeroDivisionError:
                avg_train_time = 0

            if avg_train_time <= self.time_threthold:
                total_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                per_class.append(client.sample_per_class)

        # 沒有任何上傳：保持欄位一致、直接返回  ### NEW
        if len(self.uploaded_ids) == 0:
            self.uploaded_samples = [0 for _ in range(self.num_classes)]
            self.uploaded_class_weight = [np.array([], dtype=np.float32) for _ in range(self.num_classes)]
            return

        # normalize sample-based weights  ### NEW: 避免 total_samples==0
        if total_samples > 0:
            self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        else:
            # 平均分配
            avg_w = 1.0 / len(self.uploaded_weights)
            self.uploaded_weights = [avg_w for _ in self.uploaded_weights]

        # compute class weights  ### NEW: 魯棒化處理
        self.uploaded_samples = []
        self.uploaded_class_weight = []
        per_class = np.asarray(per_class, dtype=np.float32)  # shape: [num_active, num_classes]

        for c in range(self.num_classes):
            class_weights = per_class[:, c] if per_class.size > 0 else np.array([], dtype=np.float32)
            s = float(np.sum(class_weights)) if class_weights.size > 0 else 0.0
            self.uploaded_samples.append(s)
            if s > 0 and class_weights.size > 0:
                self.uploaded_class_weight.append(class_weights / s)
            else:
                # 平均分配或空
                if class_weights.size > 0:
                    self.uploaded_class_weight.append(
                        np.full_like(class_weights, 1.0 / len(class_weights), dtype=np.float32)
                    )
                else:
                    self.uploaded_class_weight.append(np.array([], dtype=np.float32))

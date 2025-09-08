import copy
import time
import random
import torch
import numpy as np

from flcore.clients.clientCalm import clientCalm
from flcore.servers.serverbase import Server


def _flatten_tensor(t: torch.Tensor):
    return t.view(-1)


def _dot(a: torch.Tensor, b: torch.Tensor):
    return torch.dot(a, b)


def _norm(a: torch.Tensor, eps=1e-12):
    return torch.norm(a) + eps


class FedCALM(Server):
    """
    Server implementing FedCALM aggregation.
    Two-stage per-layer procedure:
      1) Project conflicting client updates away (orthogonalize pairs with negative cosine similarity).
      2) Solve a small QP in the dual (approximated) to get d* = Δθ_perp + Σ λ_i Δθ_i, with 0 ≤ λ_i ≤ C.

    Notes:
    - We aggregate only the shared/base parameters if 'base' exists, else all params.
    - We compute client updates as Δθ_i = (θ_i - θ_G_prev).
    """
    def __init__(self, args, times):
        super().__init__(args, times)

        # hyper-parameters for CALM (defaults if not provided)
        self.calm_eps = getattr(args, "calm_eps", 0.01)   # ε in the paper
        self.calm_C = getattr(args, "calm_C", 1.0)        # C in the paper (box constraint)
        self.calm_ridge = getattr(args, "calm_ridge", 1e-8)

        # select slow clients and instantiate
        self.set_slow_clients()
        self.set_clients(clientCalm)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def _param_groups_base_or_all(self, model):
        """Return a list of parameter tensors for model.base if present, else model.parameters()."""
        if hasattr(model, "base"):
            return list(model.base.parameters())
        return list(model.parameters())

    def _gather_layerwise_updates(self, prev_global_params, client_models):
        """
        Build per-layer updates:
            deltas[k][i] = flattened Δθ_i^(k)
        where k indexes parameter tensors (layers) and i indexes clients.
        """
        num_layers = len(prev_global_params)
        deltas = [[] for _ in range(num_layers)]
        for cm in client_models:
            cm_params = self._param_groups_base_or_all(cm)
            for k, (pg, pc) in enumerate(zip(prev_global_params, cm_params)):
                d = _flatten_tensor(pc.data - pg.data).detach().to(self.device)
                deltas[k].append(d)
        return deltas

    def _project_conflicts(self, vecs):
        """
        Given a list of vectors [d1, d2, ...], return the conflict-free average Δθ_perp
        using the projection rule: for each i, subtract projections onto j with cos<0, then average.
        """
        m = len(vecs)
        if m == 0:
            return None
        # Precompute norms
        norms = [float(_norm(v)) for v in vecs]
        adjusted = []
        for i in range(m):
            vi = vecs[i].clone()
            for j in range(m):
                if i == j:
                    continue
                vj = vecs[j]
                denom = float(_norm(vj)) ** 2
                if denom <= 0:
                    continue
                cos_ij = float(_dot(vecs[i], vj) / (_norm(vecs[i]) * _norm(vj)))
                if cos_ij < 0.0:
                    coeff = float(_dot(vi, vj)) / denom
                    vi = vi - coeff * vj
            adjusted.append(vi)
        # average
        avg = sum(adjusted) / float(m)
        return avg

    def _solve_box_qp_dual(self, G, b, C, ridge):
        """
        Solve (approximately) max_λ  b^T λ - 0.5 λ^T G λ,  s.t. 0 ≤ λ ≤ C
        where b_i = ε||Δθ_i|| - Δθ_perp^T Δθ_i  and  G_ij = Δθ_i^T Δθ_j.

        Strategy:
        - Solve (G + ridge I) λ = b, then clip to [0, C].
        - Do a small number of projected gradient refinement steps.
        """
        m = G.shape[0]
        if m == 0:
            return torch.zeros(0, device=G.device)
        I = torch.eye(m, device=G.device, dtype=G.dtype)
        G_reg = G + ridge * I
        try:
            lam = torch.linalg.solve(G_reg, b)
        except RuntimeError:
            # fall back to least-squares
            lam = torch.linalg.lstsq(G_reg, b).solution
        lam = torch.clamp(lam, 0.0, C)

        # One or two projected gradient steps to reduce KKT residuals
        # grad = b - G @ lam
        for _ in range(2):
            grad = b - G @ lam
            # step size (diagonal precondition / Lipschitz approx)
            L = torch.maximum(torch.sum(torch.abs(G), dim=1), torch.tensor(1e-6, device=G.device))
            step = grad / L
            lam = torch.clamp(lam + step, 0.0, C)
        return lam

    def aggregate_parameters(self):
        """CALM aggregation over the uploaded models."""
        assert (len(self.uploaded_models) > 0)

        # Snapshot previous global shared/base parameters
        prev_global_params_full = self._param_groups_base_or_all(self.prev_global_model)
        # Per-client models (only shared/base params will be used)
        client_models = self.uploaded_models

        # Build layer-wise client updates
        deltas = self._gather_layerwise_updates(prev_global_params_full, client_models)

        # Prepare new global params container
        new_params = []
        device = self.device

        eps = float(self.calm_eps)
        C = float(self.calm_C)
        ridge = float(self.calm_ridge)

        # For each "layer" (parameter tensor), compute d* and update
        for k, (pg) in enumerate(prev_global_params_full):
            vecs = deltas[k]  # list of Δθ_i^(k) as flat tensors
            m = len(vecs)
            if m == 0:
                new_params.append(pg.data.clone())
                continue

            # Stage 1: conflict-free average
            delta_perp = self._project_conflicts(vecs)

            # Build Gram matrix and b
            # G_ij = di^T dj ;  b_i = ε||di|| - delta_perp^T di
            G = torch.zeros((m, m), device=device, dtype=pg.data.dtype)
            b = torch.zeros((m,), device=device, dtype=pg.data.dtype)
            for i in range(m):
                di = vecs[i]
                b[i] = eps * _norm(di) - _dot(delta_perp, di)
                for j in range(i, m):
                    val = _dot(di, vecs[j])
                    G[i, j] = val
                    G[j, i] = val

            # Stage 2: conflict-aware mitigation via approximate dual QP
            lam = self._solve_box_qp_dual(G, b, C=C, ridge=ridge)

            # d* = delta_perp + sum_i λ_i * di
            d_star = delta_perp.clone()
            for i in range(m):
                d_star = d_star + lam[i] * vecs[i]

            # reshape back to parameter tensor shape
            new_param = pg.data + d_star.view_as(pg.data)
            new_params.append(new_param)

        # Write back into the *global model's* shared/base parameters;
        # leave other parts (e.g., head) as previous global to avoid overwriting personalized heads.
        if hasattr(self.global_model, "base"):
            # Update only base
            for p, new_p in zip(self.global_model.base.parameters(), new_params):
                p.data = new_p.clone()
        else:
            # Update all
            for p, new_p in zip(self.global_model.parameters(), new_params):
                p.data = new_p.clone()

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Keep a copy of current global BEFORE sending to clients for delta computation
            self.prev_global_model = copy.deepcopy(self.global_model)

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # receive uploaded client models (subset / drop rate handled inside)
            self.receive_models()

            # Optionally privacy evaluation etc.
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # CALM aggregation
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if len(self.rs_test_acc) else None)
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print(0.0)

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCalm)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

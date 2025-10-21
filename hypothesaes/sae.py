import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArchetypalDictionary(nn.Module):
    def __init__(self, C: torch.Tensor, k: int, delta: float = 1.0):
        super().__init__()
        self.register_buffer("C", C)
        self.W = nn.Parameter(torch.eye(k, C.shape[0]))
        self.Lambda = nn.Parameter(torch.zeros(k, C.shape[1]))
        self.delta = delta

    @torch.no_grad()
    def _project(self):
        W = torch.relu(self.W)
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)
        self.W.data = W
        if self.delta is not None and self.delta > 0:
            norms = self.Lambda.norm(dim=-1, keepdim=True) + 1e-12
            scale = torch.clamp(self.delta / norms, max=1.0)
            self.Lambda.mul_(scale)

    def dictionary(self) -> torch.Tensor:
        self._project()
        return self.W @ self.C + self.Lambda

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        D = self.dictionary()
        return Z @ D

class SparseAutoencoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        m_total_neurons: int | list[int],
        k_active_neurons: int,
        aux_k: int | None = None, 
        multi_k: int | None = None,
        dead_neuron_threshold_steps: int = 256,
        decoder_type: str = "free",
        archetypal_C: torch.Tensor | None = None,
        ra_delta: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.aux_k = aux_k
        self.multi_k = multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps
        self.k_active_neurons = k_active_neurons if isinstance(k_active_neurons, int) else max(k_active_neurons)
        if isinstance(m_total_neurons, list):
            self.matryoshka_sizes = sorted(m_total_neurons)
            self.m_max = self.matryoshka_sizes[-1]
            self.is_matryoshka = True
        else:
            self.matryoshka_sizes = [m_total_neurons]
            self.m_max = m_total_neurons
            self.is_matryoshka = False
        self.m_total_neurons = self.m_max
        if self.aux_k is None:
            self.aux_k = 2 * self.k_active_neurons
        if self.multi_k is None:
            self.multi_k = 4 * self.k_active_neurons
        self.encoder = nn.Linear(input_dim, self.m_max, bias=False)
        self.decoder_type = decoder_type
        if self.decoder_type == "free":
            self.decoder = nn.Linear(self.m_max, input_dim, bias=False)
        else:
            if archetypal_C is None:
                raise ValueError("archetypal_C required when decoder_type='archetypal'")
            self.archetypal = ArchetypalDictionary(archetypal_C.to(device), self.m_max, delta=ra_delta)
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(self.m_max))
        self.steps_since_activation = torch.zeros(self.m_max, dtype=torch.long, device=device)
        self.to(device)

    def forward(self, x: torch.Tensor, track_dead: bool = True):
        x = x - self.input_bias
        pre_act = self.encoder(x) + self.neuron_bias
        topk_values, topk_indices = torch.topk(pre_act, k=self.k_active_neurons, dim=-1)
        topk_values = F.relu(topk_values)
        multik_values, multik_indices = torch.topk(pre_act, k=self.multi_k, dim=-1)
        multik_values = F.relu(multik_values)
        activations = torch.zeros_like(pre_act)
        activations.scatter_(-1, topk_indices, topk_values)
        multik_activations = torch.zeros_like(pre_act)
        multik_activations.scatter_(-1, multik_indices, multik_values)
        if track_dead:
            self.steps_since_activation += 1
            self.steps_since_activation.scatter_(0, topk_indices.unique(), 0)
        if self.decoder_type == "free":
            reconstruction = self.decoder(activations) + self.input_bias
            multik_reconstruction = self.decoder(multik_activations) + self.input_bias
        else:
            reconstruction = self.archetypal(activations) + self.input_bias
            multik_reconstruction = self.archetypal(multik_activations) + self.input_bias
        matryoshka_recons = None
        if self.is_matryoshka and len(self.matryoshka_sizes) > 1:
            matryoshka_recons = {}
            for m_i in self.matryoshka_sizes[:-1]:
                partial_acts = activations.clone()
                partial_acts[:, m_i:] = 0.0
                if self.decoder_type == "free":
                    matryoshka_recons[m_i] = self.decoder(partial_acts) + self.input_bias
                else:
                    matryoshka_recons[m_i] = self.archetypal(partial_acts) + self.input_bias
        aux_values, aux_indices = None, None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_neuron_pre_act = pre_act * dead_mask
            aux_values, aux_indices = torch.topk(dead_neuron_pre_act, k=self.aux_k, dim=-1)
            aux_values = F.relu(aux_values)
        return reconstruction, {
            "activations": activations,
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_reconstruction": multik_reconstruction,
            "aux_indices": aux_indices,
            "aux_values": aux_values,
            "matryoshka_recons": matryoshka_recons
        }

    def reconstruct_input_from_latents(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        activations = torch.zeros(self.m_total_neurons, device=indices.device)
        activations.scatter_(-1, indices, values)
        if self.decoder_type == "free":
            return self.decoder(activations) + self.input_bias
        else:
            return self.archetypal(activations) + self.input_bias

    def normalize_decoder_(self):
        if self.decoder_type == "free":
            with torch.no_grad():
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-12)

    def adjust_decoder_gradient_(self):
        if self.decoder_type == "free" and self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = torch.sum(self.decoder.weight * self.decoder.weight.grad, dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor):
        with torch.no_grad():
            self.input_bias.data = torch.median(data_sample, dim=0).values
            if self.decoder_type == "free":
                nn.init.xavier_uniform_(self.decoder.weight)
                self.normalize_decoder_()
                self.encoder.weight.data = self.decoder.weight.t().clone()
            else:
                nn.init.xavier_uniform_(self.encoder.weight)
            nn.init.zeros_(self.neuron_bias)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.is_matryoshka:
            m_total_neurons_config = self.matryoshka_sizes
        else:
            m_total_neurons_config = self.m_total_neurons
        config = {
            'input_dim': self.input_dim,
            'm_total_neurons': m_total_neurons_config, 
            'k_active_neurons': self.k_active_neurons,
            'aux_k': self.aux_k,
            'multi_k': self.multi_k,
            'dead_neuron_threshold_steps': self.dead_neuron_threshold_steps,
            'decoder_type': self.decoder_type
        }
        if self.decoder_type != "free":
            C = self.archetypal.C
            config['ra_delta'] = float(self.archetypal.delta)
            config['ra_n_prototypes'] = int(C.shape[0])
            config['ra_dim'] = int(C.shape[1])
        torch.save({'config': config, 'state_dict': self.state_dict()}, save_path, pickle_module=pickle)
        return save_path

    def compute_loss(self, x: torch.Tensor, recon: torch.Tensor, info: dict, aux_coef: float = 1/32, multi_coef: float = 0.0) -> torch.Tensor:
        def normalized_mse(pred, target):
            mse = F.mse_loss(pred, target)
            baseline_mse = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
            return mse / (baseline_mse + 1e-9)
        recon_loss = normalized_mse(recon, x)
        recon_loss += multi_coef * normalized_mse(info["multik_reconstruction"], x)
        if self.is_matryoshka and info["matryoshka_recons"] is not None:
            matryoshka_loss = 0.0
            for _, partial_recon in info["matryoshka_recons"].items():
                matryoshka_loss += normalized_mse(partial_recon, x)
            recon_loss += matryoshka_loss
        if self.aux_k is not None:
            error = x - recon.detach()
            aux_activations = torch.zeros_like(info["activations"])
            aux_activations.scatter_(-1, info["aux_indices"], info["aux_values"])
            if self.decoder_type == "free":
                error_reconstruction = self.decoder(aux_activations)
            else:
                error_reconstruction = self.archetypal(aux_activations)
            aux_loss = normalized_mse(error_reconstruction, error)
            total_loss = recon_loss + aux_coef * aux_loss
        else:
            total_loss = recon_loss
        return total_loss

    def fit(self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, save_dir: str | None = None, batch_size: int = 512, learning_rate: float = 5e-4, n_epochs: int = 200, aux_coef: float = 1/32, multi_coef: float = 0.0, patience: int = 5, show_progress: bool = True, clip_grad: float = 1.0):
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = None
        if X_val is not None:
            val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)
        self.initialize_weights_(X_train.to(device))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'dead_neuron_ratio': []}
        iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
        for epoch in iterator:
            self.train()
            train_losses = []
            for batch_x, in train_loader:
                batch_x = batch_x.to(device)
                recon, info = self(batch_x)
                loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                optimizer.zero_grad()
                loss.backward()
                self.adjust_decoder_gradient_()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()
                self.normalize_decoder_()
                train_losses.append(loss.item())
            avg_train_loss = float(np.mean(train_losses))
            history['train_loss'].append(avg_train_loss)
            dead_ratio = (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
            history['dead_neuron_ratio'].append(dead_ratio)
            avg_val_loss = None
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, in val_loader:
                        batch_x = batch_x.to(device)
                        recon, info = self(batch_x, track_dead=False)
                        val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                history['val_loss'].append(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            if show_progress:
                iterator.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}' if avg_val_loss else 'N/A', 'dead_ratio': f'{dead_ratio:.3f}'})
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            m_param = self.matryoshka_sizes if self.is_matryoshka else self.m_total_neurons
            def _fmt(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return "-".join(str(int(i)) for i in x)
                return str(int(x))
            m_str = _fmt(m_param)
            k_str = _fmt(self.k_active_neurons)
            dtype = "RA" if self.decoder_type != "free" else "Free"
            filename = f"SAE_{dtype}_M={m_str}_K={k_str}.pt"
            self.save(os.path.join(save_dir, filename))
        return history

    def get_activations(self, inputs, batch_size=16384):
        self.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        inputs = inputs.to(device)
        num_samples = inputs.shape[0]
        all_activations = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = inputs[i:i+batch_size]
                _, info = self(batch, track_dead=False)
                all_activations.append(info['activations'].cpu())
        return torch.cat(all_activations, dim=0).numpy()

    def get_dictionary(self) -> np.ndarray:
        if self.decoder_type == "free":
            D = self.decoder.weight.data.t()
        else:
            D = self.archetypal.dictionary().data
        D = F.normalize(D, p=2, dim=1)
        return D.detach().cpu().numpy()

def load_model(path: str) -> SparseAutoencoder:
    checkpoint = torch.load(path, pickle_module=pickle, map_location=device)
    sd = checkpoint['state_dict']
    cfg = checkpoint['config']
    if 'decoder_type' in cfg and cfg['decoder_type'] != 'free':
        n_prime = cfg.get('ra_n_prototypes', None)
        d = cfg.get('ra_dim', cfg['input_dim'])
        if n_prime is None:
            for kname, tensor in sd.items():
                if kname.endswith('archetypal.C'):
                    n_prime, d = tensor.shape
                    break
        C_placeholder = torch.zeros(n_prime, d)
        model = SparseAutoencoder(
            input_dim=cfg['input_dim'],
            m_total_neurons=cfg['m_total_neurons'],
            k_active_neurons=cfg['k_active_neurons'],
            aux_k=cfg['aux_k'],
            multi_k=cfg['multi_k'],
            dead_neuron_threshold_steps=cfg['dead_neuron_threshold_steps'],
            decoder_type='archetypal',
            archetypal_C=C_placeholder,
            ra_delta=cfg.get('ra_delta', 1.0)
        )
    else:
        model = SparseAutoencoder(
            input_dim=cfg['input_dim'],
            m_total_neurons=cfg['m_total_neurons'],
            k_active_neurons=cfg['k_active_neurons'],
            aux_k=cfg['aux_k'],
            multi_k=cfg['multi_k'],
            dead_neuron_threshold_steps=cfg['dead_neuron_threshold_steps'],
            decoder_type='free'
        )
    model.load_state_dict(sd, strict=True)
    return model.to(device)

def dictionary_from_model(model: SparseAutoencoder) -> np.ndarray:
    return model.get_dictionary()

def stability_score(D1: np.ndarray, D2: np.ndarray) -> float:
    A = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-12)
    B = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    cost = -S 
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(S[row_ind, col_ind].mean())

def average_pairwise_stability(dicts: list[np.ndarray]) -> float:
    n = len(dicts)
    if n < 2:
        return 1.0
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(stability_score(dicts[i], dicts[j]))
    return float(np.mean(vals))

def get_multiple_sae_activations(sae_list, X, return_neuron_source_info=False):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float().to(device)
    activations_list = []
    neuron_source_sae_info = []
    for s in sae_list:
        activations_list.append(s.get_activations(X))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    if return_neuron_source_info:
        return activations, neuron_source_sae_info
    else:
        return activations

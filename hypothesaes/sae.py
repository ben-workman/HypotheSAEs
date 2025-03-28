"""Sparse autoencoder implementation and training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, Union  # <-- Added Union for type annotations
from tqdm.auto import tqdm
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseAutoencoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        m_total_neurons: Union[int, list],  # Matryoshka change: allow list
        k_active_neurons: Union[int, list],   # Matryoshka change: allow list
        aux_k: Optional[int] = None, # Number of neurons to consider for dead neuron revival
        multi_k: Optional[int] = None, # Number of neurons for secondary reconstruction
        dead_neuron_threshold_steps: int = 256 # Number of non-firing steps after which a neuron is considered dead
    ):
        super().__init__()
        self.input_dim = input_dim
        # If m_total_neurons is provided as a list, store it and set m_total_neurons to its maximum value.
        if isinstance(m_total_neurons, list):
            self.m_list = m_total_neurons  # Matryoshka change: store the list of nested sizes
            self.m_total_neurons = m_total_neurons[-1]
        else:
            self.m_list = None
            self.m_total_neurons = m_total_neurons
        
        # Similarly, if k_active_neurons is a list, store it and use its maximum for actual activation.
        if isinstance(k_active_neurons, list):
            self.k_list = k_active_neurons  # Matryoshka change: store the list of nested active counts
            self.k_active_neurons = k_active_neurons[-1]
        else:
            self.k_list = None
            self.k_active_neurons = k_active_neurons

        self.aux_k = 2 * self.k_active_neurons if aux_k is None else aux_k
        self.multi_k = 4 * self.k_active_neurons if multi_k is None else multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        # Core layers
        self.encoder = nn.Linear(input_dim, self.m_total_neurons, bias=False)
        self.decoder = nn.Linear(self.m_total_neurons, input_dim, bias=False)
        
        # Biases as separate parameters
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(self.m_total_neurons))
        
        # Tracking dead neurons
        self.steps_since_activation = torch.zeros(self.m_total_neurons, dtype=torch.long, device=device)

        # Put model on correct device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Center input
        x = x - self.input_bias
        pre_act = self.encoder(x) + self.neuron_bias
        # Compute full activation f(x) = ReLU(pre_act)
        f_full = F.relu(pre_act)  # Matryoshka change: store full activation for multi-scale reconstruction
        
        # Main top-k activation using the full activations
        topk_values, topk_indices = torch.topk(f_full, k=self.k_active_neurons, dim=-1)
        # (Redundant ReLU call removed since f_full is already ReLU)
        
        # Create sparse activation tensors
        activations = torch.zeros_like(pre_act)
        activations.scatter_(-1, topk_indices, topk_values)
        
        # Multi-k activation for loss computation
        multik_values, multik_indices = torch.topk(f_full, k=self.multi_k, dim=-1)
        multik_activations = torch.zeros_like(pre_act)
        multik_activations.scatter_(-1, multik_indices, multik_values)
        
        # Update dead neuron tracking
        self.steps_since_activation += 1
        self.steps_since_activation.scatter_(0, topk_indices.unique(), 0)
        
        # Compute reconstructions from sparse activations
        reconstruction = self.decoder(activations) + self.input_bias
        multik_reconstruction = self.decoder(multik_activations) + self.input_bias
        
        # Handle auxiliary dead neuron revival
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
            "f_full": f_full  # Matryoshka change: include full activations for multi-scale loss
        }

    def reconstruct_input_from_latents(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the sparse latent representation.
        Inputs:
            indices: Tensor of indices of active neurons
            values: Tensor of values of active neurons
        Returns:
            Reconstructed input tensor
        """
        activations = torch.zeros(self.m_total_neurons, device=indices.device)
        activations.scatter_(-1, indices, values)
        return self.decoder(activations) + self.input_bias

    def normalize_decoder_(self):
        """Normalize decoder weights to unit norm in-place."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def adjust_decoder_gradient_(self):
        """Adjust decoder gradient to maintain unit norm constraint."""
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                proj = torch.sum(self.decoder.weight * self.decoder.weight.grad, dim=0, keepdim=True)
                self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor):
        """Initialize parameters from data statistics."""
        # Initialize bias to median of data -- See O'Neill et al. (2024) Appendix A.1
        self.input_bias.data = torch.median(data_sample, dim=0).values 
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)

    def save(self, save_path: str):
        """Save model state dict and config to specified directory."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config = {
            'input_dim': self.input_dim,
            'm_total_neurons': self.m_total_neurons if self.m_list is None else self.m_list,  # Matryoshka change: save original list if available
            'k_active_neurons': self.k_active_neurons if self.k_list is None else self.k_list,   # Matryoshka change: save original list if available
            'aux_k': self.aux_k,
            'multi_k': self.multi_k,
            'dead_neuron_threshold_steps': self.dead_neuron_threshold_steps
        }
        
        torch.save({
            'config': config,
            'state_dict': self.state_dict()
        }, save_path, pickle_module=pickle)
        print(f"Saved model to {save_path}")

        return save_path

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float = 1/32,
        multi_coef: float = 0.0
    ) -> torch.Tensor:
        def normalized_mse(pred, target):
            mse = F.mse_loss(pred, target)
            baseline_mse = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
            return mse / baseline_mse

        if self.m_list is not None and self.k_list is not None:
            f_full = info["f_full"]
            total_loss = 0.0
            for i, m in enumerate(self.m_list):
                k = self.k_list[i]
                sub_activation = f_full[:, :m]
                sub_topk_values, sub_topk_indices = torch.topk(sub_activation, k=k, dim=-1)
                sparse_sub_activation = torch.zeros_like(sub_activation)
                sparse_sub_activation.scatter_(-1, sub_topk_indices, sub_topk_values)
                reconstruction_m = sparse_sub_activation @ self.decoder.weight[:, :m].T + self.input_bias
                total_loss += normalized_mse(reconstruction_m, x)
        else:
            total_loss = normalized_mse(recon, x) + multi_coef * normalized_mse(info["multik_reconstruction"], x)

        if self.aux_k is not None:
            error = x - recon.detach()
            aux_activations = torch.zeros_like(info["activations"])
            aux_activations.scatter_(-1, info["aux_indices"], info["aux_values"])
            error_reconstruction = self.decoder(aux_activations)
            aux_loss = normalized_mse(error_reconstruction, error)
            total_loss += aux_coef * aux_loss

        return total_loss

    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1/32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float = 1.0
    ) -> Dict:
        """Train the sparse autoencoder on input data.
        
        Args:
            X_train: Training data tensor.
            X_val: Optional validation data tensor.
            save_dir: Optional directory to save the trained model.
            batch_size: Batch size for training.
            learning_rate: Learning rate for training.
            n_epochs: Maximum number of training epochs.
            aux_coef: Coefficient for auxiliary loss.
            multi_coef: Coefficient for multi-k loss.
            patience: Early stopping patience.
            show_progress: Whether to display a progress bar.
            clip_grad: Gradient clipping value.
        
        Returns:
            A dictionary containing training history.
        """
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size) if X_val is not None else None
        
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
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                
                optimizer.step()
                self.normalize_decoder_()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            dead_ratio = (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
            history['dead_neuron_ratio'].append(dead_ratio)
            
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, in val_loader:
                        batch_x = batch_x.to(device)
                        recon, info = self(batch_x)
                        val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            if show_progress:
                iterator.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}' if val_loader else 'N/A',
                    'dead_ratio': f'{dead_ratio:.3f}'
                })
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.save(os.path.join(save_dir, f'SAE_M={self.m_total_neurons}_K={self.k_active_neurons}.pt'))
            
        return history

    def get_activations(self, inputs, batch_size=16384):
        """Get sparse activations for input data with batching to prevent CUDA OOM.
        
        Args:
            inputs: Input data as numpy array or torch tensor.
            batch_size: Number of samples per batch (default: 16384).
        
        Returns:
            Numpy array of activations.
        """
        self.eval()
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        
        inputs = inputs.to(device)
        num_samples = inputs.shape[0]
        all_activations = []
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Computing activations (batchsize={batch_size})"):
                batch = inputs[i:i+batch_size]
                _, info = self(batch)
                batch_activations = info['activations']
                all_activations.append(batch_activations.cpu())
        
        return torch.cat(all_activations, dim=0).numpy()

def load_model(path: str) -> SparseAutoencoder:
    """Load a saved model from path."""
    checkpoint = torch.load(path, pickle_module=pickle)
    config = checkpoint['config']
    # Restore list types if needed
    m_total = config['m_total_neurons']
    k_active = config['k_active_neurons']
    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        m_total_neurons=m_total,
        k_active_neurons=k_active,
        aux_k=config['aux_k'],
        multi_k=config['multi_k'],
        dead_neuron_threshold_steps=config['dead_neuron_threshold_steps']
    )
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from {path}")
    return model

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

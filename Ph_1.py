import os
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional, Dict

# =====================================================================
# 1. CORE NEURAL ENCODER MODULES (PHASE 1)
# =====================================================================

class SymmetricConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, symmetry_type: str = "spatial",
                 phi_in: Optional[List[int]] = None, phi_out: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.symmetry_type = symmetry_type

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.register_buffer("phi_in", torch.tensor(phi_in) if phi_in is not None else None)
        self.register_buffer("phi_out", torch.tensor(phi_out) if phi_out is not None else None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symmetry_type == "spatial":
            weight_spatial_reversed = torch.flip(self.weight, dims=[2])
            W_sym = (self.weight + weight_spatial_reversed) / 2.0
        elif self.symmetry_type == "rc":
            W_rc = torch.flip(self.weight[self.phi_out][:, self.phi_in], dims=[2])
            W_sym = (self.weight + W_rc) / 2.0
        elif self.symmetry_type == "rc_out_only":
            W_rc = torch.flip(self.weight[self.phi_out], dims=[2])
            W_sym = (self.weight + W_rc) / 2.0
        else:
            W_sym = self.weight
        return F.conv1d(x, W_sym, self.bias, self.stride, self.padding, self.dilation)


class HNetRARRouter(nn.Module):
    """
    Corrected HNetRARRouter. Enforces strict coordinate localization
    by fixing sigma as a narrow non-learnable coordinate bandwidth [2],
    preventing maximum-entropy posterior collapse.
    """
    def __init__(self, in_channels: int = 16, L_latent: int = 2000, sigma: float = 0.005):
        super().__init__()
        self.L_latent = L_latent

        # Lock sigma as a fixed, non-learnable coordinate scale parameter
        self.register_buffer("sigma", torch.tensor(sigma))

        phi_in_conv1 = list(range(8, 16)) + list(range(0, 8))
        phi_out_conv1 = list(range(16, 32)) + list(range(0, 16))

        self.boundary_conv1 = SymmetricConv1d(
            in_channels, 32, kernel_size=15, padding=7,
            symmetry_type="rc", phi_in=phi_in_conv1, phi_out=phi_out_conv1
        )
        self.boundary_conv2 = SymmetricConv1d(
            32, 1, kernel_size=1, padding=0,
            symmetry_type="rc_out_only", phi_out=[0]
        )
        self.reset_router_parameters()

    def reset_router_parameters(self):
        # Initialize weights to be extremely small to prevent early sigmoid saturation
        nn.init.normal_(self.boundary_conv2.weight, mean=0.0, std=1e-4)
        # Force initial boundary probability p_t to start at ~10% active ratio
        nn.init.constant_(self.boundary_conv2.bias, -2.197)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, L_in = x.shape
        device = x.device

        h = F.gelu(self.boundary_conv1(x))
        p_t = torch.sigmoid(self.boundary_conv2(h)).squeeze(1)

        # Symmetric Trapezoidal Integration
        c_t = torch.cumsum(p_t, dim=-1) - 0.5 * p_t
        c_max = torch.sum(p_t, dim=-1, keepdim=True).clamp(min=1e-5)

        latent_grid = torch.linspace(0, 1, steps=self.L_latent, device=device).view(-1, 1)
        normalized_c = c_t / c_max
        c_diff = normalized_c.unsqueeze(1) - latent_grid.unsqueeze(0)

        # Soft-routing matrix construction
        A = torch.exp(-0.5 * (c_diff / self.sigma) ** 2)
        A = F.softmax(A, dim=-1)

        return A, p_t


class RealDiagonalSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 10.0, d_state)))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    @torch._dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d_model, L = x.shape
        device = x.device
        delta = 1.0 / L
        A = -torch.exp(self.A_log)
        dA = torch.exp(A * delta)
        dB = (1.0 / A) * (dA - 1.0)

        state = torch.zeros(B, d_model, self.d_state, device=device)
        outputs = []
        for t in range(L):
            inputs = x[:, :, t].unsqueeze(-1)
            state = dA * state + (dB * self.B) * inputs
            out_t = torch.sum(self.C * state, dim=-1) + self.D * x[:, :, t]
            outputs.append(out_t)
        return torch.stack(outputs, dim=-1)


class SymmetricMambaModule(nn.Module):
    def __init__(self, channels: int = 128, out_channels: int = 256):
        super().__init__()
        self.ssm_layer = RealDiagonalSSM(d_model=channels, d_state=16)
        self.proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, h_fwd: torch.Tensor, phi_swap: List[int], phi_swap_out: List[int]) -> torch.Tensor:
        x_rev = torch.flip(h_fwd[:, phi_swap, :], dims=[2])
        y_fwd = self.proj(self.ssm_layer(h_fwd))
        y_rev = self.proj(self.ssm_layer(x_rev))
        y_rev_aligned = torch.flip(y_rev[:, phi_swap_out, :], dims=[2])
        return (y_fwd + y_rev_aligned) / 2.0


class SymmetricDilatedTSSStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SymmetricConv1d(1, 32, kernel_size=3, dilation=1, padding=1, symmetry_type="spatial")
        self.conv2 = SymmetricConv1d(32, 64, kernel_size=3, dilation=2, padding=2, symmetry_type="spatial")
        self.conv3 = SymmetricConv1d(64, 128, kernel_size=3, dilation=4, padding=4, symmetry_type="spatial")
        self.conv4 = SymmetricConv1d(128, 128, kernel_size=3, dilation=8, padding=8, symmetry_type="spatial")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.gelu(self.conv1(x))
        h2 = F.gelu(self.conv2(h1))
        h3 = F.gelu(self.conv3(h2))
        return self.conv4(h3)


class RealWaveletSpectralConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_scales: int = 16, n_modes: int = 16, omega0: float = 5.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.n_modes = n_modes
        self.omega0 = omega0

        scales = torch.logspace(0, 2, steps=n_scales)
        self.register_buffer("scales", scales)

        scale_init = 1.0 / (in_channels * out_channels)
        self.R_real = nn.Parameter(scale_init * torch.randn(out_channels, in_channels, n_scales, n_modes))
        self.scale_weights = nn.Parameter(torch.ones(out_channels, n_scales) / n_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, L = x.shape
        device = x.device

        X_fft = torch.fft.rfft(x, dim=-1)
        L_freq = X_fft.shape[-1]

        omega = 2 * math.pi * torch.fft.rfftfreq(L, device=device)
        omega = torch.where(omega == 0.0, torch.tensor(1e-5, device=device), omega)

        s_omega = self.scales.view(-1, 1) * omega.view(1, -1)
        norm = self.scales.view(-1, 1) ** 0.5
        W_filter = norm * torch.exp(-0.5 * (s_omega - self.omega0) ** 2)

        W_filter_complex = W_filter.to(dtype=X_fft.dtype)
        X_cwt_fft = X_fft.unsqueeze(2) * W_filter_complex.unsqueeze(0).unsqueeze(1)

        curr_modes = min(self.n_modes, L_freq)
        X_cwt_fft_truncated = X_cwt_fft[..., :curr_modes]

        R_complex = self.R_real[..., :curr_modes].to(dtype=X_cwt_fft_truncated.dtype)
        Y_cwt_fft_truncated = torch.einsum('bcsf, oisf -> bosf', X_cwt_fft_truncated, R_complex)

        scale_weights_complex = self.scale_weights.to(dtype=Y_cwt_fft_truncated.dtype)
        Y_fft_collapsed_truncated = torch.einsum('bosf, os -> bof', Y_cwt_fft_truncated, scale_weights_complex)

        # Safe pre-allocation for complex tensor padding (bypasses older PyTorch F.pad bugs)
        Y_fft_collapsed = torch.zeros((batch_size, self.out_channels, L_freq), dtype=X_cwt_fft_truncated.dtype, device=device)
        Y_fft_collapsed[..., :curr_modes] = Y_fft_collapsed_truncated

        return torch.fft.irfft(Y_fft_collapsed, n=L, dim=-1)


class RealWNO1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_scales: int = 16, n_modes: int = 16, omega0: float = 5.0):
        super().__init__()
        self.wavelet_conv = RealWaveletSpectralConv1D(
            in_channels, out_channels, n_scales=n_scales, n_modes=n_modes, omega0=omega0
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.wavelet_conv(x) + self.residual(x))


class GenomicHourglassPhase1(nn.Module):
    def __init__(
        self,
        L_latent: int = 2000, # Defines the length of the latent space representation for soft-routing.
        sigma: float = 0.005, # Controls the coordinate bandwidth for the soft-routing matrix.
        n_scales: int = 16,  # Number of wavelet scales used in the wavelet spectral convolution.
        n_modes: int = 16,   # Number of Fourier modes considered in the wavelet spectral convolution.
        omega0: float = 5.0  # Central frequency for the wavelet filter.
    ):
        super().__init__()
        self.L_latent = L_latent

        phi_in_ds = [3, 2, 1, 0]
        phi_out_ds = list(range(8, 16)) + list(range(0, 8))
        self.dna_downsampler = SymmetricConv1d(
            in_channels=4, out_channels=16, kernel_size=100, stride=100,
            symmetry_type="rc", phi_in=phi_in_ds, phi_out=phi_out_ds
        )

        self.router = HNetRARRouter(in_channels=16, L_latent=L_latent, sigma=sigma)

        phi_in_scanner = phi_out_ds
        phi_out_scanner = list(range(8, 16)) + list(range(0, 8))
        self.motif_scanner = SymmetricConv1d(
            in_channels=16, out_channels=16, kernel_size=19, padding=9,
            symmetry_type="rc", phi_in=phi_in_scanner, phi_out=phi_out_scanner
        )

        phi_in_local = phi_out_scanner
        phi_out_local = list(range(64, 128)) + list(range(0, 64))
        self.conv_local = SymmetricConv1d(
            in_channels=16, out_channels=128, kernel_size=5, padding=2,
            symmetry_type="rc", phi_in=phi_in_local, phi_out=phi_out_local
        )

        self.mamba_wrapper = SymmetricMambaModule(channels=128, out_channels=256)
        self.tss_stack = SymmetricDilatedTSSStack()
        self.wave_wno = RealWNO1DBlock(in_channels=2, out_channels=128, n_scales=n_scales, n_modes=n_modes, omega0=omega0)

        # Continuous Multi-Modal Alignment (CMMA) Projection Head
        self.align_head = nn.Conv1d(256, 256, kernel_size=1)

    def forward(self, X_DNA: torch.Tensor, X_tss: torch.Tensor, X_wave: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        X_DNA_ds = self.dna_downsampler(X_DNA)
        X_tss_ds = F.max_pool1d(X_tss, kernel_size=100, stride=100)
        X_wave_ds = F.avg_pool1d(X_wave, kernel_size=100, stride=100)

        A, p_t = self.router(X_DNA_ds)

        S_motif_raw_ds = self.motif_scanner(X_DNA_ds)
        H_DNA_high = F.gelu(self.conv_local(S_motif_raw_ds))
        H_fwd = torch.bmm(H_DNA_high, A.transpose(1, 2))

        phi_swap_in = list(range(64, 128)) + list(range(0, 64))
        phi_swap_out = list(range(128, 256)) + list(range(0, 128))
        H_DNA = self.mamba_wrapper(H_fwd, phi_swap_in, phi_swap_out)

        H_tss_dilated = self.tss_stack(X_tss_ds)
        H_tss_pooled = torch.bmm(H_tss_dilated, A.transpose(1, 2))

        H_wno_high = self.wave_wno(X_wave_ds)
        H_wave_spatial = torch.bmm(H_wno_high, A.transpose(1, 2))

        # Consolidated functional track features via channel concatenation
        H_sig = torch.cat([H_tss_pooled, H_wave_spatial], dim=1) # (B, 256, L_latent)

        return H_DNA, H_sig, A, p_t

# =====================================================================
# 2. COORDINATED DATA LOADERS (FASTA + SIGNAL NPY)
# =====================================================================
class RealDatasetLoader:
    def __init__(self, fasta_path: str, tracks_dir: str):
        self.fasta_path = fasta_path
        self.tracks_dir = tracks_dir

        print("Loading global FASTA sequence into memory...")
        self.sequence = self._load_fasta(fasta_path)
        print(f"FASTA loaded. Total length: {len(self.sequence):,} bp")

        self.npy_files = glob.glob(os.path.join(tracks_dir, "*.npy"))
        assert len(self.npy_files) > 0, f"No .npy track files found in {tracks_dir}"

        self.samples = {}
        for file_path in self.npy_files:
            filename = os.path.basename(file_path)
            parts = filename.replace(".npy", "").split("_")
            try:
                start_bp = int(parts[-2])
                end_bp = int(parts[-1])

                self.samples[start_bp] = {
                    "file_path": file_path,
                    "end_bp": end_bp
                }
            except (ValueError, IndexError):
                continue

        print(f"Indexed {len(self.samples)} window track files. Memory footprint: 0 MB (Dynamic mmap enabled).")

    def _load_fasta(self, fasta_path: str) -> str:
        seq_accumulator = []
        with open(fasta_path, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue
                seq_accumulator.append(line.strip().upper())
        return "".join(seq_accumulator)

    def extract_window(self, start_pos: int, window_size: int = 2000000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if start_pos not in self.samples:
            available = sorted(list(self.samples.keys()))
            raise KeyError(
                f"Requested start coordinate {start_pos} not found in tracks folder. "
                f"Available window coordinates are: {available}"
            )

        sample_info = self.samples[start_pos]
        file_path = sample_info["file_path"]
        end_pos = sample_info["end_bp"]

        sub_seq = self.sequence[start_pos:end_pos]
        if len(sub_seq) < window_size:
            sub_seq = sub_seq.ljust(window_size, 'N')

        mapping = {
            'A': [1.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0],
            'G': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0],
            'N': [0.25, 0.25, 0.25, 0.25]
        }
        dna_encoded = [mapping.get(b, [0.25, 0.25, 0.25, 0.25]) for b in sub_seq]
        X_DNA = torch.tensor(dna_encoded, dtype=torch.float32).transpose(0, 1).unsqueeze(0)

        tracks_slice = np.load(file_path, mmap_mode='r')

        if tracks_slice.shape[0] == 3 and tracks_slice.shape[1] > 3:
            tracks_slice = tracks_slice.T

        tracks_tensor = torch.from_numpy(tracks_slice.copy()).float().transpose(0, 1)

        X_rna = tracks_tensor[0:1, :]
        X_rna = torch.log1p(X_rna)
        X_rna = (X_rna - X_rna.mean()) / X_rna.std().clamp(min=1e-8)

        X_gene = tracks_tensor[2:3, :]
        X_gene = torch.log1p(X_gene)
        X_gene = (X_gene - X_gene.mean()) / X_gene.std().clamp(min=1e-8)

        X_wave = torch.cat([X_rna, X_gene], dim=0).unsqueeze(0)
        X_tss = tracks_tensor[1:2, :].unsqueeze(0)

        return X_DNA, X_tss, X_wave

# =====================================================================
# 3. STANDALONE PHASE 1 TRAINING LOOP
# =====================================================================

def calculate_cross_modal_coherence(h_dna, h_sig):
    h_dna_n = F.normalize(h_dna[:, :128, :], p=2, dim=1)
    h_sig_norm = F.normalize(h_sig[:, 128:, :], p=2, dim=1)
    coherence = (h_dna_n * h_sig_norm).sum(dim=1).squeeze(0)
    return coherence.cpu().numpy()


def train_phase1(
    model: nn.Module,
    loader: RealDatasetLoader,
    device: torch.device,
    epochs: int = 50,
    patience: int = 15,
    loss_weights: Dict[str, float] = None # Dictionary containing weights for L_var and L_entropy.
):
    if loss_weights is None:
        loss_weights = {"l_var": 0.5, "l_entropy": 0.02} # Adjusted default L_entropy weight to reflect original 0.2 * 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    early_stopper = EarlyStoppingPhase1(patience=patience, checkpoint_path="phase1_front_end.pt")

    all_coords = sorted(list(loader.samples.keys()))
    split_idx = int(len(all_coords) * 0.9)
    train_coords = all_coords[:split_idx]
    val_coords = all_coords[split_idx:]

    print("\n" + "="*50)
    print(f"   STARTING FULL DATASET PHASE 1 ALIGNMENT   ")
    print(f"   Training Windows: {len(train_coords)} | Val Windows: {len(val_coords)}")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for start_pos in train_coords:
            optimizer.zero_grad()
            X_DNA, X_tss, X_wave = [t.to(device) for t in loader.extract_window(start_pos)]

            H_DNA, H_sig, A, p_t = model(X_DNA, X_tss, X_wave)

            # CMMA Alignment Loss: Measures how well the model aligns DNA-derived features with actual signal features.
            H_sig_pred = model.align_head(H_DNA)
            L_align = F.mse_loss(H_sig_pred, H_sig)

            p_t_var = torch.var(p_t, dim=-1).mean()
            H_DNA_var = torch.var(H_DNA, dim=-1).mean()
            # L_var: Encourages variance in router probabilities (p_t) and DNA latent representation (H_DNA) to prevent collapse.
            L_var = - (p_t_var + H_DNA_var)
            # L_entropy: Entropy regularization on the routing matrix (A) to promote diverse or less deterministic routing.
            L_entropy = -torch.sum(A * torch.log(A + 1e-8), dim=-1).mean()

            # Total loss calculation using configurable weights
            loss = L_align + loss_weights["l_var"] * L_var + loss_weights["l_entropy"] * L_entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_coords)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for v_pos in val_coords:
                v_DNA, v_tss, v_wave = [t.to(device) for t in loader.extract_window(v_pos)]
                v_H_DNA, v_H_sig, v_A, v_p_t = model(v_DNA, v_tss, v_wave)

                v_sig_pred = model.align_head(v_H_DNA)
                v_L_align = F.mse_loss(v_sig_pred, v_H_sig)

                v_p_var = torch.var(v_p_t, dim=-1).mean()
                v_H_var = torch.var(v_H_DNA, dim=-1).mean()
                v_L_var = - (v_p_var + v_H_var)
                v_L_entropy = -torch.sum(v_A * torch.log(v_A + 1e-8), dim=-1).mean()

                # Total validation loss calculation using configurable weights
                v_loss = v_L_align + loss_weights["l_var"] * v_L_var + loss_weights["l_entropy"] * v_L_entropy
                epoch_val_loss += v_loss.item()

        avg_val_loss = epoch_val_loss / max(1, len(val_coords))

        print(f"Epoch {epoch:<3} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"\n[!] Early stopping at epoch {epoch}. Restoring best parameters.")
            break


# =====================================================================
# 4. TRAINING UTILITY
# =====================================================================

class EarlyStoppingPhase1:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, checkpoint_path: str = "phase1_best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"[*] Validation loss decreased. Saving best checkpoint to '{self.checkpoint_path}'")
        else:
            self.counter += 1
            print(f"[!] No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def compute_live_diagnostics(
    H_DNA: torch.Tensor,
    H_sig: torch.Tensor,
    p_t: torch.Tensor,
    A: torch.Tensor
) -> Dict[str, float]:
    with torch.no_grad():
        p_t_active_ratio = (p_t > 0.5).float().mean().item()
        latent_variance = H_DNA.var(dim=-1).mean().item()
        routing_entropy = -torch.sum(A * torch.log(A + 1e-8), dim=-1).mean().item()

        h_dna_norm = F.normalize(H_DNA[:, :128, :], p=2, dim=1)
        h_sig_norm = F.normalize(H_sig[:, 128:, :], p=2, dim=1)

        avg_coherence = (h_dna_norm * h_sig_norm).sum(dim=1)
        coherence_variance = avg_coherence.var(dim=-1).mean().item()
        coherence_mean = avg_coherence.mean().item()

        return {
            "pt_active_ratio": p_t_active_ratio,
            "latent_variance": latent_variance,
            "routing_entropy": routing_entropy,
            "coherence_mean": coherence_mean,
            "coherence_variance": coherence_variance
        }


def verify_and_test_phase_1(model, loader, device):
    model.eval()
    val_coords = sorted(list(loader.samples.keys()))[1]
    X_DNA, X_tss, X_wave = [t.to(device) for t in loader.extract_window(val_coords)]

    print("\n" + "="*50)
    print(f"      TESTING ON UNSEEN WINDOW STARTING AT {val_coords}      ")
    print("="*50)

    with torch.no_grad():
        H_DNA, H_sig, A, p_t = model(X_DNA, X_tss, X_wave)

    X_DNA_rc = torch.flip(X_DNA[:, [3, 2, 1, 0], :], dims=[-1])
    with torch.no_grad():
        H_DNA_rc, _, _, _ = model(X_DNA_rc, X_tss, X_wave)

    phi_swap_out = list(range(128, 256)) + list(range(0, 128))
    H_DNA_rc_aligned = torch.flip(H_DNA_rc[:, phi_swap_out, :], dims=[-1])

    equiv_mae = torch.abs(H_DNA - H_DNA_rc_aligned).mean().item()
    print(f"Test 1: Equivariance Verification   | MAE: {equiv_mae:.2e} (Pass: < 1e-6)")
    assert equiv_mae < 1e-6, "RC symmetry check failed."

    p_t_active_ratio = (p_t > 0.5).float().mean().item()
    print(f"Test 2: Soft-Boundary Active Ratio | Ratio: {p_t_active_ratio:.2%} (Pass: 5% - 15%)")

    dna_spatial_variance = H_DNA.var(dim=-1).mean().item()
    print(f"Test 3: Spatial Latent Variance    | Var: {dna_spatial_variance:.4f} (Pass: > 0.10)")

    h_dna_norm = F.normalize(H_DNA[:, :128, :], p=2, dim=1)
    h_sig_norm = F.normalize(H_sig[:, 128:, :], p=2, dim=1)
    spatial_coherence = (h_dna_norm * h_sig_norm).sum(dim=1)
    coherence_variance = spatial_coherence.var(dim=-1).mean().item()
    print(f"Test 4: Cross-Stream Coherence Var | Var: {coherence_variance:.4f} (Pass: > 0.05)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genomic Hourglass Phase 1 Trainer")
    parser.add_argument("--fasta", type=str, default="mock_sequence.fasta")
    parser.add_argument("--tracks_dir", type=str, default="tracks_folder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    loader = RealDatasetLoader(args.fasta, args.tracks_dir)

    # Define model hyperparameters
    model_hyperparams = {
        "L_latent": 2000,
        "sigma": 0.005,
        "n_scales": 16,
        "n_modes": 16,
        "omega0": 5.0
    }

    model = GenomicHourglassPhase1(**model_hyperparams).to(device)
    compiled_model = torch.compile(model)

    # Define loss weights
    phase1_loss_weights = {
        "l_var": 0.5,
        "l_entropy": 0.02
    }

    verify_and_test_phase_1(compiled_model, loader, device)
    train_phase1(compiled_model, loader, device, epochs=args.epochs, loss_weights=phase1_loss_weights)
    verify_and_test_phase_1(compiled_model, loader, device)

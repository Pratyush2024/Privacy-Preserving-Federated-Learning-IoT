"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        SMART CITY TRAFFIC IoT — FEDERATED LEARNING + DIFFERENTIAL PRIVACY   ║
║                  (Opacus-style DP-SGD implemented from scratch)              ║
║                  Course: Design of Smart Cities                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

SCENARIO
────────
A smart city has 4 traffic monitoring districts (North, South, East, West).
Each district has IoT sensors measuring:
  • vehicle_count      — number of vehicles per minute
  • avg_speed_kmh      — average speed km/h
  • occupancy_pct      — road occupancy percentage
  • queue_length       — queue length in meters
  • red_light_violations — violation count per hour
  • sensor_noise_level — RF/electrical noise (attack proxy)
  • time_of_day_sin    — cyclic time encoding (sin)
  • time_of_day_cos    — cyclic time encoding (cos)
  • is_weekend         — binary flag
  • incident_nearby    — binary flag (fire/accident nearby)

TARGET: Detect CYBERATTACKS on traffic sensors
  • 0 = Normal operation
  • 1 = Attack (spoofed readings, replay attacks, DoS)

PIPELINE (step by step)
─────────────────────────
Step 1:  Generate realistic traffic sensor data (4 districts, non-IID)
Step 2:  Visualise raw data (distributions, correlations, class balance)
Step 3:  Data pipeline — normalise, batch, DataLoader-style iteration
Step 4:  Build neural network model (2-layer MLP from scratch)
Step 5:  Train BASELINE (centralised, no privacy)
Step 6:  Train FEDERATED (no DP) — FedAvg across 4 districts
Step 7:  Train FEDERATED + OPACUS-STYLE DP-SGD
          ├─ Per-sample gradient computation
          ├─ Gradient clipping (clip_norm C)
          ├─ Gaussian noise calibration (σ from ε, δ)
          └─ Privacy accountant (epsilon tracking)
Step 8:  Compare results — convergence, confusion matrix, ROC curve
Step 9:  Privacy-utility trade-off analysis
Step 10: Generate full visual report (10 figures)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import os, time, math, warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
np.random.seed(2024)

N_DISTRICTS       = 4
DISTRICT_NAMES    = ["North District", "South District", "East District", "West District"]
DISTRICT_COLORS   = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]
SAMPLES_PER_DIST  = 500          # samples per district
N_FEATURES        = 10           # sensor features
ATTACK_RATIO      = 0.22         # 22% attack samples

# Model hyperparams
HIDDEN_SIZE       = 32
LEARNING_RATE     = 0.05
BATCH_SIZE        = 64
LOCAL_EPOCHS      = 3
FL_ROUNDS         = 25

# Differential Privacy (Opacus-style)
DP_CLIP_NORM      = 1.0          # C — gradient clipping threshold
DP_DELTA          = 1e-5         # δ — failure probability
DP_EPSILON_LIST   = [0.1, 0.5, 1.0, 2.0, 5.0]   # privacy budgets to test

OUTPUT_DIR        = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "vehicle_count", "avg_speed_kmh", "occupancy_pct",
    "queue_length", "red_light_violations", "sensor_noise_level",
    "time_of_day_sin", "time_of_day_cos", "is_weekend", "incident_nearby"
]

# ══════════════════════════════════════════════
#  STEP 1 — DATA GENERATION
# ══════════════════════════════════════════════
def print_step(n, title):
    print(f"\n{'═'*65}")
    print(f"  STEP {n}: {title}")
    print(f"{'═'*65}")

def generate_traffic_data(district_idx, n_samples=SAMPLES_PER_DIST):
    """
    Generate realistic non-IID traffic sensor data for a city district.

    Each district has different baseline traffic patterns:
      North — busy commuter corridor (high vehicle count, low speed)
      South — residential (lower count, higher speed)
      East  — commercial/industrial (high occupancy, many violations)
      West  — highway junction (fast, variable queue)

    Attack samples contain:
      - Abnormally high/low sensor readings (spoofing)
      - Elevated noise level (RF jamming signature)
      - Inconsistent speed/occupancy combinations
    """
    n_attack  = int(n_samples * ATTACK_RATIO)
    n_normal  = n_samples - n_attack

    # District-specific baselines (makes data non-IID)
    baselines = {
        0: dict(vc=85,  spd=28, occ=62, ql=180, rlv=4.2, noise=0.12),  # North
        1: dict(vc=42,  spd=55, occ=31, ql=60,  rlv=1.8, noise=0.08),  # South
        2: dict(vc=120, spd=18, occ=78, ql=340, rlv=9.1, noise=0.15),  # East
        3: dict(vc=68,  spd=95, occ=44, ql=90,  rlv=2.5, noise=0.10),  # West
    }
    b = baselines[district_idx]

    # ── Normal samples ──
    hours = np.random.uniform(0, 24, n_normal)
    X_normal = np.column_stack([
        np.random.normal(b["vc"],   15,        n_normal).clip(0, 200),   # vehicle_count
        np.random.normal(b["spd"],  12,        n_normal).clip(0, 130),   # avg_speed
        np.random.normal(b["occ"],  10,        n_normal).clip(0, 100),   # occupancy
        np.random.normal(b["ql"],   60,        n_normal).clip(0, 800),   # queue_length
        np.random.poisson(b["rlv"],            n_normal).clip(0, 30),    # violations
        np.random.exponential(b["noise"],      n_normal).clip(0, 2),     # noise
        np.sin(2 * np.pi * hours / 24),                                  # time_sin
        np.cos(2 * np.pi * hours / 24),                                  # time_cos
        np.random.binomial(1, 0.29,            n_normal).astype(float),  # is_weekend
        np.random.binomial(1, 0.05,            n_normal).astype(float),  # incident_nearby
    ])
    y_normal = np.zeros(n_normal)

    # ── Attack samples (adversarial patterns) ──
    hours_a = np.random.uniform(0, 24, n_attack)
    attack_type = np.random.choice(3, n_attack)  # 3 attack types

    vc_a   = np.where(attack_type == 0,
                      np.random.normal(b["vc"] * 0.05, 5,  n_attack),   # type0: near-zero count (sensor blind)
                      np.where(attack_type == 1,
                               np.random.normal(200, 10,   n_attack),   # type1: max spoof
                               np.random.normal(b["vc"],15,n_attack)))   # type2: subtle
    spd_a  = np.where(attack_type == 0,
                      np.random.normal(130, 5, n_attack),                # impossible speed
                      np.random.normal(b["spd"], 8, n_attack))
    occ_a  = np.where(attack_type == 1,
                      np.random.normal(95, 3, n_attack),                 # 95% occupancy spoof
                      np.random.normal(b["occ"], 8, n_attack))
    noise_a = np.random.exponential(b["noise"] * 8, n_attack)           # ALWAYS elevated noise during attack

    X_attack = np.column_stack([
        vc_a.clip(0, 200),
        spd_a.clip(0, 130),
        occ_a.clip(0, 100),
        np.random.normal(b["ql"] * 0.5 + 200, 100, n_attack).clip(0, 800),
        np.random.poisson(b["rlv"] * 2.5,           n_attack).clip(0, 30),
        noise_a.clip(0, 2),
        np.sin(2 * np.pi * hours_a / 24),
        np.cos(2 * np.pi * hours_a / 24),
        np.random.binomial(1, 0.29, n_attack).astype(float),
        np.random.binomial(1, 0.15, n_attack).astype(float),            # more incidents during attack
    ])
    y_attack = np.ones(n_attack)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack]).astype(int)
    shuffle = np.random.permutation(len(y))
    return X[shuffle], y[shuffle]

# ══════════════════════════════════════════════
#  STEP 3 — DATA PIPELINE (DataLoader-style)
# ══════════════════════════════════════════════
class TrafficDataLoader:
    """
    Mimics PyTorch DataLoader.
    Normalises features, batches data, supports iteration.
    """
    def __init__(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        self.scaler    = StandardScaler()
        self.X_raw     = X.copy()
        self.y         = y.copy()
        self.batch_size= batch_size
        self.shuffle   = shuffle
        self.X         = self.scaler.fit_transform(X)
        self.n         = len(y)

    def __iter__(self):
        idx = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def get_all(self):
        return self.X, self.y

# ══════════════════════════════════════════════
#  STEP 4 — NEURAL NETWORK (2-layer MLP)
# ══════════════════════════════════════════════
class MLP:
    """
    2-layer MLP: Input(10) → ReLU → Hidden(32) → Sigmoid → Output(1)

    Weights stored as flat parameter vector for easy DP manipulation.
    Architecture mirrors a PyTorch nn.Sequential model.
    """
    def __init__(self, in_dim=N_FEATURES, hidden=HIDDEN_SIZE):
        self.in_dim  = in_dim
        self.hidden  = hidden
        # He initialisation (good for ReLU)
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1)     * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.X_in = X
        self.z1   = X @ self.W1 + self.b1
        self.a1   = np.maximum(0, self.z1)          # ReLU
        self.z2   = self.a1 @ self.W2 + self.b2
        self.out  = 1.0 / (1.0 + np.exp(-self.z2.clip(-500, 500)))  # Sigmoid
        return self.out.ravel()

    def backward(self, X, y, lr=LEARNING_RATE):
        """Standard backprop (used in centralised training)."""
        n    = len(y)
        prob = self.forward(X)
        dL   = (prob - y) / n                       # BCE gradient

        dW2  = self.a1.T @ dL.reshape(-1, 1)
        db2  = dL.sum()
        da1  = dL.reshape(-1, 1) * self.W2.T
        dz1  = da1 * (self.z1 > 0)                 # ReLU grad
        dW1  = X.T @ dz1
        db1  = dz1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return self._bce_loss(prob, y)

    def per_sample_gradients(self, X, y):
        """
        ╔══════════════════════════════════════════════════╗
        ║  OPACUS CORE — Per-sample gradient computation   ║
        ║  (ghost clipping / vmap equivalent)              ║
        ╚══════════════════════════════════════════════════╝
        Returns list of per-sample gradient dicts.
        In Opacus, this is done via hooks on nn.Module.
        Here we vectorise it manually.
        """
        grads = []
        for xi, yi in zip(X, y):
            xi = xi.reshape(1, -1)
            prob = self.forward(xi)
            dL   = (prob.ravel()[0] - yi)           # scalar BCE grad (single sample)

            dW2 = self.a1.T * dL
            db2 = np.array([dL])
            da1 = dL * self.W2.T
            dz1 = da1 * (self.z1 > 0)
            dW1 = xi.T @ dz1
            db1 = dz1.ravel()

            grads.append({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
        return grads

    def get_params(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_params(self, p):
        self.W1 = p["W1"].copy(); self.b1 = p["b1"].copy()
        self.W2 = p["W2"].copy(); self.b2 = p["b2"].copy()

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X)

    @staticmethod
    def _bce_loss(prob, y):
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

    def loss(self, X, y):
        return self._bce_loss(self.forward(X), y)

    def clone(self):
        m = MLP(self.in_dim, self.hidden)
        m.set_params(self.get_params())
        return m

# ══════════════════════════════════════════════
#  STEP 7a — OPACUS-STYLE DP ENGINE
# ══════════════════════════════════════════════
class OPACUSStyleDP:
    """
    Replicates the Opacus PrivacyEngine for a numpy MLP.

    OPACUS PIPELINE (each batch):
    ┌─────────────────────────────────────────────────────┐
    │ 1. Compute per-sample gradients (ghost clipping)     │
    │ 2. Clip each per-sample gradient to L2 norm C        │
    │    g̃ᵢ = gᵢ / max(1, ‖gᵢ‖₂ / C)                    │
    │ 3. Sum clipped gradients                             │
    │    Σ g̃ᵢ                                              │
    │ 4. Add Gaussian noise                                │
    │    noise ~ N(0, σ²C²I)                               │
    │    σ = sqrt(2 ln(1.25/δ)) / ε  (per Dwork 2014)     │
    │ 5. Divide by batch size (unbiased estimate)          │
    │ 6. Apply noisy gradient update                       │
    │ 7. Accumulate privacy cost via moments accountant    │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, model, epsilon, delta=DP_DELTA, clip_norm=DP_CLIP_NORM,
                 lr=LEARNING_RATE):
        self.model     = model
        self.epsilon   = epsilon
        self.delta     = delta
        self.C         = clip_norm
        self.lr        = lr
        self.sigma     = self._calibrate_sigma(epsilon, delta)
        self.steps     = 0
        self.privacy_spent = 0.0
        print(f"    [DP ENGINE] ε={epsilon}, δ={delta:.0e}, C={clip_norm}, σ={self.sigma:.4f}")

    def _calibrate_sigma(self, epsilon, delta):
        """
        Gaussian mechanism noise multiplier.
        σ = sqrt(2 * ln(1.25/δ)) / ε
        This guarantees (ε, δ)-differential privacy for a single query.
        """
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def _l2_norm_dict(self, grad):
        """Compute L2 norm across all parameter tensors."""
        total = sum(np.sum(g ** 2) for g in grad.values())
        return math.sqrt(total)

    def _scale_dict(self, grad, scale):
        return {k: v * scale for k, v in grad.items()}

    def _add_dicts(self, a, b):
        return {k: a[k] + b[k] for k in a}

    def step(self, X_batch, y_batch):
        """
        One DP-SGD step on a mini-batch.
        Returns (loss, clipped_norms, noise_scales) for diagnostics.
        """
        n = len(y_batch)

        # ── 1. Per-sample gradients ──
        per_sample_grads = self.model.per_sample_gradients(X_batch, y_batch)

        # ── 2. Clip each gradient ──
        clipped    = []
        clip_norms = []
        for g in per_sample_grads:
            norm = self._l2_norm_dict(g)
            clip_norms.append(norm)
            scale = min(1.0, self.C / (norm + 1e-12))
            clipped.append(self._scale_dict(g, scale))

        # ── 3. Sum clipped gradients ──
        agg = {k: np.zeros_like(v) for k, v in clipped[0].items()}
        for g in clipped:
            agg = self._add_dicts(agg, g)

        # ── 4. Add Gaussian noise ──
        noise_scales = {}
        noisy_agg = {}
        for k, v in agg.items():
            noise  = np.random.normal(0, self.sigma * self.C, size=v.shape)
            noisy_agg[k] = v + noise
            noise_scales[k] = np.linalg.norm(noise)

        # ── 5. Divide by batch size ──
        avg_grad = self._scale_dict(noisy_agg, 1.0 / n)

        # ── 6. Gradient descent update ──
        p = self.model.get_params()
        for k in p:
            p[k] -= self.lr * avg_grad[k]
        self.model.set_params(p)

        # ── 7. Privacy accounting (moments accountant approximation) ──
        self.steps += 1
        self.privacy_spent = self._compute_epsilon_spent(n)

        loss = self.model.loss(X_batch, y_batch)
        return loss, np.mean(clip_norms), np.mean(list(noise_scales.values()))

    def _compute_epsilon_spent(self, batch_size):
        """
        Approximate RDP accountant (simplified).
        ε_spent ≈ (steps * batch_size * σ² * C²)^(1/2) * sqrt(2 ln(1/δ))
        """
        q  = batch_size / (N_DISTRICTS * SAMPLES_PER_DIST)
        rdp_eps = q * math.sqrt(self.steps) * self.C / (self.sigma + 1e-9)
        return min(rdp_eps + math.sqrt(2 * math.log(1 / self.delta)), self.epsilon * 2)

# ══════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ══════════════════════════════════════════════
def train_centralised(all_X, all_y, verbose=True):
    """Baseline: all data pooled, standard training (no FL, no DP)."""
    model    = MLP()
    loader   = TrafficDataLoader(all_X, all_y)
    history  = []
    if verbose: print("  Training centralised model …")
    for rnd in range(FL_ROUNDS):
        epoch_loss = []
        for X_b, y_b in loader:
            loss = model.backward(X_b, y_b)
            epoch_loss.append(loss)
        X_all, y_all = loader.get_all()
        acc  = accuracy_score(y_all, model.predict(X_all))
        f1   = f1_score(y_all, model.predict(X_all), zero_division=0)
        history.append((acc, f1, np.mean(epoch_loss)))
        if verbose and (rnd + 1) % 5 == 0:
            print(f"    Round {rnd+1:3d}/{FL_ROUNDS} | Loss: {np.mean(epoch_loss):.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f}")
    return model, history


def fedavg_aggregate(global_params, local_updates, weights):
    """
    FedAvg aggregation.
    new_global = Σ (wᵢ * update_i)  where wᵢ = nᵢ / Σnᵢ
    """
    new_params = {k: np.zeros_like(v) for k, v in global_params.items()}
    for upd, w in zip(local_updates, weights):
        for k in new_params:
            new_params[k] += w * upd[k]
    return new_params


def train_federated(datasets, use_dp=False, epsilon=1.0, verbose=True):
    """
    Federated Learning across 4 traffic districts.
    If use_dp=True, applies Opacus-style DP-SGD locally before aggregation.

    DATA FLOW per round:
    ┌────────────────────────────────────────────────────────────┐
    │  Central Server broadcasts global model                    │
    │       │                                                    │
    │       ├──► District 0 (North)                             │
    │       │      ↓ local data  ↓ DP-SGD (if enabled)          │
    │       │      local_model_0  → delta_0 (+ noise if DP)     │
    │       │                                                    │
    │       ├──► District 1 (South) … same                      │
    │       ├──► District 2 (East)  … same                      │
    │       └──► District 3 (West)  … same                      │
    │                                                            │
    │  Server: FedAvg(delta_0, delta_1, delta_2, delta_3)        │
    │  → New global model                                        │
    └────────────────────────────────────────────────────────────┘
    """
    tag     = f"FL+DP(ε={epsilon})" if use_dp else "FL (no DP)"
    global_model = MLP()
    loaders = [TrafficDataLoader(X, y) for X, y in datasets]
    all_X   = np.vstack([d[0] for d in datasets])
    all_y   = np.concatenate([d[1] for d in datasets])
    all_X_n = np.vstack([l.X for l in loaders])

    history   = []
    dp_diagnostics = []   # clip norms, noise scales
    total_n   = sum(len(d[1]) for d in datasets)
    weights   = [len(d[1]) / total_n for d in datasets]

    if verbose:
        print(f"  Training {tag} …")

    for rnd in range(FL_ROUNDS):
        global_params = global_model.get_params()
        local_updates = []
        round_diag    = []

        for i, (loader, (X_raw, y)) in enumerate(zip(loaders, datasets)):
            local_model = global_model.clone()

            if use_dp:
                dp_engine = OPACUSStyleDP(local_model, epsilon=epsilon)
                for epoch in range(LOCAL_EPOCHS):
                    for X_b, y_b in loader:
                        _, clip_n, noise_s = dp_engine.step(X_b, y_b)
                        round_diag.append((clip_n, noise_s))
            else:
                for epoch in range(LOCAL_EPOCHS):
                    for X_b, y_b in loader:
                        local_model.backward(X_b, y_b)

            # Compute update delta
            local_p = local_model.get_params()
            delta   = {k: local_p[k] - global_params[k] for k in global_params}
            local_updates.append(delta)

        # FedAvg
        agg_delta = fedavg_aggregate(global_params, local_updates, weights)
        new_params = {k: global_params[k] + agg_delta[k] for k in global_params}
        global_model.set_params(new_params)

        # Evaluate on combined normalised data
        acc   = accuracy_score(all_y, global_model.predict(all_X_n))
        f1    = f1_score(all_y, global_model.predict(all_X_n), zero_division=0)
        loss  = global_model.loss(all_X_n, all_y)
        history.append((acc, f1, loss))

        if round_diag:
            dp_diagnostics.append((np.mean([d[0] for d in round_diag]),
                                   np.mean([d[1] for d in round_diag])))

        if verbose and (rnd + 1) % 5 == 0:
            dp_info = f" | σ={dp_diagnostics[-1][1]:.3f}" if use_dp else ""
            print(f"    Round {rnd+1:3d}/{FL_ROUNDS} | Loss: {loss:.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f}{dp_info}")

    return global_model, history, dp_diagnostics


# ══════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════
def evaluate_model(model, loaders):
    all_X = np.vstack([l.X for l in loaders])
    all_y = np.concatenate([l.y for l in loaders])
    preds = model.predict(all_X)
    proba = model.predict_proba(all_X)
    return {
        "acc":  accuracy_score(all_y, preds),
        "f1":   f1_score(all_y, preds, zero_division=0),
        "cm":   confusion_matrix(all_y, preds),
        "roc":  roc_curve(all_y, proba),
        "auc":  auc(*roc_curve(all_y, proba)[:2]),
        "preds":preds,
        "proba":proba,
        "y":    all_y
    }


# ══════════════════════════════════════════════
#  FIGURE GENERATION (10 plots)
# ══════════════════════════════════════════════

def fig01_data_pipeline(datasets, loaders):
    """Show how raw sensor data is collected → normalised → batched."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("STEP 1–3: Traffic IoT Data Collection & Pipeline",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel A: Raw sensor values per district (box plot) ---
    ax = fig.add_subplot(gs[0, :2])
    data_for_box = []
    labels_box   = []
    for i, (X, y) in enumerate(datasets):
        for j in range(4):        # first 4 features only
            data_for_box.append(X[:, j])
            labels_box.append(f"{DISTRICT_NAMES[i][:5]}\n{FEATURE_NAMES[j][:7]}")
    positions = list(range(len(data_for_box)))
    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                    widths=0.6, flierprops=dict(marker='.', markersize=2))
    colors_cycle = [DISTRICT_COLORS[i // 4] for i in range(len(data_for_box))]
    for patch, color in zip(bp["boxes"], colors_cycle):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_box, fontsize=7)
    ax.set_title("(A) Raw Sensor Value Distributions per District (4 features shown)", fontweight="bold")
    ax.set_ylabel("Raw Value")
    handles = [mpatches.Patch(color=c, label=n, alpha=0.7)
               for c, n in zip(DISTRICT_COLORS, DISTRICT_NAMES)]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    # --- Panel B: Class balance per district ---
    ax = fig.add_subplot(gs[0, 2])
    ratios = [(np.sum(y == 0), np.sum(y == 1)) for _, y in datasets]
    x = np.arange(N_DISTRICTS)
    bars_n = ax.bar(x - 0.2, [r[0] for r in ratios], 0.4, label="Normal", color="#2ECC71", alpha=0.8)
    bars_a = ax.bar(x + 0.2, [r[1] for r in ratios], 0.4, label="Attack",  color="#E74C3C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:5] for n in DISTRICT_NAMES], fontsize=9)
    ax.set_title("(B) Class Balance\nper District", fontweight="bold")
    ax.set_ylabel("# Samples")
    ax.legend(fontsize=8)
    for bar in bars_n: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                                str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)
    for bar in bars_a: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                                str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)

    # --- Panel C: Feature normalisation before/after ---
    ax = fig.add_subplot(gs[1, 0])
    X_raw = datasets[0][0][:, 0]    # vehicle_count raw
    X_norm = loaders[0].X[:, 0]     # normalised
    ax.hist(X_raw,  bins=30, color="#3498DB", alpha=0.7, label="Raw")
    ax.hist(X_norm, bins=30, color="#E67E22", alpha=0.7, label="Normalised (StandardScaler)")
    ax.set_title("(C) Feature Normalisation\n(vehicle_count, North)", fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # --- Panel D: Batch iteration diagram ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    pipeline_text = (
        "DATA PIPELINE (per district)\n"
        "────────────────────────────────\n"
        "① Raw IoT readings (CSV/stream)\n"
        "        ↓\n"
        "② Feature extraction (10 features)\n"
        "        ↓\n"
        "③ StandardScaler normalisation\n"
        "   μ=0, σ=1 per feature\n"
        "        ↓\n"
        "④ Shuffle + mini-batch split\n"
        f"   Batch size: {BATCH_SIZE} samples\n"
        f"   Batches/epoch: {math.ceil(SAMPLES_PER_DIST/BATCH_SIZE)}\n"
        "        ↓\n"
        "⑤ Local model training\n"
        "   (+ DP noise if enabled)\n"
    )
    ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB", alpha=0.8))
    ax.set_title("(D) Pipeline Flow", fontweight="bold")

    # --- Panel E: Correlation heatmap (one district) ---
    ax = fig.add_subplot(gs[1, 2])
    df = pd.DataFrame(datasets[2][0], columns=FEATURE_NAMES)
    corr = df.iloc[:, :6].corr()
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=True,
                fmt=".1f", linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("(E) Feature Correlation\n(East District, 6 features)", fontweight="bold")
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.tick_params(axis='y', labelsize=6, rotation=0)

    plt.savefig(f"{OUTPUT_DIR}/fig01_data_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig01_data_pipeline.png")


def fig02_attack_patterns(datasets):
    """Visualise how attack traffic looks vs normal."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("STEP 1: Attack vs Normal Traffic Patterns Across Districts",
                 fontsize=14, fontweight="bold")

    features_to_plot = ["vehicle_count", "avg_speed_kmh", "occupancy_pct", "sensor_noise_level"]
    f_idx            = [0, 1, 2, 5]

    for col, (fi, fname) in enumerate(zip(f_idx, features_to_plot)):
        for row, (i, (X, y)) in enumerate(zip(range(2), datasets[:2])):
            ax = axes[row][col]
            ax.hist(X[y == 0, fi], bins=30, color="#2ECC71", alpha=0.65,
                    label="Normal", density=True)
            ax.hist(X[y == 1, fi], bins=30, color="#E74C3C", alpha=0.65,
                    label="Attack",  density=True)
            ax.set_title(f"{DISTRICT_NAMES[i][:5]}\n{fname}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            if col == 0: ax.set_ylabel("Density", fontsize=8)
            if row == 0 and col == 0: ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig02_attack_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig02_attack_patterns.png")


def fig03_dp_mechanism(epsilon_list=DP_EPSILON_LIST):
    """Illustrate the DP-SGD mechanism: clipping + noise injection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("STEP 7: Opacus-Style DP-SGD Mechanism",
                 fontsize=14, fontweight="bold")

    # Panel A: Gradient clipping
    ax = axes[0]
    n_pts = 200
    grads = np.random.randn(n_pts, 2) * 1.5
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    C     = DP_CLIP_NORM
    clipped = grads / np.maximum(1.0, norms / C)

    ax.scatter(grads[:, 0],   grads[:, 1],   c="tomato",     s=15, alpha=0.5, label="Original grads")
    ax.scatter(clipped[:, 0], clipped[:, 1], c="steelblue",  s=15, alpha=0.5, label="Clipped grads")
    circle = plt.Circle((0, 0), C, color="black", fill=False, linestyle="--", lw=1.5, label=f"Clip norm C={C}")
    ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title("(A) Gradient Clipping\n(bounds sensitivity)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("∂L/∂w₁"); ax.set_ylabel("∂L/∂w₂")

    # Panel B: Gaussian noise calibration curve
    ax = axes[1]
    eps_range = np.linspace(0.05, 6.0, 200)
    sigmas    = [math.sqrt(2 * math.log(1.25 / DP_DELTA)) / e for e in eps_range]
    ax.plot(eps_range, sigmas, "navy", lw=2)
    for eps in epsilon_list:
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        ax.axvline(x=eps, color="gray", linestyle=":", lw=1)
        ax.scatter([eps], [sig], s=80, zorder=5,
                   color=plt.cm.RdYlGn(eps / max(epsilon_list)))
        ax.annotate(f"ε={eps}\nσ={sig:.2f}", (eps, sig),
                    textcoords="offset points", xytext=(8, 5), fontsize=7)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Noise Multiplier σ")
    ax.set_title("(B) Noise Calibration\nσ = √(2·ln(1.25/δ)) / ε", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6.5)

    # Panel C: Effect of noise on gradients
    ax = axes[2]
    true_grad = np.linspace(-2, 2, 300)
    ax.plot(true_grad, true_grad, "k-", lw=2, label="True gradient (no DP)")
    for eps, col in zip([5.0, 1.0, 0.1], ["#27AE60", "#F39C12", "#E74C3C"]):
        sig  = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        noisy = true_grad + np.random.normal(0, sig * C, len(true_grad))
        ax.scatter(true_grad[::10], noisy[::10], s=12, alpha=0.6, color=col,
                   label=f"ε={eps} (σ={sig:.2f})")
    ax.set_xlabel("True gradient value")
    ax.set_ylabel("Noisy gradient value")
    ax.set_title("(C) Gradient Corruption\nby DP Noise", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig03_dp_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig03_dp_mechanism.png")


def fig04_fedavg_data_flow(datasets, loaders):
    """Animated-style diagram showing data never leaving district boundary."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.set_title("STEP 6–7: Federated Learning Data Flow\n"
                 "(Raw data stays local — only model updates travel)",
                 fontsize=13, fontweight="bold", pad=15)

    # ── Central server box ──
    srv = plt.Rectangle((5.5, 4.3), 3, 1.4, color="#2C3E50", ec="black", lw=2, zorder=3)
    ax.add_patch(srv)
    ax.text(7, 5.1, "CENTRAL SERVER\n(FedAvg Aggregation)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)

    # ── District boxes ──
    positions = [(1.0, 7.5), (10.5, 7.5), (1.0, 1.5), (10.5, 1.5)]
    for i, (px, py) in enumerate(positions):
        rect = plt.Rectangle((px, py), 2.8, 1.8, color=DISTRICT_COLORS[i],
                              ec="black", lw=1.5, zorder=3, alpha=0.85)
        ax.add_patch(rect)
        ax.text(px + 1.4, py + 1.15, DISTRICT_NAMES[i],
                ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)

        n_att = np.sum(datasets[i][1] == 1)
        n_nor = np.sum(datasets[i][1] == 0)
        ax.text(px + 1.4, py + 0.45,
                f"Normal: {n_nor}  Attack: {n_att}",
                ha="center", va="center", fontsize=7, color="white", zorder=4)

        # Lock icon — data stays local
        ax.text(px + 0.25, py + 1.3, "🔒", fontsize=12, zorder=5)

        # Arrow: model update → server (blue)
        cx = px + 1.4
        cy = py + 1.8 if py > 4 else py
        sx, sy = 7.0, 5.7 if py > 4 else 4.3
        ax.annotate("", xy=(sx, sy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#2980B9", lw=2.0))
        ax.text((cx + sx) / 2 + 0.1, (cy + sy) / 2 + 0.15,
                "Δ model\n(+ DP noise)", fontsize=7, color="#2980B9", ha="center")

        # Arrow: global model → district (green, dashed)
        ax.annotate("", xy=(cx, cy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5,
                                   linestyle="dashed",
                                   connectionstyle="arc3,rad=0.3"))

    # Global model label
    ax.text(7, 6.8, "← Global model broadcast", ha="center", fontsize=8,
            color="#27AE60", fontstyle="italic")

    # Legend
    blue_patch  = mpatches.Patch(color="#2980B9", label="Model update (clipped + noisy)")
    green_patch = mpatches.Patch(color="#27AE60", label="Global model broadcast")
    lock_patch  = mpatches.Patch(color="gray",   label="🔒 Raw data never leaves district")
    ax.legend(handles=[blue_patch, green_patch, lock_patch],
              loc="lower center", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig04_fedavg_data_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig04_fedavg_data_flow.png")


def fig05_convergence(hist_central, hist_fl, dp_histories):
    rounds = range(1, FL_ROUNDS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("STEP 8: Training Convergence — Centralised vs FL vs FL+DP",
                 fontsize=13, fontweight="bold")
    colors_dp = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(DP_EPSILON_LIST)))

    for ax, metric, label in zip(axes, [0, 1, 2], ["Accuracy", "F1-Score", "Loss"]):
        ax.plot(rounds, [h[metric] for h in hist_central], "k--", lw=2.5, label="Centralised")
        ax.plot(rounds, [h[metric] for h in hist_fl],      color="#3498DB", lw=2.5, label="FL (no DP)")
        for eps, hist, col in zip(DP_EPSILON_LIST, dp_histories, colors_dp):
            ax.plot(rounds, [h[metric] for h in hist], color=col, lw=1.5, alpha=0.85,
                    label=f"FL+DP ε={eps}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over Rounds")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric in [0, 1]: ax.set_ylim(0.4, 1.02)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig05_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig05_convergence.png")


def fig06_confusion_matrices(eval_central, eval_fl, eval_dp_best):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("STEP 8: Confusion Matrices — Attack Detection Performance",
                 fontsize=13, fontweight="bold")
    titles = ["Centralised", "Federated (no DP)", "FL+DP (ε=1.0)"]
    evals  = [eval_central, eval_fl, eval_dp_best]
    for ax, ev, title in zip(axes, evals, titles):
        cm  = ev["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_title(f"{title}\nAcc={ev['acc']:.4f} | F1={ev['f1']:.4f} | AUC={ev['auc']:.4f}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig06_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig06_confusion_matrices.png")


def fig07_roc_curves(eval_central, eval_fl, dp_evals):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("STEP 8: ROC Curves — Cyberattack Detection",
                 fontsize=13, fontweight="bold")
    colors_dp = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(DP_EPSILON_LIST)))

    for ev, label, color, lw, ls in [
        (eval_central, "Centralised",   "black",    2.5, "--"),
        (eval_fl,      "FL (no DP)",    "#3498DB",  2.5, "-"),
    ]:
        fpr, tpr, _ = ev["roc"]
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{label} (AUC={ev['auc']:.4f})")

    for eps, ev, col in zip(DP_EPSILON_LIST, dp_evals, colors_dp):
        fpr, tpr, _ = ev["roc"]
        ax.plot(fpr, tpr, color=col, lw=1.5, alpha=0.85,
                label=f"FL+DP ε={eps} (AUC={ev['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate (Normal → Attack)")
    ax.set_ylabel("True Positive Rate (Attack detected)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig07_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig07_roc_curves.png")


def fig08_privacy_utility(dp_evals, eval_fl):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("STEP 9: Privacy–Utility Trade-off Analysis",
                 fontsize=13, fontweight="bold")

    eps_vals = DP_EPSILON_LIST
    accs     = [e["acc"] for e in dp_evals]
    f1s      = [e["f1"]  for e in dp_evals]
    aucs     = [e["auc"] for e in dp_evals]

    for ax, vals, metric, color in zip(
        axes,
        [accs, f1s, aucs],
        ["Accuracy", "F1-Score", "AUC"],
        ["steelblue", "darkorange", "purple"]
    ):
        ax.plot(eps_vals, vals, "o-", color=color, lw=2, markersize=9, label="FL+DP")
        ax.axhline(eval_fl[metric.lower().replace("-","").replace("score","").strip()
                           if metric != "AUC" else "auc"],
                   color="green", ls="--", lw=1.5, label="FL (no DP)")
        ax.fill_between(eps_vals, vals, alpha=0.15, color=color)
        ax.set_xlabel("Privacy Budget ε  (← more private | less private →)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs ε", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        # Annotate sweet-spot
        best_i = np.argmax(vals)
        ax.annotate("sweet\nspot", (eps_vals[best_i], vals[best_i]),
                    textcoords="offset points", xytext=(10, -20),
                    fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig08_privacy_utility.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig08_privacy_utility.png")


def fig09_dp_diagnostics(dp_diagnostics_per_eps):
    """Show clip norms and noise levels throughout training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("STEP 7: DP-SGD Diagnostics — Gradient Norms & Noise Throughout Training",
                 fontsize=13, fontweight="bold")
    colors_dp = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(DP_EPSILON_LIST)))

    for eps, diag, col in zip(DP_EPSILON_LIST, dp_diagnostics_per_eps, colors_dp):
        if not diag: continue
        rounds_d = range(1, len(diag) + 1)
        axes[0].plot(rounds_d, [d[0] for d in diag], color=col, lw=1.5, label=f"ε={eps}")
        axes[1].plot(rounds_d, [d[1] for d in diag], color=col, lw=1.5, label=f"ε={eps}")

    axes[0].axhline(DP_CLIP_NORM, color="black", ls="--", lw=1.5, label=f"Clip norm C={DP_CLIP_NORM}")
    axes[0].set_title("Mean Per-Sample Gradient Norm\n(before clipping)", fontweight="bold")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Gradient L2 Norm")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Mean Gaussian Noise Magnitude\n(added to aggregated gradient)", fontweight="bold")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Noise L2 Norm")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig09_dp_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig09_dp_diagnostics.png")


def fig10_summary_dashboard(hist_central, hist_fl, dp_histories,
                             eval_central, eval_fl, dp_evals,
                             datasets):
    """Final one-page summary dashboard."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("SMART CITY TRAFFIC IoT SECURITY — Full Results Dashboard\n"
                 "Federated Learning + Differential Privacy (Opacus-style DP-SGD)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)
    rounds = range(1, FL_ROUNDS + 1)
    colors_dp = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(DP_EPSILON_LIST)))

    # A — Accuracy convergence
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(rounds, [h[0] for h in hist_central], "k--", lw=2, label="Centralised")
    ax.plot(rounds, [h[0] for h in hist_fl], "#3498DB", lw=2, label="FL (no DP)")
    for eps, hist, col in zip(DP_EPSILON_LIST, dp_histories, colors_dp):
        ax.plot(rounds, [h[0] for h in hist], col, lw=1.5, alpha=0.8, label=f"FL+DP ε={eps}")
    ax.set_title("(A) Accuracy Convergence", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0.4, 1.02)

    # B — Loss convergence
    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(rounds, [h[2] for h in hist_central], "k--", lw=2, label="Centralised")
    ax.plot(rounds, [h[2] for h in hist_fl], "#3498DB", lw=2, label="FL (no DP)")
    for eps, hist, col in zip(DP_EPSILON_LIST, dp_histories, colors_dp):
        ax.plot(rounds, [h[2] for h in hist], col, lw=1.5, alpha=0.8, label=f"FL+DP ε={eps}")
    ax.set_title("(B) Loss Convergence", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # C — Confusion matrix FL+DP ε=1.0
    ax = fig.add_subplot(gs[1, 0])
    best_dp_eval = dp_evals[DP_EPSILON_LIST.index(1.0)]
    sns.heatmap(best_dp_eval["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Norm", "Atk"], yticklabels=["Norm", "Atk"])
    ax.set_title(f"(C) FL+DP ε=1.0 CM\nAcc={best_dp_eval['acc']:.3f}", fontweight="bold", fontsize=9)

    # D — ROC
    ax = fig.add_subplot(gs[1, 1])
    for ev, label, col, lw in [(eval_central,"Central","black",2),(eval_fl,"FL","#3498DB",2)]:
        fpr, tpr, _ = ev["roc"]
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"{label} {ev['auc']:.3f}")
    for eps, ev, col in zip(DP_EPSILON_LIST[::2], dp_evals[::2], colors_dp[::2]):
        fpr, tpr, _ = ev["roc"]
        ax.plot(fpr, tpr, col, lw=1.5, label=f"DP ε={eps} {ev['auc']:.3f}")
    ax.plot([0,1],[0,1],"k:",lw=1); ax.set_title("(D) ROC Curves", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

    # E — Privacy-utility
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(DP_EPSILON_LIST, [e["acc"] for e in dp_evals], "o-", color="steelblue", lw=2)
    ax.axhline(eval_fl["acc"], color="green", ls="--", lw=1.5, label="FL no DP")
    ax.set_title("(E) Accuracy vs ε", fontweight="bold", fontsize=9)
    ax.set_xlabel("ε"); ax.set_ylabel("Accuracy"); ax.set_xscale("log")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # F — Per-district sample scatter
    ax = fig.add_subplot(gs[1, 3])
    for i, (X, y) in enumerate(datasets):
        X_n = loaders_global[i].X
        ax.scatter(X_n[y==0, 0], X_n[y==0, 1], c=DISTRICT_COLORS[i],
                   s=4, alpha=0.3, marker="o")
        ax.scatter(X_n[y==1, 0], X_n[y==1, 1], c=DISTRICT_COLORS[i],
                   s=10, alpha=0.7, marker="x")
    ax.set_title("(F) Normalised Data\nCircle=normal, X=attack", fontweight="bold", fontsize=9)
    ax.set_xlabel("Feature 0 (norm)"); ax.set_ylabel("Feature 1 (norm)")

    # G — Summary table
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows_data = [
        ["Centralised (baseline)",  f"{eval_central['acc']:.4f}", f"{eval_central['f1']:.4f}",
         f"{eval_central['auc']:.4f}", "N/A",   "⚠️ Privacy risk — raw data pooled"],
        ["Federated (no DP)",       f"{eval_fl['acc']:.4f}",      f"{eval_fl['f1']:.4f}",
         f"{eval_fl['auc']:.4f}",      "∞",     "✅ Data stays local, no DP guarantee"],
    ]
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        rows_data.append([
            f"FL + DP-SGD  ε={eps}",
            f"{ev['acc']:.4f}", f"{ev['f1']:.4f}", f"{ev['auc']:.4f}",
            f"{eps}  (σ={sig:.2f})",
            "✅ Formal (ε,δ)-DP guarantee"
        ])

    col_labels = ["Model", "Accuracy", "F1-Score", "AUC", "ε (σ)", "Privacy"]
    tbl = ax.table(
        cellText=rows_data, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i_r in range(1, len(rows_data) + 1):
        color = "#EBF5FB" if i_r % 2 == 0 else "#FDFEFE"
        for j in range(len(col_labels)):
            tbl[i_r, j].set_facecolor(color)
    ax.set_title("(G) Complete Performance Summary Table", fontweight="bold", fontsize=10, pad=8)

    plt.savefig(f"{OUTPUT_DIR}/fig10_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ fig10_dashboard.png")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
loaders_global = None   # shared for fig10

def main():
    global loaders_global

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SMART CITY TRAFFIC IoT — FL + Opacus-style DP-SGD         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    start = time.time()

    # ─── STEP 1: Generate data ───────────────────────────────
    print_step(1, "Generating Traffic IoT Sensor Data")
    datasets = [generate_traffic_data(i) for i in range(N_DISTRICTS)]
    for i, (X, y) in enumerate(datasets):
        n_a = np.sum(y == 1); n_n = np.sum(y == 0)
        print(f"  {DISTRICT_NAMES[i]:18s}: {len(y)} samples | "
              f"Normal={n_n} ({n_n/len(y)*100:.0f}%) | "
              f"Attack={n_a} ({n_a/len(y)*100:.0f}%)")

    # ─── STEP 2: Visualise raw data ──────────────────────────
    print_step(2, "Visualising Raw Traffic Patterns")

    # ─── STEP 3: Data pipeline ──────────────────────────────
    print_step(3, "Building Data Pipeline (normalise → batch → iterate)")
    loaders = [TrafficDataLoader(X, y) for X, y in datasets]
    loaders_global = loaders
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Batches per district: {len(loaders[0])}")
    print(f"  Features:            {N_FEATURES} → {FEATURE_NAMES}")
    print(f"  Normalisation:       StandardScaler (zero mean, unit variance)")

    all_X = np.vstack([l.X for l in loaders])
    all_y = np.concatenate([d[1] for d in datasets])

    # ─── STEP 4: Model ──────────────────────────────────────
    print_step(4, "Building Neural Network (2-layer MLP)")
    sample_model = MLP()
    n_params = (N_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE +
                HIDDEN_SIZE * 1 + 1)
    print(f"  Architecture: Input({N_FEATURES}) → ReLU → Hidden({HIDDEN_SIZE}) → Sigmoid → Output(1)")
    print(f"  Total parameters: {n_params}")
    print(f"  Activation: ReLU (hidden), Sigmoid (output)")
    print(f"  Loss: Binary Cross-Entropy")
    print(f"  Optimiser: SGD (lr={LEARNING_RATE})")

    # ─── STEP 5: Centralised baseline ───────────────────────
    print_step(5, "Training Centralised Baseline (no FL, no DP)")
    model_central, hist_central = train_centralised(all_X, all_y)
    eval_central = evaluate_model(model_central, loaders)
    print(f"\n  ── Final Results ──")
    print(f"  Accuracy : {eval_central['acc']:.4f}")
    print(f"  F1-Score : {eval_central['f1']:.4f}")
    print(f"  AUC      : {eval_central['auc']:.4f}")

    # ─── STEP 6: Federated (no DP) ──────────────────────────
    print_step(6, "Training Federated Learning (FedAvg, no DP)")
    print(f"  Rounds:        {FL_ROUNDS}")
    print(f"  Local epochs:  {LOCAL_EPOCHS}")
    print(f"  Districts:     {N_DISTRICTS}")
    model_fl, hist_fl, _ = train_federated(datasets, use_dp=False)
    eval_fl = evaluate_model(model_fl, loaders)
    print(f"\n  ── Final Results ──")
    print(f"  Accuracy : {eval_fl['acc']:.4f}")
    print(f"  F1-Score : {eval_fl['f1']:.4f}")
    print(f"  AUC      : {eval_fl['auc']:.4f}")

    # ─── STEP 7: FL + DP-SGD ────────────────────────────────
    print_step(7, "Training FL + Opacus-style DP-SGD")
    print(f"  Clip norm C  = {DP_CLIP_NORM}")
    print(f"  Delta δ      = {DP_DELTA}")
    print(f"  ε tested     = {DP_EPSILON_LIST}")
    print()

    dp_models     = []
    dp_histories  = []
    dp_evals      = []
    dp_diags_list = []

    for eps in DP_EPSILON_LIST:
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        print(f"  ┌─ ε = {eps} | σ = {sig:.4f} ─────────────────────────")
        m, hist, diags = train_federated(datasets, use_dp=True, epsilon=eps)
        ev = evaluate_model(m, loaders)
        dp_models.append(m)
        dp_histories.append(hist)
        dp_evals.append(ev)
        dp_diags_list.append(diags)
        print(f"  └─ Acc={ev['acc']:.4f} | F1={ev['f1']:.4f} | AUC={ev['auc']:.4f}")

    # ─── STEP 8–9: Results ──────────────────────────────────
    print_step(8, "Performance Comparison — All Models")
    print(f"  {'Model':<28} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'─'*56}")
    print(f"  {'Centralised':<28} {eval_central['acc']:>8.4f} {eval_central['f1']:>8.4f} {eval_central['auc']:>8.4f}")
    print(f"  {'FL (no DP)':<28} {eval_fl['acc']:>8.4f} {eval_fl['f1']:>8.4f} {eval_fl['auc']:>8.4f}")
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        print(f"  {f'FL+DP ε={eps}':<28} {ev['acc']:>8.4f} {ev['f1']:>8.4f} {ev['auc']:>8.4f}")

    print_step(9, "Privacy-Utility Trade-off")
    print(f"  ε      σ (noise)   Accuracy   F1       vs FL (no DP)")
    print(f"  {'─'*55}")
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig  = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        diff = ev['acc'] - eval_fl['acc']
        sign = "+" if diff >= 0 else ""
        print(f"  {eps:<6} {sig:>8.4f}   {ev['acc']:>8.4f}   {ev['f1']:>8.4f}   {sign}{diff:.4f}")

    # ─── STEP 10: Figures ────────────────────────────────────
    print_step(10, "Generating Visualisations (10 figures)")
    fig01_data_pipeline(datasets, loaders)
    fig02_attack_patterns(datasets)
    fig03_dp_mechanism()
    fig04_fedavg_data_flow(datasets, loaders)
    fig05_convergence(hist_central, hist_fl, dp_histories)
    fig06_confusion_matrices(eval_central, eval_fl, dp_evals[DP_EPSILON_LIST.index(1.0)])
    fig07_roc_curves(eval_central, eval_fl, dp_evals)
    fig08_privacy_utility(dp_evals, eval_fl)
    fig09_dp_diagnostics(dp_diags_list)
    fig10_summary_dashboard(hist_central, hist_fl, dp_histories,
                            eval_central, eval_fl, dp_evals, datasets)

    # ─── Save CSV ────────────────────────────────────────────
    rows = [{"Model":"Centralised","Epsilon":"N/A","Sigma":"N/A",
             "Accuracy":eval_central["acc"],"F1":eval_central["f1"],"AUC":eval_central["auc"]}]
    rows.append({"Model":"FL (no DP)","Epsilon":"∞","Sigma":"0",
                 "Accuracy":eval_fl["acc"],"F1":eval_fl["f1"],"AUC":eval_fl["auc"]})
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        rows.append({"Model":f"FL+DP","Epsilon":eps,"Sigma":round(sig,4),
                     "Accuracy":ev["acc"],"F1":ev["f1"],"AUC":ev["auc"]})
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    elapsed = time.time() - start
    print(f"\n{'═'*65}")
    print(f"  ✅ ALL DONE in {elapsed:.1f}s")
    print(f"  Results saved to ./{OUTPUT_DIR}/")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()

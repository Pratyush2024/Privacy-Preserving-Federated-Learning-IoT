import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os, time, math, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              roc_curve, auc as sk_auc)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
np.random.seed(2024)

N_DISTRICTS      = 4
DISTRICT_NAMES   = ["North District", "South District", "East District", "West District"]
DISTRICT_COLORS  = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]
SAMPLES_PER_DIST = 500
N_FEATURES       = 10
ATTACK_RATIO     = 0.22

HIDDEN_SIZE      = 32
LEARNING_RATE    = 0.05
BATCH_SIZE       = 64
LOCAL_EPOCHS     = 3
FL_ROUNDS        = 25

DP_CLIP_NORM     = 1.0
DP_DELTA         = 1e-5
DP_EPSILON_LIST  = [0.1, 0.5, 1.0, 2.0, 5.0]

OUTPUT_DIR       = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "vehicle_count", "avg_speed_kmh", "occupancy_pct",
    "queue_length", "red_light_violations", "sensor_noise_level",
    "time_of_day_sin", "time_of_day_cos", "is_weekend", "incident_nearby"
]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def section(n, title):
    print(f"\n{'═'*65}")
    print(f"  STEP {n}: {title}")
    print(f"{'═'*65}")

# ══════════════════════════════════════════════
#  STEP 1 — DATA GENERATION
# ══════════════════════════════════════════════
def generate_traffic_data(district_idx):
    """
    Realistic non-IID traffic sensor data for one city district.

    Each district has its own baseline traffic profile:
      North — busy commuter corridor  (high vehicle count, slow speed)
      South — residential area        (light traffic, faster speeds)
      East  — commercial/industrial   (high occupancy, many violations)
      West  — highway junction        (fast, variable queue)

    Three attack types are simulated:
      Type 0 — Sensor blind attack  (near-zero count, impossible speed)
      Type 1 — Max-spoof attack     (counts pinned to maximum)
      Type 2 — Subtle manipulation  (slight drift, hard to detect)

    The key attack signature is always elevated sensor_noise_level
    (caused by RF jamming or hardware interference).
    """
    baselines = {
        0: dict(vc=85,  spd=28, occ=62, ql=180, rlv=4.2, noise=0.12),
        1: dict(vc=42,  spd=55, occ=31, ql=60,  rlv=1.8, noise=0.08),
        2: dict(vc=120, spd=18, occ=78, ql=340, rlv=9.1, noise=0.15),
        3: dict(vc=68,  spd=95, occ=44, ql=90,  rlv=2.5, noise=0.10),
    }
    b = baselines[district_idx]
    n_attack = int(SAMPLES_PER_DIST * ATTACK_RATIO)
    n_normal = SAMPLES_PER_DIST - n_attack

    # ── Normal readings ──
    hours = np.random.uniform(0, 24, n_normal)
    X_normal = np.column_stack([
        np.random.normal(b["vc"],   15,        n_normal).clip(0, 200),
        np.random.normal(b["spd"],  12,        n_normal).clip(0, 130),
        np.random.normal(b["occ"],  10,        n_normal).clip(0, 100),
        np.random.normal(b["ql"],   60,        n_normal).clip(0, 800),
        np.random.poisson(b["rlv"],            n_normal).clip(0, 30),
        np.random.exponential(b["noise"],      n_normal).clip(0, 2),
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.random.binomial(1, 0.29,            n_normal).astype(float),
        np.random.binomial(1, 0.05,            n_normal).astype(float),
    ])

    # ── Attack readings ──
    hours_a     = np.random.uniform(0, 24, n_attack)
    attack_type = np.random.choice(3, n_attack)
    vc_a   = np.where(attack_type == 0,
                 np.random.normal(b["vc"] * 0.05, 5,  n_attack),
                 np.where(attack_type == 1,
                     np.random.normal(200, 10, n_attack),
                     np.random.normal(b["vc"], 15, n_attack)))
    spd_a  = np.where(attack_type == 0,
                 np.random.normal(130, 5, n_attack),
                 np.random.normal(b["spd"], 8, n_attack))
    occ_a  = np.where(attack_type == 1,
                 np.random.normal(95, 3, n_attack),
                 np.random.normal(b["occ"], 8, n_attack))
    noise_a = np.random.exponential(b["noise"] * 8, n_attack)   # always elevated

    X_attack = np.column_stack([
        vc_a.clip(0, 200),
        spd_a.clip(0, 130),
        occ_a.clip(0, 100),
        np.random.normal(b["ql"] * 0.5 + 200, 100, n_attack).clip(0, 800),
        np.random.poisson(b["rlv"] * 2.5, n_attack).clip(0, 30),
        noise_a.clip(0, 2),
        np.sin(2 * np.pi * hours_a / 24),
        np.cos(2 * np.pi * hours_a / 24),
        np.random.binomial(1, 0.29, n_attack).astype(float),
        np.random.binomial(1, 0.15, n_attack).astype(float),
    ])

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_attack)]).astype(int)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# ══════════════════════════════════════════════
#  STEP 2 — DATA LOADER
# ══════════════════════════════════════════════
class TrafficDataLoader:
    """
    Normalises features with StandardScaler, then yields mini-batches.
    Mirrors the PyTorch DataLoader interface used by Opacus.

    Usage:
        loader = TrafficDataLoader(X_raw, y)
        for X_batch, y_batch in loader:
            ...  # train on batch
    """
    def __init__(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        self.scaler     = StandardScaler()
        self.X          = self.scaler.fit_transform(X)
        self.y          = y.copy()
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n          = len(y)

    def __iter__(self):
        idx = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield self.X[b], self.y[b]

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


# ══════════════════════════════════════════════
#  STEP 3 — NEURAL NETWORK (2-layer MLP)
# ══════════════════════════════════════════════
class MLP:
    """
    2-layer MLP built from scratch with NumPy.

    Architecture:
        Input(10) ──► Linear ──► ReLU ──► Linear ──► Sigmoid ──► Output(1)
                        W1,b1             W2,b2

    This mirrors a PyTorch nn.Sequential model and is compatible
    with the Opacus-style per-sample gradient computation below.
    """
    def __init__(self):
        self.W1 = np.random.randn(N_FEATURES, HIDDEN_SIZE) * np.sqrt(2.0 / N_FEATURES)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, 1) * np.sqrt(2.0 / HIDDEN_SIZE)
        self.b2 = np.zeros(1)

    # ── Forward pass ──
    def forward(self, X):
        self.z1  = X @ self.W1 + self.b1
        self.a1  = np.maximum(0, self.z1)          # ReLU
        self.z2  = self.a1 @ self.W2 + self.b2
        self.out = 1.0 / (1.0 + np.exp(-self.z2.clip(-500, 500)))   # Sigmoid
        return self.out.ravel()

    # ── Standard backward pass (batch gradient, used by centralised training) ──
    def backward(self, X, y, lr=LEARNING_RATE):
        n    = len(y)
        prob = self.forward(X)
        dL   = (prob - y) / n
        dW2  = self.a1.T @ dL.reshape(-1, 1)
        db2  = float(dL.sum())
        da1  = dL.reshape(-1, 1) * self.W2.T
        dz1  = da1 * (self.z1 > 0)
        dW1  = X.T @ dz1
        db1  = dz1.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return self._loss(prob, y)

    # ── Per-sample gradients (Opacus ghost-clipping equivalent) ──
    def per_sample_gradients(self, X, y):
        """
        Computes one gradient dict per training sample.
        In Opacus, this is done via PyTorch hooks (grad_sample).
        Here we loop explicitly for clarity.

        Returns: list of dicts, each with keys W1, b1, W2, b2
        """
        grads = []
        for xi, yi in zip(X, y):
            xi   = xi.reshape(1, -1)
            prob = self.forward(xi)
            dL   = float(prob.ravel()[0] - yi)   # scalar
            dW2  = self.a1.T * dL
            db2  = np.array([dL])
            da1  = dL * self.W2.T
            dz1  = da1 * (self.z1 > 0)
            dW1  = xi.T @ dz1
            db1  = dz1.ravel()
            grads.append({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
        return grads

    # ── Parameter helpers ──
    def get_params(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_params(self, p):
        self.W1 = p["W1"].copy(); self.b1 = p["b1"].copy()
        self.W2 = p["W2"].copy(); self.b2 = p["b2"].copy()

    def clone(self):
        m = MLP(); m.set_params(self.get_params()); return m

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X)

    def loss(self, X, y):
        return self._loss(self.forward(X), y)

    @staticmethod
    def _loss(prob, y):
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))


# ══════════════════════════════════════════════
#  OPACUS-STYLE DP ENGINE
# ══════════════════════════════════════════════
class OPACUSStyleDP:
    """
    Implements the Opacus PrivacyEngine algorithm for numpy models.

    ┌─────────────────────────────────────────────────────────────┐
    │  Opacus DP-SGD pipeline (one mini-batch):                   │
    │                                                             │
    │  1. Compute per-sample gradients  ← ghost clipping / hooks  │
    │                                                             │
    │  2. Clip each gradient to L2 norm C:                        │
    │       g̃ᵢ = gᵢ / max(1,  ‖gᵢ‖₂ / C)                       │
    │                                                             │
    │  3. Sum clipped gradients:   Σ g̃ᵢ                          │
    │                                                             │
    │  4. Add Gaussian noise:                                     │
    │       noise ~ N(0, σ²C²I)                                   │
    │       σ = √(2 ln(1.25/δ)) / ε     (Dwork et al. 2014)      │
    │                                                             │
    │  5. Divide by batch size  →  unbiased noisy gradient        │
    │                                                             │
    │  6. Apply gradient descent update                           │
    └─────────────────────────────────────────────────────────────┘

    Args:
        model     : MLP instance
        epsilon   : privacy budget ε  (smaller = more private)
        delta     : failure probability δ (typically 1e-5)
        clip_norm : gradient clipping threshold C
        lr        : learning rate
    """

    def __init__(self, model, epsilon, delta=DP_DELTA,
                 clip_norm=DP_CLIP_NORM, lr=LEARNING_RATE):
        self.model  = model
        self.C      = clip_norm
        self.lr     = lr
        self.sigma  = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        self.steps  = 0

    # ── Utility: L2 norm of a gradient dict ──
    def _l2(self, g):
        return math.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))

    def step(self, X_batch, y_batch):
        """
        One DP-SGD step. Returns (loss, mean_clip_norm, mean_noise_norm).
        The diagnostics are useful for verifying the DP mechanism is working.
        """
        n = len(y_batch)

        # 1. Per-sample gradients
        per_grads = self.model.per_sample_gradients(X_batch, y_batch)

        # 2. Clip each gradient
        clipped, clip_norms = [], []
        for g in per_grads:
            norm = self._l2(g)
            clip_norms.append(norm)
            scale = min(1.0, self.C / (norm + 1e-12))
            clipped.append({k: v * scale for k, v in g.items()})

        # 3. Sum
        agg = {k: np.zeros_like(v) for k, v in clipped[0].items()}
        for g in clipped:
            for k in agg:
                agg[k] += g[k]

        # 4. Add Gaussian noise  (σ * C  per coordinate)
        noisy, noise_norms = {}, []
        for k, v in agg.items():
            noise    = np.random.normal(0, self.sigma * self.C, size=v.shape)
            noisy[k] = v + noise
            noise_norms.append(float(np.linalg.norm(noise)))

        # 5. Average over batch
        avg = {k: v / n for k, v in noisy.items()}

        # 6. Gradient descent
        p = self.model.get_params()
        for k in p:
            p[k] -= self.lr * avg[k]
        self.model.set_params(p)

        self.steps += 1
        loss = self.model.loss(X_batch, y_batch)
        return loss, float(np.mean(clip_norms)), float(np.mean(noise_norms))


# ══════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ══════════════════════════════════════════════
def train_centralised(all_X, all_y):
    """Baseline: pool all district data and train normally."""
    model   = MLP()
    loader  = TrafficDataLoader(all_X, all_y)
    history = []
    for rnd in range(FL_ROUNDS):
        batch_losses = []
        for Xb, yb in loader:
            loss = model.backward(Xb, yb)
            batch_losses.append(loss)
        Xa, ya  = loader.X, loader.y
        acc     = accuracy_score(ya, model.predict(Xa))
        f1      = f1_score(ya, model.predict(Xa), zero_division=0)
        history.append((acc, f1, float(np.mean(batch_losses))))
        if (rnd + 1) % 5 == 0:
            print(f"    Round {rnd+1:2d}/{FL_ROUNDS}  "
                  f"loss={history[-1][2]:.4f}  acc={acc:.4f}  f1={f1:.4f}")
    return model, history


def fedavg_aggregate(global_params, deltas, weights):
    """
    FedAvg weighted aggregation:
        new_global = global + Σ (wᵢ × Δᵢ)
    where wᵢ = nᵢ / Σnᵢ  (proportional to dataset size)
    """
    agg = {k: np.zeros_like(v) for k, v in global_params.items()}
    for delta, w in zip(deltas, weights):
        for k in agg:
            agg[k] += w * delta[k]
    return {k: global_params[k] + agg[k] for k in global_params}


def train_federated(datasets, use_dp=False, epsilon=1.0):
    """
    Federated training across N_DISTRICTS districts.

    Data flow each round:
    ┌────────────────────────────────────────────────────────┐
    │  Server  ──── broadcasts global model ────►            │
    │                                                        │
    │  District 0  local train  (+ DP-SGD if enabled)        │
    │  District 1  local train  (+ DP-SGD if enabled)        │
    │  District 2  local train  (+ DP-SGD if enabled)        │
    │  District 3  local train  (+ DP-SGD if enabled)        │
    │                                                        │
    │     ◄──── sends Δmodel (noisy if DP) ────  each dist.  │
    │                                                        │
    │  Server  FedAvg(Δ₀, Δ₁, Δ₂, Δ₃) → new global model   │
    └────────────────────────────────────────────────────────┘
    Raw sensor data NEVER leaves each district.
    """
    tag     = f"FL+DP(ε={epsilon})" if use_dp else "FL (no DP)"
    gm      = MLP()
    loaders = [TrafficDataLoader(X, y) for X, y in datasets]
    all_X   = np.vstack([l.X for l in loaders])
    all_y   = np.concatenate([l.y for l in loaders])
    total_n = sum(len(d[1]) for d in datasets)
    weights = [len(d[1]) / total_n for d in datasets]

    history  = []
    dp_diags = []      # (mean_clip_norm, mean_noise_norm) per round

    for rnd in range(FL_ROUNDS):
        gp      = gm.get_params()
        deltas  = []
        rd_diag = []

        for i, (loader, (_, y)) in enumerate(zip(loaders, datasets)):
            lm = gm.clone()

            if use_dp:
                dp = OPACUSStyleDP(lm, epsilon=epsilon)
                for _ in range(LOCAL_EPOCHS):
                    for Xb, yb in loader:
                        _, cn, ns = dp.step(Xb, yb)
                        rd_diag.append((cn, ns))
            else:
                for _ in range(LOCAL_EPOCHS):
                    for Xb, yb in loader:
                        lm.backward(Xb, yb)

            lp = lm.get_params()
            deltas.append({k: lp[k] - gp[k] for k in gp})

        gm.set_params(fedavg_aggregate(gp, deltas, weights))

        acc  = accuracy_score(all_y, gm.predict(all_X))
        f1   = f1_score(all_y, gm.predict(all_X), zero_division=0)
        loss = gm.loss(all_X, all_y)
        history.append((acc, f1, loss))

        if rd_diag:
            dp_diags.append((float(np.mean([d[0] for d in rd_diag])),
                             float(np.mean([d[1] for d in rd_diag]))))

        if (rnd + 1) % 5 == 0:
            dp_info = f"  noise={dp_diags[-1][1]:.2f}" if use_dp else ""
            print(f"    Round {rnd+1:2d}/{FL_ROUNDS}  "
                  f"loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}{dp_info}")

    return gm, history, dp_diags


def evaluate(model, loaders):
    X     = np.vstack([l.X for l in loaders])
    y     = np.concatenate([l.y for l in loaders])
    preds = model.predict(X)
    proba = model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "acc":   float(accuracy_score(y, preds)),
        "f1":    float(f1_score(y, preds, zero_division=0)),
        "auc":   float(sk_auc(fpr, tpr)),
        "cm":    confusion_matrix(y, preds),
        "fpr":   fpr,
        "tpr":   tpr,
        "y":     y,
        "proba": proba,
    }


# ══════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════
def colors_dp():
    return plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(DP_EPSILON_LIST)))


# ── Figure 1: DP Mechanism ──
def fig_dp_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Opacus-Style DP-SGD Mechanism", fontsize=14, fontweight="bold")

    # A: Gradient clipping
    ax = axes[0]
    grads   = np.random.randn(200, 2) * 1.5
    norms   = np.linalg.norm(grads, axis=1, keepdims=True)
    clipped = grads / np.maximum(1.0, norms / DP_CLIP_NORM)
    ax.scatter(grads[:, 0],   grads[:, 1],   c="tomato",    s=15, alpha=0.5, label="Original")
    ax.scatter(clipped[:, 0], clipped[:, 1], c="steelblue", s=15, alpha=0.5, label="Clipped")
    circle = plt.Circle((0, 0), DP_CLIP_NORM, color="black",
                         fill=False, linestyle="--", lw=1.5, label=f"Clip norm C={DP_CLIP_NORM}")
    ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title("(A) Gradient Clipping\nbounds sensitivity", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("∂L/∂w₁"); ax.set_ylabel("∂L/∂w₂")

    # B: σ vs ε curve
    ax = axes[1]
    eps_range = np.linspace(0.05, 6.0, 200)
    sigmas    = [math.sqrt(2 * math.log(1.25 / DP_DELTA)) / e for e in eps_range]
    ax.plot(eps_range, sigmas, "navy", lw=2)
    cdp = colors_dp()
    for eps, col in zip(DP_EPSILON_LIST, cdp):
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        ax.scatter([eps], [sig], s=80, zorder=5, color=col)
        ax.annotate(f"ε={eps}\nσ={sig:.2f}", (eps, sig),
                    textcoords="offset points", xytext=(8, 5), fontsize=7)
    ax.set_xlabel("Privacy Budget ε"); ax.set_ylabel("Noise Multiplier σ")
    ax.set_title("(B) Noise Calibration\nσ = √(2·ln(1.25/δ)) / ε", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 6.5)

    # C: Effect of noise on a 1-D gradient
    ax = axes[2]
    true_g = np.linspace(-2, 2, 300)
    ax.plot(true_g, true_g, "k-", lw=2, label="True gradient (no DP)")
    for eps, col in zip([5.0, 1.0, 0.1], ["#27AE60", "#F39C12", "#E74C3C"]):
        sig   = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        noisy = true_g + np.random.normal(0, sig * DP_CLIP_NORM, len(true_g))
        ax.scatter(true_g[::10], noisy[::10], s=12, alpha=0.6,
                   color=col, label=f"ε={eps} (σ={sig:.2f})")
    ax.set_xlabel("True gradient"); ax.set_ylabel("Noisy gradient")
    ax.set_title("(C) Gradient Corruption\nby DP Noise", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig01_dp_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig01_dp_mechanism.png")


# ── Figure 2: Data Flow Diagram ──
def fig_data_flow(datasets):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off"); fig.patch.set_facecolor("#F0F4F8")
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.set_title("Federated Learning Data Flow\n"
                 "Raw data stays local — only DP-noised model updates travel to server",
                 fontsize=13, fontweight="bold", pad=15)

    # Central server
    ax.add_patch(plt.Rectangle((5.5, 4.3), 3, 1.4, color="#2C3E50", ec="black", lw=2, zorder=3))
    ax.text(7, 5.1, "CENTRAL SERVER\n(FedAvg Aggregation)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)

    positions = [(1.0, 7.5), (10.5, 7.5), (1.0, 1.5), (10.5, 1.5)]
    for i, (px, py) in enumerate(positions):
        ax.add_patch(plt.Rectangle((px, py), 2.8, 1.8,
                                   color=DISTRICT_COLORS[i], ec="black", lw=1.5, zorder=3, alpha=0.85))
        n_att = int(np.sum(datasets[i][1] == 1))
        n_nor = int(np.sum(datasets[i][1] == 0))
        ax.text(px + 1.4, py + 1.2, DISTRICT_NAMES[i],
                ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
        ax.text(px + 1.4, py + 0.5,
                f"Normal: {n_nor}   Attack: {n_att}",
                ha="center", va="center", fontsize=7, color="white", zorder=4)
        ax.text(px + 0.3, py + 1.4, "🔒", fontsize=13, zorder=5)

        cx  = px + 1.4
        cy  = py + 1.8 if py > 4 else py
        sx  = 7.0
        sy  = 5.7 if py > 4 else 4.3

        # update arrow → server
        ax.annotate("", xy=(sx, sy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#2980B9", lw=2.0))
        ax.text((cx + sx) / 2 + 0.1, (cy + sy) / 2 + 0.2,
                "Δ model\n(+ DP noise)", fontsize=7, color="#2980B9", ha="center")

        # global model ← server
        ax.annotate("", xy=(cx, cy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5,
                                   connectionstyle="arc3,rad=0.35"))

    ax.text(7, 6.9, "← Global model broadcast", ha="center",
            fontsize=8, color="#27AE60", fontstyle="italic")

    ax.legend(handles=[
        mpatches.Patch(color="#2980B9", label="Noisy model update (DP-clipped + Gaussian noise)"),
        mpatches.Patch(color="#27AE60", label="Global model broadcast"),
        mpatches.Patch(color="gray",   label="🔒 Raw sensor data never leaves district"),
    ], loc="lower center", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig02_data_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig02_data_flow.png")


# ── Figure 3: Attack Pattern Comparison ──
def fig_attack_patterns(datasets):
    features_to_plot = [
        (0, "vehicle_count"),
        (1, "avg_speed_kmh"),
        (2, "occupancy_pct"),
        (5, "sensor_noise_level"),
    ]
    fig, axes = plt.subplots(N_DISTRICTS, len(features_to_plot), figsize=(16, 10))
    fig.suptitle("Attack vs Normal Traffic Sensor Patterns per District",
                 fontsize=14, fontweight="bold")

    for row, (X, y) in enumerate(datasets):
        for col, (fi, fname) in enumerate(features_to_plot):
            ax = axes[row][col]
            ax.hist(X[y == 0, fi], bins=30, color="#2ECC71", alpha=0.65,
                    label="Normal", density=True)
            ax.hist(X[y == 1, fi], bins=30, color="#E74C3C", alpha=0.65,
                    label="Attack", density=True)
            if row == 0:
                ax.set_title(fname, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(DISTRICT_NAMES[row][:5], fontsize=9, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig03_attack_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig03_attack_patterns.png")


# ── Figure 4: Convergence ──
def fig_convergence(hist_c, hist_fl, dp_histories):
    rounds = range(1, FL_ROUNDS + 1)
    cdp    = colors_dp()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Training Convergence — Centralised vs FL vs FL+DP",
                 fontsize=13, fontweight="bold")

    for ax, metric_idx, label in zip(axes, [0, 1, 2], ["Accuracy", "F1-Score", "Loss"]):
        ax.plot(rounds, [h[metric_idx] for h in hist_c],
                "k--", lw=2.5, label="Centralised")
        ax.plot(rounds, [h[metric_idx] for h in hist_fl],
                color="#3498DB", lw=2.5, label="FL (no DP)")
        for eps, hist, col in zip(DP_EPSILON_LIST, dp_histories, cdp):
            ax.plot(rounds, [h[metric_idx] for h in hist],
                    color=col, lw=1.5, alpha=0.9, label=f"FL+DP ε={eps}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over Rounds", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric_idx in (0, 1):
            ax.set_ylim(0.4, 1.02)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig04_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig04_convergence.png")


# ── Figure 5: Confusion Matrices ──
def fig_confusion(ev_c, ev_fl, dp_evals):
    titles = ["Centralised", "FL (no DP)", "FL+DP (ε=1.0)",
              "FL+DP (ε=2.0)", "FL+DP (ε=5.0)"]
    evals  = [ev_c, ev_fl,
              dp_evals[DP_EPSILON_LIST.index(1.0)],
              dp_evals[DP_EPSILON_LIST.index(2.0)],
              dp_evals[DP_EPSILON_LIST.index(5.0)]]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Confusion Matrices — Cyberattack Detection",
                 fontsize=13, fontweight="bold")

    for ax, ev, title in zip(axes, evals, titles):
        sns.heatmap(ev["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_title(f"{title}\nAcc={ev['acc']:.3f}  F1={ev['f1']:.3f}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig05_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig05_confusion_matrices.png")


# ── Figure 6: ROC Curves ──
def fig_roc(ev_c, ev_fl, dp_evals):
    cdp = colors_dp()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves — Traffic Cyberattack Detection", fontsize=13, fontweight="bold")

    for ev, label, color, lw, ls in [
        (ev_c,  "Centralised", "black",   2.5, "--"),
        (ev_fl, "FL (no DP)", "#3498DB", 2.5, "-"),
    ]:
        ax.plot(ev["fpr"], ev["tpr"], color=color, lw=lw, ls=ls,
                label=f"{label}  (AUC = {ev['auc']:.4f})")

    for eps, ev, col in zip(DP_EPSILON_LIST, dp_evals, cdp):
        ax.plot(ev["fpr"], ev["tpr"], color=col, lw=1.5, alpha=0.9,
                label=f"FL+DP  ε={eps}  (AUC = {ev['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate  (Normal classified as Attack)")
    ax.set_ylabel("True Positive Rate  (Attack correctly detected)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig06_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig06_roc_curves.png")


# ── Figure 7: Privacy-Utility Trade-off ──
def fig_privacy_utility(dp_evals, ev_fl):
    cdp = colors_dp()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Privacy–Utility Trade-off  (ε vs model performance)",
                 fontsize=13, fontweight="bold")

    # Reference values from FL (no DP) — used as horizontal baselines
    fl_ref = {"Accuracy": ev_fl["acc"], "F1-Score": ev_fl["f1"], "AUC": ev_fl["auc"]}
    dp_vals = {
        "Accuracy": [e["acc"] for e in dp_evals],
        "F1-Score": [e["f1"]  for e in dp_evals],
        "AUC":      [e["auc"] for e in dp_evals],
    }
    plot_colors = {"Accuracy": "steelblue", "F1-Score": "darkorange", "AUC": "purple"}

    for ax, metric in zip(axes, ["Accuracy", "F1-Score", "AUC"]):
        vals = dp_vals[metric]
        color = plot_colors[metric]

        ax.plot(DP_EPSILON_LIST, vals, "o-", color=color,
                lw=2, markersize=9, label="FL + DP-SGD")
        ax.fill_between(DP_EPSILON_LIST, vals, alpha=0.12, color=color)

        # FL (no DP) baseline — simple fixed reference, no messy key manipulation
        ax.axhline(fl_ref[metric], color="green", ls="--", lw=1.5,
                   label=f"FL no DP ({fl_ref[metric]:.4f})")

        # Annotate each point with its ε value
        for eps, v, col in zip(DP_EPSILON_LIST, vals, cdp):
            ax.annotate(f"ε={eps}", (eps, v),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center", color="black")

        ax.set_xlabel("Privacy Budget ε  (← stronger privacy | weaker privacy →)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs ε", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig07_privacy_utility.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fig07_privacy_utility.png")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Smart City Traffic IoT — FL + Opacus-style DP-SGD          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    t0 = time.time()

    # ── Step 1: Generate data ──────────────────────────────────────
    section(1, "Generating Traffic Sensor Data  (4 districts, non-IID)")
    datasets = [generate_traffic_data(i) for i in range(N_DISTRICTS)]
    for i, (X, y) in enumerate(datasets):
        n_a = int(np.sum(y == 1))
        n_n = int(np.sum(y == 0))
        print(f"  {DISTRICT_NAMES[i]:18s} │ samples={len(y)} │ "
              f"normal={n_n} ({n_n/len(y)*100:.0f}%) │ "
              f"attack={n_a} ({n_a/len(y)*100:.0f}%)")

    print(f"\n  Features ({N_FEATURES}):")
    for name in FEATURE_NAMES:
        print(f"    • {name}")

    # ── Step 2: Data loaders ────────────────────────────────────────
    section(2, "Building Data Loaders  (StandardScaler → mini-batches)")
    loaders = [TrafficDataLoader(X, y) for X, y in datasets]
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Batches/dist  : {len(loaders[0])}")
    print(f"  Normalisation : StandardScaler — zero mean, unit variance per feature")

    # ── Step 3: Model ───────────────────────────────────────────────
    section(3, "Neural Network Architecture")
    n_params = N_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1
    print(f"  Input({N_FEATURES}) → Linear → ReLU → Hidden({HIDDEN_SIZE}) → Linear → Sigmoid → Output(1)")
    print(f"  Total trainable parameters : {n_params}")
    print(f"  Loss function              : Binary Cross-Entropy")
    print(f"  Optimiser                  : SGD  (lr={LEARNING_RATE})")

    # ── Step 4: Centralised baseline ───────────────────────────────
    section(4, "Training Centralised Baseline  (no FL, no DP)")
    all_X = np.vstack([l.X for l in loaders])
    all_y = np.concatenate([l.y for l in loaders])
    model_c, hist_c = train_centralised(all_X, all_y)
    ev_c = evaluate(model_c, loaders)
    print(f"\n  ── Centralised Final ──")
    print(f"  Accuracy : {ev_c['acc']:.4f}")
    print(f"  F1-Score : {ev_c['f1']:.4f}")
    print(f"  AUC      : {ev_c['auc']:.4f}")

    # ── Step 5: Federated (no DP) ───────────────────────────────────
    section(5, "Training Federated Learning  (FedAvg, no DP)")
    print(f"  Districts      : {N_DISTRICTS}")
    print(f"  Rounds         : {FL_ROUNDS}")
    print(f"  Local epochs   : {LOCAL_EPOCHS}")
    model_fl, hist_fl, _ = train_federated(datasets, use_dp=False)
    ev_fl = evaluate(model_fl, loaders)
    print(f"\n  ── FL Final ──")
    print(f"  Accuracy : {ev_fl['acc']:.4f}")
    print(f"  F1-Score : {ev_fl['f1']:.4f}")
    print(f"  AUC      : {ev_fl['auc']:.4f}")

    # ── Step 6: FL + DP-SGD ─────────────────────────────────────────
    section(6, "Training FL + Opacus-Style DP-SGD")
    print(f"  Clip norm  C = {DP_CLIP_NORM}")
    print(f"  Delta      δ = {DP_DELTA}")
    print(f"  Testing ε  ∈ {DP_EPSILON_LIST}")

    dp_models    = []
    dp_histories = []
    dp_evals     = []
    dp_diags     = []

    for eps in DP_EPSILON_LIST:
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        print(f"\n  ┌─── ε = {eps}  │  σ = {sig:.4f}  │  "
              f"C = {DP_CLIP_NORM}  │  δ = {DP_DELTA} ───────────")
        m, hist, diag = train_federated(datasets, use_dp=True, epsilon=eps)
        ev = evaluate(m, loaders)
        dp_models.append(m); dp_histories.append(hist)
        dp_evals.append(ev); dp_diags.append(diag)
        diff = ev["acc"] - ev_fl["acc"]
        sign = "+" if diff >= 0 else ""
        print(f"  └─── Acc={ev['acc']:.4f}  F1={ev['f1']:.4f}  "
              f"AUC={ev['auc']:.4f}  Δacc vs FL={sign}{diff:.4f}")

    # ── Step 7: Results Table ────────────────────────────────────────
    section(7, "Performance Summary")
    header = f"  {'Model':<28} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>8}  Privacy"
    divider = "  " + "─" * 72
    print(header); print(divider)
    print(f"  {'Centralised (pooled)':<28} {ev_c['acc']:>10.4f} "
          f"{ev_c['f1']:>10.4f} {ev_c['auc']:>8.4f}  ⚠ No privacy — raw data pooled")
    print(f"  {'FL (no DP)':<28} {ev_fl['acc']:>10.4f} "
          f"{ev_fl['f1']:>10.4f} {ev_fl['auc']:>8.4f}  ✔ Data local, no formal DP")
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        print(f"  {f'FL + DP-SGD  ε={eps}':<28} {ev['acc']:>10.4f} "
              f"{ev['f1']:>10.4f} {ev['auc']:>8.4f}  ✔ (ε,δ)-DP  σ={sig:.3f}")
    print(divider)

    section(7, "Privacy–Utility Trade-off Detail")
    print(f"  {'ε':<6} {'σ (noise)':>10} {'Accuracy':>10} {'F1':>8}  {'vs FL (no DP)':>14}")
    print("  " + "─" * 55)
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig  = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        diff = ev["acc"] - ev_fl["acc"]
        sign = "+" if diff >= 0 else ""
        print(f"  {eps:<6} {sig:>10.4f} {ev['acc']:>10.4f} {ev['f1']:>8.4f}  {sign}{diff:.4f}")

    # ── Step 8: Figures ──────────────────────────────────────────────
    section(8, "Generating Result Figures  (7 figures)")
    fig_dp_mechanism()
    fig_data_flow(datasets)
    fig_attack_patterns(datasets)
    fig_convergence(hist_c, hist_fl, dp_histories)
    fig_confusion(ev_c, ev_fl, dp_evals)
    fig_roc(ev_c, ev_fl, dp_evals)
    fig_privacy_utility(dp_evals, ev_fl)

    # Save results CSV
    rows = [
        {"Model": "Centralised",  "Epsilon": "N/A", "Sigma": "N/A",
         "Accuracy": ev_c["acc"], "F1": ev_c["f1"], "AUC": ev_c["auc"]},
        {"Model": "FL (no DP)",   "Epsilon": "∞",   "Sigma": "0",
         "Accuracy": ev_fl["acc"],"F1": ev_fl["f1"],"AUC": ev_fl["auc"]},
    ]
    for eps, ev in zip(DP_EPSILON_LIST, dp_evals):
        sig = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / eps
        rows.append({"Model": "FL+DP", "Epsilon": eps, "Sigma": round(sig, 4),
                     "Accuracy": ev["acc"], "F1": ev["f1"], "AUC": ev["auc"]})
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'═'*65}")
    print(f"  ✅  Finished in {elapsed:.1f}s  —  results saved to ./{OUTPUT_DIR}/")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()

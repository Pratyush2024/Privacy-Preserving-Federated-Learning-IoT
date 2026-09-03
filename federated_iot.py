"""
=============================================================
  Smart City IoT Security: Federated Learning + Differential Privacy
  Course: Design of Smart Cities
=============================================================
  Scenario:
    - 5 IoT zones in a smart city (traffic, energy, water, health, environment)
    - Each zone has local sensor data with anomalies (cyberattacks)
    - We train a local anomaly-detection model per zone
    - Federated Aggregation (FedAvg) combines models WITHOUT sharing raw data
    - Differential Privacy (Gaussian noise) is applied to model updates
    - We compare: Centralized vs Federated vs FL+DP performance
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import os

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
np.random.seed(42)

N_CLIENTS = 5
ZONE_NAMES = ["Traffic", "Energy", "Water", "Health", "Environment"]
SAMPLES_PER_CLIENT = 300
N_FEATURES = 8
N_ROUNDS = 20
LEARNING_RATE = 0.05

# Differential Privacy budget
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]   # privacy budget ε
DP_CLIP_NORM = 1.0                              # gradient clipping threshold

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA GENERATION  (simulated IoT sensor readings)
# ─────────────────────────────────────────────
def generate_iot_data(zone_idx, n_samples=SAMPLES_PER_CLIENT):
    """
    Generate synthetic IoT sensor data for a smart city zone.
    Features represent normalised sensor readings.
    Label: 0 = normal, 1 = anomaly / cyberattack
    Each zone has a slightly different data distribution (non-IID).
    """
    # Base normal readings (zone-specific mean)
    zone_bias = zone_idx * 0.15
    X_normal = np.random.randn(int(n_samples * 0.8), N_FEATURES) * 0.5 + zone_bias
    y_normal = np.zeros(len(X_normal))

    # Anomaly readings (attack pattern: higher values, more variance)
    X_anomaly = np.random.randn(int(n_samples * 0.2), N_FEATURES) * 1.5 + zone_bias + 2.0
    y_anomaly = np.ones(len(X_anomaly))

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx].astype(int)


# ─────────────────────────────────────────────
# 2. LOGISTIC REGRESSION MODEL (from scratch)
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def init_model(n_features=N_FEATURES):
    return {"W": np.zeros(n_features), "b": np.float64(0.0)}

def copy_model(model):
    return {"W": model["W"].copy(), "b": model["b"].copy()}

def predict(model, X):
    prob = sigmoid(X @ model["W"] + model["b"])
    return (prob >= 0.5).astype(int)

def compute_loss(model, X, y):
    prob = sigmoid(X @ model["W"] + model["b"])
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

def train_local(model, X, y, lr=LEARNING_RATE, epochs=5):
    """One round of local SGD training."""
    m = copy_model(model)
    n = len(y)
    for _ in range(epochs):
        prob = sigmoid(X @ m["W"] + m["b"])
        grad_W = (X.T @ (prob - y)) / n
        grad_b = np.mean(prob - y)
        m["W"] -= lr * grad_W
        m["b"] -= lr * grad_b
    return m

def model_delta(global_model, local_model):
    """Compute the update (delta) from global to local model."""
    return {
        "W": local_model["W"] - global_model["W"],
        "b": local_model["b"] - global_model["b"]
    }

# ─────────────────────────────────────────────
# 3. DIFFERENTIAL PRIVACY
# ─────────────────────────────────────────────
def clip_gradient(delta, clip_norm=DP_CLIP_NORM):
    """Clip gradient update to bound sensitivity."""
    flat = np.concatenate([delta["W"], [delta["b"]]])
    norm = np.linalg.norm(flat)
    if norm > clip_norm:
        scale = clip_norm / norm
        return {"W": delta["W"] * scale, "b": delta["b"] * scale}
    return delta

def add_gaussian_noise(delta, epsilon, n_samples, clip_norm=DP_CLIP_NORM):
    """
    Add Gaussian noise calibrated to (ε, δ)-DP.
    sigma = sqrt(2 * ln(1.25/δ)) * clip_norm / (ε * n_samples)
    Using δ = 1e-5 (standard choice).
    """
    delta_val = 1e-5
    sigma = (np.sqrt(2 * np.log(1.25 / delta_val)) * clip_norm) / (epsilon * n_samples)
    noisy_W = delta["W"] + np.random.normal(0, sigma, size=delta["W"].shape)
    noisy_b = delta["b"] + np.random.normal(0, sigma)
    return {"W": noisy_W, "b": noisy_b}

# ─────────────────────────────────────────────
# 4. FEDERATED AVERAGING (FedAvg)
# ─────────────────────────────────────────────
def fedavg(global_model, local_updates, weights=None):
    """Aggregate local model updates using weighted averaging."""
    n = len(local_updates)
    if weights is None:
        weights = [1.0 / n] * n
    agg_W = sum(w * u["W"] for w, u in zip(weights, local_updates))
    agg_b  = sum(w * u["b"] for w, u in zip(weights, local_updates))
    new_model = {"W": global_model["W"] + agg_W,
                 "b": global_model["b"] + agg_b}
    return new_model

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, datasets):
    """Evaluate model on all client datasets combined."""
    all_X = np.vstack([d[0] for d in datasets])
    all_y = np.concatenate([d[1] for d in datasets])
    preds = predict(model, all_X)
    acc  = accuracy_score(all_y, preds)
    f1   = f1_score(all_y, preds, zero_division=0)
    loss = compute_loss(model, all_X, all_y)
    return acc, f1, loss

# ─────────────────────────────────────────────
# 6. TRAINING LOOPS
# ─────────────────────────────────────────────

def run_centralized(datasets):
    """Baseline: train on all data pooled together."""
    all_X = np.vstack([d[0] for d in datasets])
    all_y = np.concatenate([d[1] for d in datasets])
    model = init_model()
    history = []
    for _ in range(N_ROUNDS):
        model = train_local(model, all_X, all_y, epochs=5)
        acc, f1, loss = evaluate(model, datasets)
        history.append((acc, f1, loss))
    return model, history

def run_federated(datasets, epsilon=None):
    """
    Federated Learning with optional Differential Privacy.
    epsilon=None → no DP; epsilon=float → apply DP noise.
    """
    global_model = init_model()
    history = []
    n_clients = len(datasets)

    for rnd in range(N_ROUNDS):
        local_updates = []
        for i in range(n_clients):
            X_i, y_i = datasets[i]
            local_m = train_local(global_model, X_i, y_i)
            delta = model_delta(global_model, local_m)

            if epsilon is not None:
                delta = clip_gradient(delta)
                delta = add_gaussian_noise(delta, epsilon, len(y_i))

            local_updates.append(delta)

        # Aggregate
        n_total = sum(len(d[1]) for d in datasets)
        weights = [len(d[1]) / n_total for d in datasets]
        global_model = fedavg(global_model, local_updates, weights)

        acc, f1, loss = evaluate(global_model, datasets)
        history.append((acc, f1, loss))

    return global_model, history

# ─────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Smart City IoT Security — FL + Differential Privacy")
    print("=" * 60)

    # Generate data
    print("\n[1] Generating IoT sensor data for 5 city zones …")
    datasets = [generate_iot_data(i) for i in range(N_CLIENTS)]
    for i, (X, y) in enumerate(datasets):
        print(f"    Zone {ZONE_NAMES[i]:12s}: {len(y)} samples | "
              f"normal={sum(y==0)} | anomaly={sum(y==1)}")

    # ── Centralized ──
    print("\n[2] Training Centralized model (baseline) …")
    _, hist_central = run_centralized(datasets)

    # ── Federated (no DP) ──
    print("[3] Training Federated Learning model (no DP) …")
    model_fl, hist_fl = run_federated(datasets, epsilon=None)

    # ── FL + DP at various ε ──
    print("[4] Training FL + Differential Privacy at various ε …")
    fl_dp_results = {}
    for eps in EPSILON_VALUES:
        print(f"    ε = {eps} …", end="", flush=True)
        _, hist = run_federated(datasets, epsilon=eps)
        fl_dp_results[eps] = hist
        print(f"  final acc = {hist[-1][0]:.4f}")

    # ─────────────────────────────────────────────
    # 8. RESULTS TABLE
    # ─────────────────────────────────────────────
    print("\n[5] Final Performance Summary (Round 20)")
    print("-" * 55)
    print(f"{'Model':<30} {'Accuracy':>10} {'F1-Score':>10} {'Loss':>10}")
    print("-" * 55)

    c_acc, c_f1, c_loss = hist_central[-1]
    print(f"{'Centralized (pooled)':<30} {c_acc:>10.4f} {c_f1:>10.4f} {c_loss:>10.4f}")

    fl_acc, fl_f1, fl_loss = hist_fl[-1]
    print(f"{'Federated (no DP)':<30} {fl_acc:>10.4f} {fl_f1:>10.4f} {fl_loss:>10.4f}")

    for eps, hist in fl_dp_results.items():
        acc, f1, loss = hist[-1]
        label = f"FL + DP (ε={eps})"
        print(f"{label:<30} {acc:>10.4f} {f1:>10.4f} {loss:>10.4f}")
    print("-" * 55)

    # Save results to CSV
    rows = []
    rows.append({"Model": "Centralized", "Accuracy": c_acc, "F1": c_f1, "Loss": c_loss, "Epsilon": "N/A"})
    rows.append({"Model": "Federated (no DP)", "Accuracy": fl_acc, "F1": fl_f1, "Loss": fl_loss, "Epsilon": "∞"})
    for eps, hist in fl_dp_results.items():
        acc, f1, loss = hist[-1]
        rows.append({"Model": f"FL+DP", "Accuracy": acc, "F1": f1, "Loss": loss, "Epsilon": eps})
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/results_summary.csv", index=False)

    # ─────────────────────────────────────────────
    # 9. PLOTS
    # ─────────────────────────────────────────────
    print("\n[6] Generating plots …")
    plot_results(hist_central, hist_fl, fl_dp_results, datasets, model_fl)

    print(f"\n✓ Done! All results saved to ./{OUTPUT_DIR}/")

# ─────────────────────────────────────────────
# 10. VISUALISATION
# ─────────────────────────────────────────────
def plot_results(hist_central, hist_fl, fl_dp_results, datasets, model_fl):
    rounds = range(1, N_ROUNDS + 1)
    colors = plt.cm.tab10.colors

    # ── Figure 1: Accuracy & Loss Convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Federated Learning for Smart City IoT Anomaly Detection", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(rounds, [h[0] for h in hist_central], "k--", lw=2, label="Centralized")
    ax.plot(rounds, [h[0] for h in hist_fl], color=colors[0], lw=2, label="FL (no DP)")
    for i, (eps, hist) in enumerate(fl_dp_results.items()):
        ax.plot(rounds, [h[0] for h in hist], color=colors[i+1], lw=1.5, alpha=0.8, label=f"FL+DP ε={eps}")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy over Rounds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    ax = axes[1]
    ax.plot(rounds, [h[2] for h in hist_central], "k--", lw=2, label="Centralized")
    ax.plot(rounds, [h[2] for h in hist_fl], color=colors[0], lw=2, label="FL (no DP)")
    for i, (eps, hist) in enumerate(fl_dp_results.items()):
        ax.plot(rounds, [h[2] for h in hist], color=colors[i+1], lw=1.5, alpha=0.8, label=f"FL+DP ε={eps}")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss over Rounds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_convergence.png", dpi=150)
    plt.close()

    # ── Figure 2: Privacy-Utility Trade-off ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Privacy–Utility Trade-off (ε)", fontsize=13, fontweight="bold")

    epsilons = list(fl_dp_results.keys())
    final_acc  = [fl_dp_results[e][-1][0] for e in epsilons]
    final_f1   = [fl_dp_results[e][-1][1] for e in epsilons]

    ax = axes[0]
    ax.plot(epsilons, final_acc, "o-", color="steelblue", lw=2, markersize=8, label="FL+DP")
    ax.axhline(hist_fl[-1][0], color="green", linestyle="--", lw=1.5, label="FL (no DP)")
    ax.axhline(hist_central[-1][0], color="red", linestyle=":", lw=1.5, label="Centralized")
    ax.set_xlabel("Privacy Budget ε (lower = more private)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy vs Privacy Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    ax = axes[1]
    ax.plot(epsilons, final_f1, "s-", color="darkorange", lw=2, markersize=8, label="FL+DP")
    ax.axhline(hist_fl[-1][1], color="green", linestyle="--", lw=1.5, label="FL (no DP)")
    ax.axhline(hist_central[-1][1], color="red", linestyle=":", lw=1.5, label="Centralized")
    ax.set_xlabel("Privacy Budget ε (lower = more private)")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score vs Privacy Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_privacy_utility.png", dpi=150)
    plt.close()

    # ── Figure 3: Data Distribution across Zones ──
    fig, axes = plt.subplots(1, N_CLIENTS, figsize=(16, 4))
    fig.suptitle("IoT Sensor Data Distribution per Smart City Zone", fontsize=13, fontweight="bold")

    for i, (X, y) in enumerate(datasets):
        ax = axes[i]
        ax.scatter(X[y==0, 0], X[y==0, 1], c="royalblue", s=10, alpha=0.5, label="Normal")
        ax.scatter(X[y==1, 0], X[y==1, 1], c="tomato",    s=10, alpha=0.5, label="Anomaly")
        ax.set_title(f"Zone: {ZONE_NAMES[i]}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_data_distribution.png", dpi=150)
    plt.close()

    # ── Figure 4: Confusion Matrix for FL (no DP) model ──
    all_X = np.vstack([d[0] for d in datasets])
    all_y = np.concatenate([d[1] for d in datasets])
    preds = predict(model_fl, all_X)
    cm = confusion_matrix(all_y, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Federated Model (no DP)")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14, color="black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_confusion_matrix.png", dpi=150)
    plt.close()

    # ── Figure 5: Architecture Diagram (text-based via matplotlib) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    fig.patch.set_facecolor("#f7f9fc")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("Federated Learning + Differential Privacy Architecture\nfor Smart City IoT Security",
                 fontsize=13, fontweight="bold", pad=15)

    # Zones
    zone_colors = ["#AED6F1","#A9DFBF","#F9E79F","#F1948A","#D7BDE2"]
    xs = [1, 2.5, 4, 5.5, 7]
    for i, (x, zn, zc) in enumerate(zip(xs, ZONE_NAMES, zone_colors)):
        rect = plt.Rectangle((x-0.6, 3.8), 1.2, 1.2, color=zc, ec="gray", lw=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 4.4, zn, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        ax.text(x, 4.0, "IoT Zone", ha="center", va="center", fontsize=7, color="gray", zorder=4)

        # DP annotation
        ax.annotate("+ DP Noise\n(clipping+\nGaussian)", xy=(x, 3.8), xytext=(x, 3.1),
                    ha="center", fontsize=6, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1))

    # Central server
    rect2 = plt.Rectangle((3.8, 0.8), 2.4, 1.2, color="#85C1E9", ec="steelblue", lw=2, zorder=3)
    ax.add_patch(rect2)
    ax.text(5, 1.4, "Central Server\n(FedAvg Aggregation)", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=4)

    # Arrows from DP layer to server
    for x in xs:
        ax.annotate("", xy=(5, 2.0), xytext=(x, 3.1),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2))

    # Global model arrow back
    ax.annotate("Global Model\n(broadcast)", xy=(4.0, 5.0), xytext=(3.2, 2.0),
                fontsize=7, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5, connectionstyle="arc3,rad=0.3"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_architecture.png", dpi=150)
    plt.close()

    print(f"    Saved 5 figures to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

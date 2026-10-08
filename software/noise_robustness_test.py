"""
noise_robustness_sweep.py

Noise-robustness comparison of the RF classifier
(chf_rf_reproduction.py) vs. the HDC classifier (chf_hdc_implementation.py)
on the CHF-detection pipeline.

Design (see accompanying discussion):
  - Train both classifiers ONCE on clean DS1.
  - Corrupt only DS2 (held-out, disjoint patients) at each
    (noise_type x SNR) condition using PhysioNet's Noise Stress Test
    Database (nstdb: bw/ma/em) + a synthetic Gaussian control.
  - Noise is injected into the RAW ECG signal, per record, at a full
    random offset each repeat -- BEFORE windowing / CA4 extraction --
    so this tests robustness of the whole pipeline (feature extraction
    + classifier), which is the realistic wearable-device threat model.
  - RF and HDC are evaluated on the *exact same* noisy signal realization
    at each (noise_type, snr, repeat) so any accuracy gap is attributable
    to the classifier, not to different noise draws.
  - Each condition is repeated N_REPEATS times with different random
    noise offsets/seeds and reported as mean +/- std.

Outputs:
  - results/noise_sweep_results.csv   (one row per condition x repeat x classifier)
  - results/noise_sweep_<noisetype>.png   (accuracy vs SNR, RF vs HDC)
  - results/noise_sweep_summary_bar.png   (accuracy at one fixed SNR, all noise types)

Requirements:
    pip install wfdb scikit-learn scipy numpy matplotlib

Run:
    python noise_robustness_sweep.py
    python noise_robustness_sweep.py --quick        # small dim/levels/repeats, fast smoke test
    python noise_robustness_sweep.py --data-dir /path/to/local/physionet/folders
"""

import argparse
import os
import time

import numpy as np
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


from chf_hdc_implementation import (
    load_lead_i, extract_ca4, HDCClassifier,
    FS, WIN, N_COEF,
    CHF_DB, NSR_DB,
    TRAIN_CHF, TRAIN_NSR, TEST_CHF, TEST_NSR,
)
from chf_rf_reproduction import N_TREES, CCP_ALPHA, RANDOM_STATE as RF_RANDOM_STATE

# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
NOISE_TYPES = ["bw", "ma", "em", "gaussian"]
SNR_LEVELS_DB = [24, 18, 12, 6, 0]
N_REPEATS = 3
SUMMARY_SNR_DB = 6
NOISE_DB = "nstdb"

RESULTS_DIR = "results"


# ----------------------------------------------------------------------
# Noise loading / injection
# ----------------------------------------------------------------------
def load_noise_record(noise_type: str, data_dir: str = "") -> np.ndarray:
    """Load one nstdb noise channel, resampled to FS Hz. Returns None for
    'gaussian' (handled synthetically at injection time)."""
    if noise_type == "gaussian":
        return None

    local_path = None
    if data_dir:
        for cand in (os.path.join(data_dir, NOISE_DB, noise_type),
                     os.path.join(data_dir, noise_type)):
            if os.path.exists(cand + ".hea"):
                local_path = cand
                break

    if local_path is not None:
        rec = wfdb.rdrecord(local_path)
    else:
        rec = wfdb.rdrecord(noise_type, pn_dir=NOISE_DB)

    sig = rec.p_signal[:, 0].astype(np.float64)
    sig = np.nan_to_num(sig)

    if rec.fs != FS:
        from scipy.signal import resample
        n_samples = int(round(len(sig) * FS / rec.fs))
        sig = resample(sig, n_samples)

    return sig


def inject_noise_full_record(sig: np.ndarray, noise_full, snr_db: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Add noise to `sig` at a target SNR (dB), computed over the whole
    record. `noise_full` is either a 1-D noise array (nstdb channel) or
    None (-> synthetic Gaussian noise is generated instead). A random
    offset into the noise record is drawn each call (tiling with
    wraparound if the noise record is shorter than `sig`), so repeated
    calls with the same noise source produce different noise instances."""
    n = len(sig)

    if noise_full is None:
        noise_seg = rng.normal(0.0, 1.0, size=n)
    else:
        noise_len = len(noise_full)
        if noise_len >= n:
            offset = int(rng.integers(0, noise_len - n + 1))
            noise_seg = noise_full[offset:offset + n]
        else:
            reps = int(np.ceil(n / noise_len))
            tiled = np.tile(noise_full, reps)
            offset = int(rng.integers(0, len(tiled) - n + 1))
            noise_seg = tiled[offset:offset + n]

    sig_power = np.mean(sig ** 2)
    noise_power = np.mean(noise_seg ** 2)
    if noise_power <= 0 or sig_power <= 0:
        return sig.copy()

    scale = np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10.0))))
    return sig + scale * noise_seg


# ----------------------------------------------------------------------
# Signal caching (load each test/train record's raw signal exactly once)
# ----------------------------------------------------------------------
def cache_signals(records, db, data_dir):
    cache = {}
    for r in records:
        print(f"  caching raw signal: {r} ({db}) ...")
        cache[r] = load_lead_i(r, db, data_dir)
    return cache


def features_from_cached(sig_cache, records, label):
    """Window + CA4-extract a dict of {record: raw_signal} into (X, y)."""
    X, y = [], []
    for r in records:
        sig = sig_cache[r]
        n_windows = len(sig) // WIN
        for w in range(n_windows):
            seg = sig[w * WIN:(w + 1) * WIN]
            X.append(extract_ca4(seg, N_COEF))
            y.append(label)
    return X, y


def build_features(sig_cache_chf, sig_cache_nsr, chf_records, nsr_records):
    X, y = [], []
    f, l = features_from_cached(sig_cache_chf, chf_records, 1)
    X.extend(f); y.extend(l)
    f, l = features_from_cached(sig_cache_nsr, nsr_records, 0)
    X.extend(f); y.extend(l)
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# Metrics helper
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return acc, sens, spec


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RF vs HDC noise-robustness sweep")
    ap.add_argument("--data-dir", default="",
                     help="Local folder with offline chfdb/nsrdb/nstdb records "
                          "(falls back to PhysioNet download if not found).")
    ap.add_argument("--npy-fs", type=float, default=None)
    ap.add_argument("--dim", type=int, default=4000, help="HDC hypervector dim")
    ap.add_argument("--levels", type=int, default=100, help="HDC quantization levels")
    ap.add_argument("--epochs", type=int, default=20, help="HDC retraining epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--quick", action="store_true",
                     help="Small dim/levels/epochs/repeats + 1 SNR level, for a fast smoke test.")
    args = ap.parse_args()

    if args.quick:
        args.dim, args.levels, args.epochs = 500, 20, 5
        args.repeats = 1
        snr_levels = [6]
        noise_types = ["bw", "gaussian"]
    else:
        snr_levels = SNR_LEVELS_DB
        noise_types = NOISE_TYPES

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Train both classifiers ONCE on clean DS1
    # --------------------------------------------------------------
    print("=" * 70)
    print("Loading clean DS1 (train) ...")
    print("=" * 70)
    train_chf_cache = cache_signals(TRAIN_CHF, CHF_DB, args.data_dir)
    train_nsr_cache = cache_signals(TRAIN_NSR, NSR_DB, args.data_dir)
    X_train, y_train = build_features(train_chf_cache, train_nsr_cache, TRAIN_CHF, TRAIN_NSR)
    print(f"DS1: {X_train.shape[0]} windows, "
          f"{int((y_train == 1).sum())} CHF / {int((y_train == 0).sum())} normal")

    print("\nTraining RF ...")
    rf_clf = RandomForestClassifier(n_estimators=N_TREES, ccp_alpha=CCP_ALPHA,
                                     random_state=RF_RANDOM_STATE)
    rf_clf.fit(X_train, y_train)

    print("Training HDC ...")
    hdc_clf = HDCClassifier(dim=args.dim, n_levels=args.levels, epochs=args.epochs,
                             seed=args.seed)
    hdc_clf.fit(X_train, y_train)

    print("\nEvaluating clean-data baseline on DS2 ...")
    test_chf_cache = cache_signals(TEST_CHF, CHF_DB, args.data_dir)
    test_nsr_cache = cache_signals(TEST_NSR, NSR_DB, args.data_dir)
    X_test_clean, y_test_clean = build_features(test_chf_cache, test_nsr_cache, TEST_CHF, TEST_NSR)
    print(f"DS2: {X_test_clean.shape[0]} windows, "
          f"{int((y_test_clean == 1).sum())} CHF / {int((y_test_clean == 0).sum())} normal")

    rows = []
    for name, clf in (("RF", rf_clf), ("HDC", hdc_clf)):
        acc, sens, spec = compute_metrics(y_test_clean, clf.predict(X_test_clean))
        print(f"  clean  {name:4s}: acc={acc*100:.2f}%  sens={sens*100:.2f}%  spec={spec*100:.2f}%")
        rows.append(dict(noise_type="clean", snr_db=None, repeat=0, classifier=name,
                          accuracy=acc, sensitivity=sens, specificity=spec,
                          n_windows=len(y_test_clean)))

    # --------------------------------------------------------------
    # 2. Load noise sources once
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Loading noise sources ...")
    print("=" * 70)
    noise_cache = {}
    for nt in noise_types:
        print(f"  {nt} ...")
        noise_cache[nt] = load_noise_record(nt, args.data_dir)

    # --------------------------------------------------------------
    # 3. Sweep: for each (noise_type, snr, repeat), corrupt DS2 raw
    #    signals, re-extract features, evaluate both classifiers on the
    #    SAME noisy realization.
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running noise sweep ...")
    print("=" * 70)

    t0 = time.time()
    n_conditions = len(noise_types) * len(snr_levels) * args.repeats
    cond_i = 0

    for noise_type in noise_types:
        noise_full = noise_cache[noise_type]
        for snr_db in snr_levels:
            for rep in range(args.repeats):
                cond_i += 1
                # Deterministic-but-distinct RNG per (noise_type, snr, repeat)
                seed = hash((noise_type, snr_db, rep, "noise_sweep")) % (2 ** 31)
                rng = np.random.default_rng(seed)

                noisy_chf = {r: inject_noise_full_record(sig, noise_full, snr_db, rng)
                             for r, sig in test_chf_cache.items()}
                noisy_nsr = {r: inject_noise_full_record(sig, noise_full, snr_db, rng)
                             for r, sig in test_nsr_cache.items()}
                X_noisy, y_noisy = build_features(noisy_chf, noisy_nsr, TEST_CHF, TEST_NSR)

                for name, clf in (("RF", rf_clf), ("HDC", hdc_clf)):
                    acc, sens, spec = compute_metrics(y_noisy, clf.predict(X_noisy))
                    rows.append(dict(noise_type=noise_type, snr_db=snr_db, repeat=rep,
                                      classifier=name, accuracy=acc, sensitivity=sens,
                                      specificity=spec, n_windows=len(y_noisy)))

                elapsed = time.time() - t0
                print(f"[{cond_i}/{n_conditions}] {noise_type:8s} SNR={snr_db:>3} dB "
                      f"rep={rep}  ({elapsed:.0f}s elapsed)")

    # --------------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------------
    import csv
    csv_path = os.path.join(RESULTS_DIR, "noise_sweep_results.csv")
    fieldnames = ["noise_type", "snr_db", "repeat", "classifier",
                  "accuracy", "sensitivity", "specificity", "n_windows"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    # --------------------------------------------------------------
    # 5. Plots
    # --------------------------------------------------------------
    make_plots(rows, noise_types, snr_levels)


def make_plots(rows, noise_types, snr_levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_acc = {r["classifier"]: r["accuracy"] for r in rows if r["noise_type"] == "clean"}

    def agg(noise_type, snr_db, classifier):
        vals = [r["accuracy"] for r in rows
                if r["noise_type"] == noise_type and r["snr_db"] == snr_db
                and r["classifier"] == classifier]
        return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    # -- per-noise-type accuracy vs SNR --
    for nt in noise_types:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, color in (("RF", "tab:blue"), ("HDC", "tab:orange")):
            means, stds = [], []
            for snr in snr_levels:
                m, s = agg(nt, snr, name)
                means.append(m * 100)
                stds.append(s * 100)
            ax.errorbar(snr_levels, means, yerr=stds, marker="o", label=name, color=color)
            ax.axhline(clean_acc[name] * 100, linestyle="--", color=color, alpha=0.4,
                       label=f"{name} clean baseline")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"RF vs HDC accuracy under {nt} noise")
        ax.invert_xaxis()  # high SNR (easy) on the left, 0 dB (hard) on the right
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"noise_sweep_{nt}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    # -- summary bar chart at one fixed SNR across noise types --
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = np.arange(len(noise_types))
    for i, (name, color) in enumerate((("RF", "tab:blue"), ("HDC", "tab:orange"))):
        means = []
        for nt in noise_types:
            m, _ = agg(nt, SUMMARY_SNR_DB, name)
            means.append(m * 100)
        ax.bar(x + (i - 0.5) * width, means, width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"RF vs HDC accuracy at {SUMMARY_SNR_DB} dB SNR, by noise type")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "noise_sweep_summary_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
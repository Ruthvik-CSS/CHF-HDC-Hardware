"""
Reproduction of the DWT (Quadratic-Spline, level-4 approximation) + Random Forest
CHF classifier described in:

  Bhardwaj, Janveja, Krishnaswamy, Pidanic, Trivedi,
  "Design of Random Forest-Based Low-Power VLSI Architecture to Detect
   Congestive Heart Failure for Wearable Devices," IEEE TVLSI, 2026.
"""

import argparse
import os

import numpy as np
from scipy.signal import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------------------------
# Constants from the paper
# ----------------------------------------------------------------------
FS = 250            # target sampling rate (Hz) used by the paper
WIN = 250            # 1-second window = 250 samples
N_COEF = 14           # 14 wavelet coefficients per 1-s window (paper, Eq. 2)
N_TREES = 3           # "three decision trees provide an acceptable accuracy"
CCP_ALPHA = 0.004        # "pruning parameter of 0.004"
RANDOM_STATE = 39        # default RF seed; not specified by the paper

CHF_DB = "chfdb"    # PhysioNet BIDMC-CHF database identifier
NSR_DB = "nsrdb"    # PhysioNet MIT-BIH Normal Sinus Rhythm DB identifier

# ----------------------------------------------------------------------
# Local caching
# ----------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SIGNAL_CACHE_DIR = os.path.join(CACHE_DIR, "signals")
FEATURE_CACHE_DIR = os.path.join(CACHE_DIR, "features")
os.makedirs(SIGNAL_CACHE_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# Exact subject-oriented split given in the paper (Section IV-A)
TRAIN_CHF = [f"chf{str(i).zfill(2)}" for i in range(1, 11)]
TRAIN_NSR = ["16265", "16272", "16273", "16420", "16483", "16539",
             "16773", "16786", "16795", "17052", "17453", "18177"]

TEST_CHF = [f"chf{str(i).zfill(2)}" for i in range(11, 16)]
TEST_NSR = ["18184", "19088", "19090", "19093", "19140", "19830"]


# ----------------------------------------------------------------------
# Signal loading (with on-disk caching so PhysioNet is only hit once)
# ----------------------------------------------------------------------
def load_lead_i(record_name: str, pn_dir: str, use_cache: bool = True) -> np.ndarray:
    """Return the Lead-I ECG channel, resampled to FS Hz.

    Checks ./cache/signals/<pn_dir>_<record_name>.npy first; only downloads
    from PhysioNet via wfdb on a cache miss.
    """
    cache_path = os.path.join(SIGNAL_CACHE_DIR, f"{pn_dir}_{record_name}.npy")

    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    import wfdb 

    rec = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    sig_names = [s.upper() for s in rec.sig_name]

    idx = 0
    for i, name in enumerate(sig_names):
        if name in ("ECG1", "I", "MLII", "V1", "V2"):
            idx = i
            break

    sig = rec.p_signal[:, idx].astype(np.float64)
    sig = np.nan_to_num(sig)

    if rec.fs != FS:
        n_samples = int(round(len(sig) * FS / rec.fs))
        sig = resample(sig, n_samples)

    if use_cache:
        np.save(cache_path, sig)

    return sig


# ----------------------------------------------------------------------
# Feature extraction: closed-form level-4 approximation coefficient, Eq. (2)
# ----------------------------------------------------------------------
def extract_ca4(x: np.ndarray, n_coef: int = N_COEF) -> np.ndarray:
    """
    Compute n_coef level-4 approximation wavelet coefficients from a 1-D
    ECG segment x, following the QS-wavelet closed-form recursion (Eq. 2):

        CA4[n] = sum_{k=1}^{7} k * ( x[16n+2k-2] + x[16n+2k-1]
                                     + x[16n+46-2k] + x[16n+47-2k] )
                 + 8 * sum_{j=14}^{31} x[16n+j]
        CA4[n] >>= 4   (i.e. divide by 16)

    Each coefficient consumes 46 input samples starting at offset 16n.
    The input is edge-padded so n_coef coefficients can always be produced.
    """
    needed_len = 16 * (n_coef - 1) + 46
    if len(x) < needed_len:
        x = np.pad(x, (0, needed_len - len(x)), mode="edge")

    coeffs = np.zeros(n_coef)
    for n in range(n_coef):
        base = 16 * n
        s = 0.0
        for k in range(1, 8):
            i1 = base + 2 * k - 2
            i2 = base + 2 * k - 1
            i3 = base + 46 - 2 * k
            i4 = base + 47 - 2 * k
            s += k * (x[i1] + x[i2] + x[i3] + x[i4])
        s += 8 * np.sum(x[base + 14: base + 32])
        coeffs[n] = s / 16.0

    return coeffs


# ----------------------------------------------------------------------
# Optional: 16-bit fixed-point quantization (Section III-D of the paper)
# ----------------------------------------------------------------------
def quantize_16bit(values: np.ndarray, scale: int = 1024) -> np.ndarray:
    """Quantize to 16-bit signed fixed-point with scaling factor 2^10, per Sec. III-D."""
    q = np.round(values * scale).astype(np.int32)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.float64) / scale


# ----------------------------------------------------------------------
# Dataset construction
# ----------------------------------------------------------------------
def make_windows(sig: np.ndarray, label: int, quantize: bool = True):
    feats, labels = [], []
    n_windows = len(sig) // WIN
    for w in range(n_windows):
        seg = sig[w * WIN:(w + 1) * WIN]
        if quantize:
            seg = quantize_16bit(seg)
        ca4 = extract_ca4(seg, N_COEF)
        if quantize:
            ca4 = quantize_16bit(ca4)
        feats.append(ca4)
        labels.append(label)
    return feats, labels


def build_dataset(chf_records, nsr_records, split_name: str,
                   quantize: bool = True, use_cache: bool = True):
    """Build (X, y) for a split, caching the final feature/label arrays under
    ./cache/features/<split_name>.npz so repeat runs skip both the PhysioNet
    download AND the Eq. (2) feature extraction entirely."""
    feat_cache_path = os.path.join(FEATURE_CACHE_DIR, f"{split_name}.npz")

    if use_cache and os.path.exists(feat_cache_path):
        data = np.load(feat_cache_path)
        print(f"  [cache hit] loaded {split_name} features from {feat_cache_path}")
        return data["X"], data["y"]

    X, y = [], []
    for r in chf_records:
        print(f"  loading CHF record {r} ...")
        sig = load_lead_i(r, CHF_DB, use_cache=use_cache)
        f, l = make_windows(sig, 1, quantize)
        X.extend(f)
        y.extend(l)
    for r in nsr_records:
        print(f"  loading NSR record {r} ...")
        sig = load_lead_i(r, NSR_DB, use_cache=use_cache)
        f, l = make_windows(sig, 0, quantize)
        X.extend(f)
        y.extend(l)

    X, y = np.array(X), np.array(y)

    if use_cache:
        np.savez_compressed(feat_cache_path, X=X, y=y)
        print(f"  [cache saved] {split_name} features -> {feat_cache_path}")

    return X, y


def evaluate_seed(X_train, y_train, X_test, y_test, seed: int):
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        ccp_alpha=CCP_ALPHA,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return acc, sens, spec


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[RANDOM_STATE],
                         help="One or more RandomForest random_state values to try "
                              "(default: 42). Data is loaded/extracted once and reused "
                              "across all seeds.")
    parser.add_argument("--no-cache", action="store_true",
                         help="Ignore the local cache; re-download from PhysioNet and "
                              "re-run feature extraction from scratch.")
    args = parser.parse_args()
    use_cache = not args.no_cache

    print("Building training set DS1 (subject-oriented) ...")
    X_train, y_train = build_dataset(TRAIN_CHF, TRAIN_NSR, "train",
                                      use_cache=use_cache)
    print(f"  DS1: {X_train.shape[0]} windows, "
          f"{int((y_train == 1).sum())} CHF / {int((y_train == 0).sum())} normal")

    print("Building test set DS2 (subject-oriented, disjoint patients) ...")
    X_test, y_test = build_dataset(TEST_CHF, TEST_NSR, "test",
                                    use_cache=use_cache)
    print(f"  DS2: {X_test.shape[0]} windows, "
          f"{int((y_test == 1).sum())} CHF / {int((y_test == 0).sum())} normal")

    print(f"\nTraining RandomForestClassifier(n_estimators={N_TREES}, "
          f"ccp_alpha={CCP_ALPHA}) across seed(s): {args.seeds}\n")

    print("=== Results (paper reports 90.5% / 93.01% / 92.26%) ===")
    accs, senss, specs = [], [], []
    for seed in args.seeds:
        acc, sens, spec = evaluate_seed(X_train, y_train, X_test, y_test, seed)
        accs.append(acc); senss.append(sens); specs.append(spec)
        print(f"seed={seed:<5d}  Accuracy: {acc*100:6.2f}%   "
              f"Sensitivity: {sens*100:6.2f}%   Specificity: {spec*100:6.2f}%")

    if len(args.seeds) > 1:
        print("\n--- Summary across seeds ---")
        print(f"Accuracy:    mean={np.mean(accs)*100:.2f}%  std={np.std(accs)*100:.2f}%  "
              f"min={min(accs)*100:.2f}%  max={max(accs)*100:.2f}%")
        print(f"Sensitivity: mean={np.mean(senss)*100:.2f}%  std={np.std(senss)*100:.2f}%  "
              f"min={min(senss)*100:.2f}%  max={max(senss)*100:.2f}%")
        print(f"Specificity: mean={np.mean(specs)*100:.2f}%  std={np.std(specs)*100:.2f}%  "
              f"min={min(specs)*100:.2f}%  max={max(specs)*100:.2f}%")


if __name__ == "__main__":
    main()
"""
Same DWT (Quadratic-Spline, level-4 approximation) feature pipeline as
chf_rf_reproduction.py, but with the classifier swapped from Random Forest
to Hyperdimensional Computing (HDC / VSA).

  Bhardwaj, Janveja, Krishnaswamy, Pidanic, Trivedi,
  "Design of Random Forest-Based Low-Power VLSI Architecture to Detect
   Congestive Heart Failure for Wearable Devices," IEEE TVLSI, 2026.

Pipeline:
  - Data: BIDMC-CHF ("chfdb") + MIT-BIH Normal Sinus Rhythm DB ("nsrdb")
  - Lead: Lead I only
  - Segmentation: 1-second windows (250 samples @ 250 Hz)
  - Feature extraction: closed-form level-4 approximation wavelet
    coefficient CA4[n], Eq. (2) in the paper -> 14 features / window

Classifier: Hyperdimensional Computing (binary/bipolar spatter-code HDC)
  with batch processing to avoid OOM.
"""

import argparse
import os
import sys
import numpy as np
import wfdb
from scipy.signal import resample
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ----------------------------------------------------------------------
# Constants from the paper (feature-extraction side)
# ----------------------------------------------------------------------
FS = 250              # target sampling rate (Hz)
WIN = 250             # 1-second window = 250 samples
N_COEF = 14           # 14 wavelet coefficients per 1-s window

CHF_DB = "chfdb"      # PhysioNet BIDMC-CHF database identifier
NSR_DB = "nsrdb"      # PhysioNet MIT-BIH Normal Sinus Rhythm DB

# Subject-oriented split given in the paper (Section IV-A)
TRAIN_CHF = [f"chf{str(i).zfill(2)}" for i in range(1, 11)]          # chf01..chf10
TRAIN_NSR = ["16265", "16272", "16273", "16420", "16483", "16539",
             "16773", "16786", "16795", "17052", "17453", "18177"]

TEST_CHF = [f"chf{str(i).zfill(2)}" for i in range(11, 16)]          # chf11..chf15
TEST_NSR = ["18184", "19088", "19090", "19093", "19140", "19830"]


# ----------------------------------------------------------------------
# Signal loading
# ----------------------------------------------------------------------
def _find_local_npy(record_name: str, db: str, data_dir: str):
    if not data_dir:
        return None
    primary = os.path.join(data_dir, f"{db}_{record_name}.npy")
    if os.path.exists(primary):
        return primary
    candidates = [
        os.path.join(data_dir, db, record_name + ".npy"),
        os.path.join(data_dir, record_name + ".npy"),
        os.path.join(data_dir, "..", db, record_name + ".npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _find_local_record(record_name: str, db: str, data_dir: str):
    if not data_dir:
        return None
    primary = os.path.join(data_dir, f"{db}_{record_name}")
    if os.path.exists(primary + ".hea"):
        return primary
    candidates = [
        os.path.join(data_dir, db, record_name),
        os.path.join(data_dir, record_name),
    ]
    for c in candidates:
        if os.path.exists(c + ".hea"):
            return c
    return None

def load_lead_i(record_name: str, db: str, data_dir: str = "",
                 npy_fs: float = None) -> np.ndarray:
    """Return the Lead-I ECG channel, resampled to FS Hz."""
    npy_path = _find_local_npy(record_name, db, data_dir)
    if npy_path is not None:
        print(f"    Loading from: {npy_path}")
        sig = np.load(npy_path).astype(np.float64)
        if sig.ndim > 1:
            sig = sig[:, 0] if sig.shape[0] > sig.shape[1] else sig[0, :]
        sig = np.nan_to_num(sig)
        if npy_fs is not None and npy_fs != FS:
            n_samples = int(round(len(sig) * FS / npy_fs))
            sig = resample(sig, n_samples)
        return sig

    local_path = _find_local_record(record_name, db, data_dir)
    if local_path is not None:
        print(f"    Loading WFDB from: {local_path}")
        rec = wfdb.rdrecord(local_path)
    else:
        print(f"    Downloading from PhysioNet: {record_name}")
        rec = wfdb.rdrecord(record_name, pn_dir=db)

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

    return sig


# ----------------------------------------------------------------------
# Feature extraction: closed-form level-4 approximation coefficient, Eq. (2)
# ----------------------------------------------------------------------
def extract_ca4(x: np.ndarray, n_coef: int = N_COEF) -> np.ndarray:
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

def make_windows(sig: np.ndarray, label: int):
    feats, labels = [], []
    n_windows = len(sig) // WIN
    for w in range(n_windows):
        seg = sig[w * WIN:(w + 1) * WIN]
        ca4 = extract_ca4(seg, N_COEF)
        feats.append(ca4)
        labels.append(label)
    return feats, labels

def build_dataset(chf_records, nsr_records, data_dir: str = "", npy_fs: float = None):
    X, y = [], []
    for r in chf_records:
        print(f"  loading CHF record {r} ...")
        sig = load_lead_i(r, CHF_DB, data_dir, npy_fs)
        f, l = make_windows(sig, 1)
        X.extend(f)
        y.extend(l)
    for r in nsr_records:
        print(f"  loading NSR record {r} ...")
        sig = load_lead_i(r, NSR_DB, data_dir, npy_fs)
        f, l = make_windows(sig, 0)
        X.extend(f)
        y.extend(l)
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# Hyperdimensional Computing (HDC) classifier with batch processing
# ----------------------------------------------------------------------
class HDCClassifier:
    def __init__(self, dim: int = 1000, n_levels: int = 50,
                 n_flip: int = None, epochs: int = 10, lr: float = 1.0,
                 batch_size: int = 50000, seed: int = 847362915, verbose: bool = True):
        self.dim = dim
        self.n_levels = n_levels
        self.n_flip = n_flip if n_flip is not None else max(1, dim // (2 * (n_levels - 1)))
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.feat_min_ = None
        self.feat_max_ = None
        self.pos_hv_ = None
        self.level_hv_ = None
        self.class_hv_ = None

    def _build_position_memory(self, n_features: int):
        self.pos_hv_ = self.rng.choice([-1, 1], size=(n_features, self.dim)).astype(np.int8)

    def _build_level_memory(self):
        levels = np.zeros((self.n_levels, self.dim), dtype=np.int8)
        base = self.rng.choice([-1, 1], size=self.dim).astype(np.int8)
        levels[0] = base
        flip_order = self.rng.permutation(self.dim)
        flips_done = 0
        current = base.copy()
        for lvl in range(1, self.n_levels):
            n_to_flip = min(self.n_flip, self.dim - flips_done)
            if n_to_flip > 0:
                idx = flip_order[flips_done:flips_done + n_to_flip]
                current = current.copy()
                current[idx] *= -1
                flips_done += n_to_flip
            levels[lvl] = current
        self.level_hv_ = levels

    def _quantize(self, X):
        levels = (X - self.feat_min_) / (self.feat_max_ - self.feat_min_ + 1e-12)
        levels = np.clip(levels, 0.0, 1.0)
        levels = np.round(levels * (self.n_levels - 1)).astype(np.int32)
        return levels

    def _encode_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Encode a batch of feature vectors into bipolar hypervectors (int8)."""
        levels = self._quantize(X_batch)
        n_samples, n_features = X_batch.shape
        out = np.zeros((n_samples, self.dim), dtype=np.int32)
        for j in range(n_features):
            bound = self.level_hv_[levels[:, j]] * self.pos_hv_[j]
            out += bound.astype(np.int32)
        return np.sign(out + 1e-9).astype(np.int8)

    def _accumulate_class_vectors(self, X, y):
        """Accumulate class hypervectors from encoded batches without storing all encodings."""
        class_hv = np.zeros((2, self.dim), dtype=np.float64)
        n_samples = X.shape[0]
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            X_batch = X[start:end]
            y_batch = y[start:end]
            H_batch = self._encode_batch(X_batch)
            for c in (0, 1):
                mask = (y_batch == c)
                if np.any(mask):
                    class_hv[c] += H_batch[mask].sum(axis=0).astype(np.float64)
        return class_hv

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        if self.verbose:
            print(f"Fitting HDC classifier on {n_samples} samples (batch size {self.batch_size})")

        # 1. Compute min/max, build memories
        self.feat_min_ = X.min(axis=0)
        self.feat_max_ = X.max(axis=0)
        self._build_position_memory(X.shape[1])
        self._build_level_memory()

        # 2. Initial class hypervectors (accumulated in batches)
        if self.verbose:
            print("  Encoding training set and accumulating class hypervectors...")
        self.class_hv_ = self._accumulate_class_vectors(X, y)

        # 3. AdaptHD-style perceptron retraining
        if self.epochs > 0 and self.verbose:
            print(f"  Retraining for {self.epochs} epochs...")

        order = np.arange(n_samples)
        for epoch in range(self.epochs):
            self.rng.shuffle(order)
            n_updates = 0

            # --- FIX: define pbar always, use tqdm only if verbose and available ---
            if tqdm is not None and self.verbose:
                pbar = tqdm(range(0, n_samples, self.batch_size),
                            desc=f"  Epoch {epoch+1}/{self.epochs}")
            else:
                pbar = range(0, n_samples, self.batch_size)
                if self.verbose:
                    print(f"  Epoch {epoch+1}/{self.epochs}")

            for start in pbar:
                end = min(start + self.batch_size, n_samples)
                idx_batch = order[start:end]
                X_batch = X[idx_batch]
                y_batch = y[idx_batch]

                H_batch = self._encode_batch(X_batch)
                # cosine similarity
                norm_c = np.linalg.norm(self.class_hv_, axis=1) + 1e-12
                dots = H_batch.astype(np.float64) @ self.class_hv_.T
                norm_h = np.linalg.norm(H_batch.astype(np.float64), axis=1, keepdims=True) + 1e-12
                sims = dots / (norm_h * norm_c[None, :])
                preds = np.argmax(sims, axis=1)

                mis = (preds != y_batch)
                if np.any(mis):
                    # update correct classes
                    for c in (0, 1):
                        mask = mis & (y_batch == c)
                        if np.any(mask):
                            self.class_hv_[c] += self.lr * H_batch[mask].sum(axis=0).astype(np.float64)
                    # subtract from wrong predicted classes
                    for d in (0, 1):
                        mask = mis & (preds == d)
                        if np.any(mask):
                            self.class_hv_[d] -= self.lr * H_batch[mask].sum(axis=0).astype(np.float64)
                    n_updates += np.sum(mis)

            if self.verbose:
                print(f"    updates: {n_updates}")
            if n_updates == 0:
                if self.verbose:
                    print("    Converged early.")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        preds = np.empty(n_samples, dtype=np.int32)
        norm_c = np.linalg.norm(self.class_hv_, axis=1) + 1e-12

        if self.verbose:
            print(f"Predicting on {n_samples} samples...")
            if tqdm is not None:
                iterator = tqdm(range(0, n_samples, self.batch_size), desc="  Predicting")
            else:
                iterator = range(0, n_samples, self.batch_size)
                print("  Predicting (no tqdm)")
        else:
            iterator = range(0, n_samples, self.batch_size)

        for start in iterator:
            end = min(start + self.batch_size, n_samples)
            X_batch = X[start:end]
            H_batch = self._encode_batch(X_batch)
            dots = H_batch.astype(np.float64) @ self.class_hv_.T
            norm_h = np.linalg.norm(H_batch.astype(np.float64), axis=1, keepdims=True) + 1e-12
            sims = dots / (norm_h * norm_c[None, :])
            preds[start:end] = np.argmax(sims, axis=1)

        return preds


# ----------------------------------------------------------------------
# Main (for standalone testing)
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CHF classification: CA4 features + HDC classifier")
    ap.add_argument("--data-dir",
                    default=r"./data/signals",
                    help="Local folder with .npy files. Looks for 'chfdb_*.npy' and 'nsrdb_*.npy'")
    ap.add_argument("--npy-fs", type=float, default=None,
                    help="Sampling rate of .npy files if not 250 Hz")
    ap.add_argument("--dim", type=int, default=10000, help="HDC hypervector dimensionality")
    ap.add_argument("--levels", type=int, default=100, help="Number of quantization levels (CIM)")
    ap.add_argument("--epochs", type=int, default=20, help="Retraining epochs (AdaptHD-style)")
    ap.add_argument("--batch-size", type=int, default=50000,
                    help="Number of samples per batch (reduce if OOM)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    print("=" * 60)
    print("CHF Classification using CA4 features + HDC Classifier")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print()

    if not os.path.exists(args.data_dir):
        print(f"Warning: Data directory '{args.data_dir}' does not exist! Will try to download from PhysioNet.")

    print("Building training set DS1 (subject-oriented) ...")
    X_train, y_train = build_dataset(TRAIN_CHF, TRAIN_NSR, args.data_dir, args.npy_fs)
    print(f"  DS1: {X_train.shape[0]} windows, "
          f"{int((y_train == 1).sum())} CHF / {int((y_train == 0).sum())} normal")

    print("\nBuilding test set DS2 (subject-oriented, disjoint patients) ...")
    X_test, y_test = build_dataset(TEST_CHF, TEST_NSR, args.data_dir, args.npy_fs)
    print(f"  DS2: {X_test.shape[0]} windows, "
          f"{int((y_test == 1).sum())} CHF / {int((y_test == 0).sum())} normal")

    print(f"\nTraining HDC classifier (dim={args.dim}, levels={args.levels}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}) ...")
    clf = HDCClassifier(dim=args.dim, n_levels=args.levels,
                        epochs=args.epochs, batch_size=args.batch_size,
                        seed=args.seed, verbose=not args.quiet)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Sensitivity: {sens * 100:.2f}%")
    print(f"Specificity: {spec * 100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:6d}   FP: {fp:6d}")
    print(f"  FN: {fn:6d}   TP: {tp:6d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
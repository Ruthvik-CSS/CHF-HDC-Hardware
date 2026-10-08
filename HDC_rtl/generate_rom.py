# generate_rom.py
#
# Generates all RTL ROM files (position, level, class, threshold) from a
# trained HDCClassifier.

import numpy as np
import os
from chf_hdc_implementation import build_dataset, HDCClassifier, TRAIN_CHF, TRAIN_NSR

DIM = 1000
LEVELS = 20
BATCH_SIZE = 50000
SEED = 0
N_FEAT = 14

# Fixed-point scale factor for feature/threshold values (THRESH_W=16, signed).
#
# IMPORTANT: this data's CA4 coefficients run up to roughly +/-127 in
# magnitude (see the feat_min_/feat_max_ printout below), NOT the small
# +/-1-ish range Q6.10 (scale=1024, "same as RF paper") was sized for.
# At scale=1024, every one of the 14 features overflows the 16-bit signed
# range (max representable magnitude is only 32767/1024 ~= 32), and the
# two's-complement wraparound actually INVERTS the ordering (large values
# wrap to small/negative numbers and vice versa), this silently
# corrupts every threshold comparison and was traced back as the actual
# cause of the RTL "always predicts the same class" symptom.
#
# scale=128 gives a representable range of +/-256, ~2x headroom over the
# observed max (~127), while keeping reasonable fractional precision.
# generate_test_vectors.py MUST use this exact same SCALE for the raw ECG
# samples it writes -- wavelet_ca4.v's math is scale-agnostic (it just
# processes whatever integers it's given), so as long as both sides agree
# on the scale, no RTL changes are needed.
SCALE = 128

def normalize_class_hypervectors(class_hv):
    
    norms = np.linalg.norm(class_hv, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit = class_hv / norms
    target_norm = norms.mean()
    return unit * target_norm

def quantize_class_hypervectors(class_hv, n_bits):
    max_val = 2 ** (n_bits - 1) - 1
    min_val = -(max_val + 1)

    max_abs = np.max(np.abs(class_hv))
    if max_abs > 0:
        scaled = class_hv * (max_val / max_abs)
    else:
        scaled = class_hv

    quantized = np.clip(np.round(scaled), min_val, max_val).astype(np.int32)
    return quantized

def compute_linear_thresholds(feat_min, feat_max, n_levels):
    """Midpoint thresholds for the exact linear quantization used by
    HDCClassifier._quantize(). Returns a (N_FEAT, n_levels-1) array."""
    thresholds = []
    for feat in range(len(feat_min)):
        rng = feat_max[feat] - feat_min[feat]
        if rng <= 0:
            rng = 1e-6 
        feat_thresh = [
            feat_min[feat] + rng * (k + 0.5) / (n_levels - 1)
            for k in range(n_levels - 1)
        ]
        thresholds.append(feat_thresh)
    return thresholds

def generate_roms():
    print("=" * 60)
    print("Generating ROM files for RTL implementation")
    print("=" * 60)

    print("\nLoading training data...")
    X_train, y_train = build_dataset(TRAIN_CHF, TRAIN_NSR, DATA_DIR)

    print(f"\nTraining HDC model (DIM={DIM}, LEVELS={LEVELS}, epochs=5)...")
    clf = HDCClassifier(
        dim=DIM,
        n_levels=LEVELS,
        epochs=5,
        batch_size=BATCH_SIZE,
        seed=SEED,
        verbose=True
    )
    clf.fit(X_train, y_train)


    raw_norms = np.linalg.norm(clf.class_hv_, axis=1)
    print(f"\nClass hypervector norms (raw, pre-fix): "
          f"class0(NSR)={raw_norms[0]:.2f}  class1(CHF)={raw_norms[1]:.2f}  "
          f"ratio={max(raw_norms)/min(raw_norms):.2f}x")
    if max(raw_norms) / min(raw_norms) > 1.3:
        print("  -> significant imbalance detected; this is almost certainly why")
        print("     the RTL (raw dot-product comparison) was biased toward one class.")

    class_hv_balanced = normalize_class_hypervectors(clf.class_hv_)

    class_hv_8bit = quantize_class_hypervectors(class_hv_balanced, 8)

    rom_dir = "roms_500"
    os.makedirs(rom_dir, exist_ok=True)

    print("\nGenerating ROM files...")
    print(f"  Output directory: {rom_dir}")
    print()

    # ========================================================================
    # 1. Position ROM (N_FEAT x DIM bits)
    # ========================================================================
    print("  Generating position ROM...")
    with open(os.path.join(rom_dir, "pos_rom_m0.mem"), "w") as f:
        for feat in range(N_FEAT):
            for bit in range(DIM):
                val = 1 if clf.pos_hv_[feat][bit] > 0 else 0
                f.write(f"{val}\n")
    print(f"    OK pos_rom_m0.mem ({N_FEAT * DIM:,} bits)")

    # ========================================================================
    # 2. Level ROM (LEVELS x DIM bits)
    # ========================================================================
    print("  Generating level ROM...")
    with open(os.path.join(rom_dir, "level_rom_m0.mem"), "w") as f:
        for level in range(LEVELS):
            for bit in range(DIM):
                val = 1 if clf.level_hv_[level][bit] > 0 else 0
                f.write(f"{val}\n")
    print(f"    OK level_rom_m0.mem ({LEVELS * DIM:,} bits)")

    # ========================================================================
    # 3. Class ROM (2 classes x DIM x 8 bits)
    # ========================================================================
    print("  Generating class ROM...")
    with open(os.path.join(rom_dir, "class_rom_m0.mem"), "w") as f:
        for cls in range(2):
            for bit in range(DIM):
                val = class_hv_8bit[cls][bit] & 0xFF
                f.write(f"{val:02x}\n")
    print(f"    OK class_rom_m0.mem ({2 * DIM * 8:,} bits)")

    # ========================================================================
    # 4. Threshold ROMs (N_FEAT x (LEVELS-1) thresholds)
    #    Linear min/max midpoints - matches clf._quantize() exactly.
    # ========================================================================
    print("  Generating threshold ROMs (linear min-max, matching _quantize)...")

    thresholds = compute_linear_thresholds(clf.feat_min_, clf.feat_max_, LEVELS)
    max_abs_seen = 0.0
    for feat in range(N_FEAT):
        print(f"    Feature {feat:2d}: min={clf.feat_min_[feat]:.4f} "
              f"max={clf.feat_max_[feat]:.4f} -> {len(thresholds[feat])} thresholds")
        max_abs_seen = max(max_abs_seen, abs(clf.feat_min_[feat]), abs(clf.feat_max_[feat]))


    max_representable = 32767 / SCALE
    print(f"\n  SCALE={SCALE}: max representable magnitude = {max_representable:.2f}, "
          f"largest feature value observed = {max_abs_seen:.2f}")
    if max_abs_seen * SCALE > 32767:
        print(f"  *** WARNING: SCALE={SCALE} OVERFLOWS 16-bit signed range for this data! ***")
        print(f"      Reduce SCALE (try {2**int(np.floor(np.log2(32767/max_abs_seen)))}) "
              f"and update generate_test_vectors.py to match.")
    elif max_abs_seen * SCALE > 32767 * 0.8:
        print(f"  NOTE: within 20% of overflow -- fine for this training data, but the")
        print(f"        held-out test set could run slightly hotter. Worth double-checking.")

    # Write threshold files
    for feat in range(N_FEAT):
        with open(os.path.join(rom_dir, f"thresh_feat{feat:02d}.mem"), "w") as f:
            for t in thresholds[feat]:
                val = int(round(t * SCALE)) & 0xFFFF
                f.write(f"{val:04x}\n")
    print(f"    OK threshold ROMs ({N_FEAT} files, {N_FEAT * (LEVELS-1)} thresholds)")

    print("\n" + "=" * 60)
    print("ROM GENERATION COMPLETE!")
    print("=" * 60)

    total_bits = (
        N_FEAT * DIM +
        LEVELS * DIM +
        2 * DIM * 8 +
        N_FEAT * (LEVELS-1) * 16
    )
    total_bytes = total_bits / 8
    total_kb = total_bytes / 1024

    print(f"\nROM Summary:")
    print(f"  Position ROM:  {N_FEAT * DIM:>8,} bits")
    print(f"  Level ROM:     {LEVELS * DIM:>8,} bits")
    print(f"  Class ROM:     {2 * DIM * 8:>8,} bits")
    print(f"  Threshold ROMs:{N_FEAT * (LEVELS-1) * 16:>8,} bits")
    print(f"  {'-' * 30}")
    print(f"  Total:         {total_bits:>8,} bits")
    print(f"  Total:         {total_bytes:>8,.0f} bytes")
    print(f"  Total:         {total_kb:>8.1f} KB")

    print(f"\nFiles written to: {rom_dir}/")
    print("\nReady for RTL simulation!")

if __name__ == "__main__":
    DATA_DIR = r"./data/signals"
    generate_roms()
# generate_test_vectors.py
#
# Writes the RTL testbench inputs for DS2 (the held-out test set):
#   ecg_samples.bin  -- RAW ECG samples, fixed-point, 16-bit signed,
#                        BIG-ENDIAN (250 samples per window, back to back)
#   labels.txt       -- one ground-truth label (0/1) per window

import numpy as np
import struct
from chf_hdc_implementation import build_raw_dataset, TEST_CHF, TEST_NSR

SCALE = 128  # MUST match SCALE in generate_rom.py -- see note above

def generate_test_data():
    print("Loading DS2 test set (raw ECG windows)...")
    X_test, y_test = build_raw_dataset(TEST_CHF, TEST_NSR, DATA_DIR)

    n_windows, win_len = X_test.shape
    print(f"DS2 test set: {n_windows} windows x {win_len} samples/window")

    raw_max_abs = float(np.max(np.abs(X_test)))
    max_representable = 32767 / SCALE
    print(f"Raw sample magnitude: max |value| = {raw_max_abs:.2f}  "
          f"(SCALE={SCALE} representable range = +/-{max_representable:.2f})")
    if raw_max_abs * SCALE > 32767:
        print(f"*** WARNING: SCALE={SCALE} OVERFLOWS 16-bit signed range for this data! ***")
        print(f"    Reduce SCALE (try {2**int(np.floor(np.log2(32767/raw_max_abs)))}) "
              f"and update generate_rom.py to match.")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_windows)
    X_test = X_test[perm]
    y_test = y_test[perm]

    # Write ECG samples (binary, 16-bit signed, BIG-ENDIAN, Q6.10)
    n_clipped = 0
    with open("ecg_samples.bin", "wb") as f:
        for window in X_test:
            for sample in window:
                scaled = int(round(sample * SCALE))
                if scaled > 32767 or scaled < -32768:
                    n_clipped += 1
                scaled = max(-32768, min(32767, scaled))
                f.write(struct.pack('>h', scaled))

    with open("labels.txt", "w") as f:
        for label in y_test:
            f.write(f"{int(label)}\n")

    print(f"✓ Wrote {n_windows * win_len} ECG samples ({n_windows} windows)")
    print(f"✓ Wrote {len(y_test)} ground truth labels")
    print(f"   CHF: {int(sum(y_test))}, NSR: {int(len(y_test) - sum(y_test))}")
    if n_clipped:
        print(f"⚠️  {n_clipped} samples clipped to int16 range -- check SCALE/data")

if __name__ == "__main__":
    DATA_DIR = r"./data/signals"
    generate_test_data()
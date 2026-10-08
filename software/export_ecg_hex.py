"""
export_ecg_hex.py

"""

import argparse
import os

import numpy as np
import wfdb


FS = 250
WIN_LEN = 250                              
N_COEF = 14
DRIVE_CYCLES = 16 * (N_COEF - 1) + 48       

SCALE = 1024                                
INT16_MIN, INT16_MAX = -32768, 32767


TEST_CHF = [f"chf{str(i).zfill(2)}" for i in range(11, 16)]
TEST_NSR = ["18184", "19088", "19090", "19093", "19140", "19830"]



def load_lead_i_local(record_name: str, database_dir: str) -> np.ndarray:
    """Read a WFDB record's Lead-I channel from local .dat/.hea files.
    No network access -- record_name.dat and record_name.hea must already
    exist in database_dir."""
    record_path = os.path.join(database_dir, record_name)
    if not os.path.exists(record_path + ".hea"):
        raise FileNotFoundError(
            f"Could not find {record_path}.hea -- make sure {record_name}.dat "
            f"and {record_name}.hea are both present in '{database_dir}'."
        )

    rec = wfdb.rdrecord(record_path)
    sig_names = [s.upper() for s in rec.sig_name]
    idx = 0
    for i, name in enumerate(sig_names):
        if name in ("ECG1", "I", "MLII", "V1", "V2"):
            idx = i
            break

    sig = rec.p_signal[:, idx].astype(np.float64)
    sig = np.nan_to_num(sig)

    if rec.fs != FS:
        from scipy.signal import resample
        n_samples = int(round(len(sig) * FS / rec.fs))
        sig = resample(sig, n_samples)

    return sig


def quantize_q6_10(mv: np.ndarray) -> np.ndarray:
    """Quantize floating-point mV to 16-bit signed Q6.10 (scale 2^10),
    identical to train_and_export.py's quantize_q6_10 -- the ADC-equivalent
    step assumed to have already happened before samples reach wavelet_ca4."""
    q = np.round(mv * SCALE).astype(np.int64)
    q = np.clip(q, INT16_MIN, INT16_MAX)
    return q.astype(np.int64)


def to_hex16(v: int) -> str:
    """4-hex-digit two's-complement representation of a 16-bit signed int
    -- exactly what $readmemh expects for a [15:0] memory word."""
    return format(int(v) & 0xFFFF, "04x")


def windows_from_signal(sig_mv: np.ndarray, max_windows: int):
    """Split a continuous ECG signal into non-overlapping 250-sample
    windows, quantize, and edge-pad each window out to DRIVE_CYCLES (256)
    samples by repeating the window's own last real sample -- identical to
    tb_chf_top.v's golden `padded[]` array and to chf_rf_reproduction.py /
    train_and_export.py's edge-padding policy."""
    sig_int = quantize_q6_10(sig_mv)
    n_windows = len(sig_int) // WIN_LEN
    if max_windows and max_windows > 0:
        n_windows = min(n_windows, max_windows)

    pad_len = DRIVE_CYCLES - WIN_LEN
    for w in range(n_windows):
        real_win = sig_int[w * WIN_LEN:(w + 1) * WIN_LEN]
        pad = np.full(pad_len, real_win[-1], dtype=real_win.dtype)
        yield np.concatenate([real_win, pad])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default="./database",
                         help="Local directory containing DS2 records' .dat/.hea files "
                              "(default: ./database)")
    parser.add_argument("--outdir", default=".",
                         help="Directory to write ecg_samples.hex / ecg_labels.txt into "
                              "(default: current directory)")
    parser.add_argument("--records", nargs="+", default=None,
                         help="Override which record names to export. Default: the paper's "
                              "exact DS2 test split (chf11..chf15 + the 6 NSR test records). "
                              "Records named 'chf*' are auto-labeled CHF=1, everything else "
                              "is labeled normal=0 -- pass --labels to override per-record.")
    parser.add_argument("--labels", nargs="+", type=int, default=None,
                         help="Explicit 0/1 label for each record in --records, same order "
                              "(only used together with --records).")
    parser.add_argument("--max-windows-per-record", type=int, default=300,
                         help="Cap on 1-second windows exported per record (default 300, "
                              "i.e. 5 minutes). Pass 0 for no limit (full record -- can be "
                              "tens of thousands of windows for BIDMC-CHF records).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.records is not None:
        if args.labels is not None:
            if len(args.labels) != len(args.records):
                raise ValueError("--labels must have the same length as --records")
            record_labels = list(zip(args.records, args.labels))
        else:
            record_labels = [(r, 1 if r.lower().startswith("chf") else 0) for r in args.records]
    else:
        record_labels = [(r, 1) for r in TEST_CHF] + [(r, 0) for r in TEST_NSR]

    hex_path = os.path.join(args.outdir, "ecg_samples.hex")
    label_path = os.path.join(args.outdir, "ecg_labels.txt")

    total_windows = 0
    with open(hex_path, "w") as hf, open(label_path, "w") as lf:
        hf.write(f"// ECG test-vector samples, {DRIVE_CYCLES} 16-bit signed Q6.10 hex "
                  f"values per window ({WIN_LEN} real + {DRIVE_CYCLES - WIN_LEN} edge-padded),"
                  f" generated by export_ecg_hex.py\n")
        lf.write(f"# one label per window (0=normal,1=CHF), same order as ecg_samples.hex\n")

        for record_name, label in record_labels:
            print(f"  loading {record_name} (label={label}) from '{args.database}' ...")
            sig = load_lead_i_local(record_name, args.database)
            n_win_available = len(quantize_q6_10(sig)) // WIN_LEN
            cap = args.max_windows_per_record if args.max_windows_per_record else n_win_available
            n_exported = min(n_win_available, cap) if cap else n_win_available
            print(f"    {n_win_available} windows available, exporting {n_exported}")

            hf.write(f"// ---- record {record_name} (label={label}), "
                      f"{n_exported} windows ----\n")
            for padded in windows_from_signal(sig, args.max_windows_per_record):
                for v in padded:
                    hf.write(to_hex16(v) + "\n")
                lf.write(f"{label}\n")
                total_windows += 1

    total_lines = total_windows * DRIVE_CYCLES
    print(f"\nWrote {hex_path}  ({total_windows} windows x {DRIVE_CYCLES} samples "
          f"= {total_lines} hex lines)")
    print(f"Wrote {label_path} ({total_windows} labels)")

    if total_windows > 5000:
        print(f"\nNOTE: {total_windows} windows is a lot for RTL simulation "
              f"(~{total_windows * DRIVE_CYCLES * 10 / 1e6:.0f}M+ simulated clock edges). "
              f"Consider a smaller --max-windows-per-record for iteration, and compile "
              f"tb_chf_top_accuracy.v with -PMAX_WINDOWS={((total_windows // 1000) + 1) * 1000} "
              f"or higher.")

    print("\nNext step:")
    print("  iverilog -g2012 -o sim rtl/wavelet_ca4.v rtl/rf_classifier.v rtl/chf_top.v \\")
    print("           tb/tb_chf_top_accuracy.v")
    print("  vvp sim +hexfile=ecg_samples.hex +labelfile=ecg_labels.txt")


if __name__ == "__main__":
    main()

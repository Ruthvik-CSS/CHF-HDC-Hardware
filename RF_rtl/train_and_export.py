"""
train_and_export.py

"""

import os
import numpy as np
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------------------------
# Constants -- identical to chf_rf_reproduction.py / the RTL
# ----------------------------------------------------------------------
FS = 250
WIN = 250
N_COEF = 14
N_TREES = 3
CCP_ALPHA = 0.004
RANDOM_STATE = 13

DW = 16
SCALE = 1024
INT16_MIN, INT16_MAX = -32768, 32767

CHF_DB = "chfdb"
NSR_DB = "nsrdb"

CACHE_DIR = "./data/signals"


NSRDB_FILES = {
    "16265": "nsrdb_16265",
    "16272": "nsrdb_16272",
    "16273": "nsrdb_16273",
    "16420": "nsrdb_16420",
    "16483": "nsrdb_16483",
    "16539": "nsrdb_16539",
    "16773": "nsrdb_16773",
    "16786": "nsrdb_16786",
    "16795": "nsrdb_16795",
    "17052": "nsrdb_17052",
    "17453": "nsrdb_17453",
    "18177": "nsrdb_18177",
    "18184": "nsrdb_18184",
    "19088": "nsrdb_19088",
    "19090": "nsrdb_19090",
    "19093": "nsrdb_19093",
    "19140": "nsrdb_19140",
    "19830": "nsrdb_19830",
}

TRAIN_CHF = [f"chf{str(i).zfill(2)}" for i in range(1, 11)]
TRAIN_NSR = ["16265", "16272", "16273", "16420", "16483", "16539",
             "16773", "16786", "16795", "17052", "17453", "18177"]
TEST_CHF = [f"chf{str(i).zfill(2)}" for i in range(11, 16)]
TEST_NSR = ["18184", "19088", "19090", "19093", "19140", "19830"]


# ----------------------------------------------------------------------
# Signal loading (from local .npy cache)
# ----------------------------------------------------------------------
def get_filename(record_name: str, pn_dir: str) -> str:
    
    if pn_dir == CHF_DB:
        return f"chfdb_{record_name}.npy"
    elif pn_dir == NSR_DB:
        if record_name in NSRDB_FILES:
            return f"{NSRDB_FILES[record_name]}.npy"
        else:
            return f"nsrdb_{record_name}.npy"
    else:
        return f"{record_name}.npy"


def load_lead_i(record_name: str, pn_dir: str) -> np.ndarray:
    """Load ECG signal from local .npy cache directory.
    
    Looks for files in CACHE_DIR/ with appropriate naming:
    - CHF records: {record_name}.npy (e.g., chf01.npy)
    - NSR records: nsrdb_{record_name}.npy (e.g., nsrdb_16265.npy)
    """
    filename = get_filename(record_name, pn_dir)
    cache_path = os.path.join(CACHE_DIR, filename)
    
    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(f"Cache directory {CACHE_DIR} not found. "
                               f"Please ensure the signals are downloaded there.")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Record {record_name} not found in {CACHE_DIR}. "
                               f"Expected file: {cache_path}")
    
    try:
        sig = np.load(cache_path)
        
        sig = sig.astype(np.float64)
        
        sig = np.nan_to_num(sig)
        
        return sig
        
    except Exception as e:
        raise RuntimeError(f"Failed to load record {record_name} from {cache_path}: {e}")


def quantize_q6_10(mv: np.ndarray) -> np.ndarray:
    q = np.round(mv * SCALE).astype(np.int64)
    q = np.clip(q, INT16_MIN, INT16_MAX)
    return q.astype(np.int64)


def extract_ca4_fixed(x_int: np.ndarray, n_coef: int = N_COEF) -> np.ndarray:
    
    needed_len = 16 * (n_coef - 1) + 46
    if len(x_int) < needed_len:
        pad = np.full(needed_len - len(x_int), x_int[-1], dtype=x_int.dtype)
        x_int = np.concatenate([x_int, pad])

    out = np.zeros(n_coef, dtype=np.int64)
    for n in range(n_coef):
        base = 16 * n
        acc = 0
        for s in range(46):
            if s < 14:
                w = (s // 2) + 1
            elif s < 32:
                w = 8
            else:
                w = 7 - ((s - 32) // 2)
            acc += w * int(x_int[base + s])
        val = acc >> 4
        val = max(INT16_MIN, min(INT16_MAX, val))
        out[n] = val
    return out


def make_windows_fixed(sig_mv: np.ndarray, label: int):
    sig_int = quantize_q6_10(sig_mv)
    feats, labels = [], []
    n_windows = len(sig_int) // WIN
    for w in range(n_windows):
        seg = sig_int[w * WIN:(w + 1) * WIN]
        ca4 = extract_ca4_fixed(seg, N_COEF)
        feats.append(ca4)
        labels.append(label)
    return feats, labels


def build_dataset_fixed(chf_records, nsr_records):
    X, y = [], []
    for r in chf_records:
        print(f"  loading CHF record {r} ...")
        sig = load_lead_i(r, CHF_DB)
        f, l = make_windows_fixed(sig, 1)
        X.extend(f)
        y.extend(l)
    for r in nsr_records:
        print(f"  loading NSR record {r} ...")
        sig = load_lead_i(r, NSR_DB)
        f, l = make_windows_fixed(sig, 0)
        X.extend(f)
        y.extend(l)
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# Verilog codegen for one trained sklearn tree
# ----------------------------------------------------------------------
def quantize_threshold(thr_float):
    """sklearn CA4 features are already integers (Q6.10 domain values
    stored as floats internally by sklearn); round defensively and clip
    to the 16-bit range."""
    t = int(round(thr_float))
    return max(INT16_MIN, min(INT16_MAX, t))


def verilog_signed_literal(v, width=16):
    if v < 0:
        return f"-{width}'sd{-v}"
    return f"{width}'sd{v}"


def emit_tree_expr(tree, node_id, wire_ctr, lines):
    
    feature = tree.feature[node_id]
    if feature == -2:  # leaf
        counts = tree.value[node_id][0]
        cls = int(np.argmax(counts))
        return "1'b1" if cls == 1 else "1'b0"

    thr = quantize_threshold(tree.threshold[node_id])
    wname = f"n{wire_ctr[0]}"
    cname = f"c{wire_ctr[0]}"
    wire_ctr[0] += 1
    lines.append(
        f"            wire {wname}; custom_comparator #(.DW(DW), "
        f".THRESHOLD({verilog_signed_literal(thr)})) {cname} "
        f"(.x(feat{feature}), .y({wname})); "
        f"// node {node_id}: feat{feature} < {thr} ?"
    )

    left_expr = emit_tree_expr(tree, tree.children_left[node_id], wire_ctr, lines)
    right_expr = emit_tree_expr(tree, tree.children_right[node_id], wire_ctr, lines)
    return f"({wname} ? {left_expr} : {right_expr})"


def emit_tree_module_body(tree_id, tree):
    lines = []
    wire_ctr = [0]
    expr = emit_tree_expr(tree, 0, wire_ctr, lines)
    body = "\n".join(lines)
    return f"""        end else if (TREE_ID == {tree_id}) begin : TREE{tree_id}
{body}
            assign vote = {expr};
"""


def export_trained_trees_verilog(clf, path):
    assert clf.n_estimators == N_TREES
    header = f"""// =============================================================================
// trained_trees.v  --  AUTO-GENERATED by train_and_export.py
//
// Real, trained decision-tree logic for the paper's RF classifier
// (n_estimators={N_TREES}, ccp_alpha={CCP_ALPHA}, subject-oriented DS1/DS2
// split), replacing the illustrative placeholder trees in rf_classifier.v.
//
// HOW TO USE: paste the generate-if body below (from the first
// "if (TREE_ID == 0)" through the last tree's closing, EXCLUSIVE of the
// module's own begin/end and the leading placeholder "if (TREE_ID==0)")
// in place of the TREE0/TREE1/TREE2 blocks inside the `decision_tree`
// module's `generate` block in rtl/rf_classifier.v. The module's port
// list, the feat0..feat13 wire declarations, and custom_comparator are
// unchanged -- only the tree bodies are replaced.
// =============================================================================

    generate
        if (TREE_ID == 0) begin : TREE0
"""
    parts = [header]
    for i, est in enumerate(clf.estimators_):
        body = emit_tree_module_body(i, est.tree_)
        if i == 0:
            body = body.replace(
                f"        end else if (TREE_ID == {i}) begin : TREE{i}\n", "", 1
            )
        parts.append(body)
    parts.append("        end\n    endgenerate\n")
    with open(path, "w") as f:
        f.write("".join(parts))
    print(f"Wrote {path}")


def export_ds2_test_vectors(X_test, y_test, path):
    with open(path, "w") as f:
        f.write(f"# {len(X_test)} DS2 test windows: 14 signed CA4 features then true label (0=normal,1=CHF)\n")
        for feats, label in zip(X_test, y_test):
            f.write(" ".join(str(int(v)) for v in feats) + f" {int(label)}\n")
    print(f"Wrote {path} ({len(X_test)} windows)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Building training set DS1 (fixed-point features) ...")
    print(f"Using cached signals from: {os.path.abspath(CACHE_DIR)}")
    print("File naming conventions:")
    print("  - CHF records: chfXX.npy (e.g., chf01.npy)")
    print("  - NSR records: nsrdb_XXXXX.npy (e.g., nsrdb_16265.npy)")
    X_train, y_train = build_dataset_fixed(TRAIN_CHF, TRAIN_NSR)
    print(f"  DS1: {X_train.shape[0]} windows, "
          f"{int((y_train == 1).sum())} CHF / {int((y_train == 0).sum())} normal")

    print("Building test set DS2 (fixed-point features) ...")
    X_test, y_test = build_dataset_fixed(TEST_CHF, TEST_NSR)
    print(f"  DS2: {X_test.shape[0]} windows, "
          f"{int((y_test == 1).sum())} CHF / {int((y_test == 0).sum())} normal")

    print(f"\nTraining RandomForestClassifier(n_estimators={N_TREES}, "
          f"ccp_alpha={CCP_ALPHA}) on fixed-point features ...")
    clf = RandomForestClassifier(
        n_estimators=N_TREES, ccp_alpha=CCP_ALPHA, random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n=== Python-side results (fixed-point features; paper reports 90.5%/93.01%/92.26%) ===")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Sensitivity: {sens * 100:.2f}%")
    print(f"Specificity: {spec * 100:.2f}%")
    print("(Expect these to differ slightly from tb_accuracy.v's numbers: sklearn's")
    print(" '<=' split convention vs. the hardware's strict '<' comparator differ")
    print(" exactly at threshold ties -- see the NOTE in emit_tree_expr().)")

    export_trained_trees_verilog(clf, "trained_trees.v")
    export_ds2_test_vectors(X_test, y_test, "ds2_test_vectors.txt")

    print("\nDone. Next steps:")
    print("  1. Paste trained_trees.v's generate-if body into rtl/rf_classifier.v")
    print("     (replacing the placeholder TREE0/TREE1/TREE2 blocks).")
    print("  2. Copy ds2_test_vectors.txt next to tb_accuracy.v (or update its path).")
    print("  3. Run: iverilog -g2012 -o sim rtl/rf_classifier.v tb/tb_accuracy.v && vvp sim")


if __name__ == "__main__":
    main()
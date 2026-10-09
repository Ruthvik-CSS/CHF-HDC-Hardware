# HDC-Based Low-Power VLSI Architecture for CHF Detection

RTL and software implementation accompanying the paper *"A HDC-Based
Low-Power VLSI Architecture for Congestive Heart Failure Detection in
Wearable Devices"* (Dhokariya, Chunduri, Rao - IIIT Bangalore, submitted to
ISCAS 2027).

This project replaces the Random Forest classifier of a prior closed form
wavelet feature CHF detector (Bhardwaj et al., IEEE TVLSI 2026) with a
multiplier free Hyperdimensional Computing (HDC) classifier, implemented
end-to-end in synthesizable RTL and verified against a Python
software model. An RF re-implementation is also provided for a matched,
apples-to-apples power/accuracy comparison (both synthesized on sky130
130 nm).

## Repository layout

```
.
├── requirements.txt          Python dependencies (numpy, scipy, scikit-learn, wfdb)
├── HDC_rtl/                 HDC classifier: RTL + ROM generation + testbench
│   ├── hdc_params.vh            Shared parameters (DIM, LEVELS, N_FEAT, widths)
│   ├── wavelet_ca4.v             CA4 wavelet feature extractor (3-slot folded accumulator)
│   ├── level_finder.v            Quantizer: maps each feature to one of L=20 levels
│   ├── basis_and_class_rom.v     Position/level/class/threshold ROM wrapper
│   ├── hdc_core.v                HDC encode (bind+bundle) + dot-product classify core
│   ├── ensemble_top.v            Top-level datapath + controller FSM
│   ├── tb_chf.v                  RTL testbench (feeds ecg_samples.bin, checks vs. labels.txt)
│   ├── generate_rom.py           Trains HDCClassifier, exports the 4 ROMs to roms_500/
│   ├── generate_test_vectors.py  Exports DS2 test windows -> ecg_samples.bin / labels.txt
│   └── roms_500/                 Generated ROM contents (pos/level/class/threshold .mem files)
│
├── RF_rtl/                  Matched Random-Forest reimplementation (comparison baseline)
│   ├── wavelet_ca4.v             Same CA4 front end as HDC_rtl
│   ├── rf_classifier.v           Comparator-tree RF classifier (3 trees)
│   ├── chf_top.v                 Top-level datapath + controller for the RF design
│   ├── tb_chf_full.v             RTL testbench for the RF design
│   └── train_and_export.py       Trains sklearn RandomForest, emits trained_trees.v + test vectors
│
├── hdc_sky130/               Post-synthesis results for the HDC design (sky130, OpenROAD)
│   ├── config.json               Synthesis flow configuration
│   └── metrics.json              Area / power / timing / DRC-LVS results (Table II, HDC column)
│
├── rf_sky130/                 Post-synthesis results for the RF design (same flow/node/clock)
│   ├── config.json
│   └── metrics.json               Area / power / timing / DRC-LVS results (Table II, RF column)
│
└── software/                 Python reference models, dataset prep, and evaluation
    ├── chf_hdc_implementation.py  Bit-accurate HDC software model (HDCClassifier) + dataset build
    ├── chf_rf_reproduction.py     Reproduction of the original RF paper's pipeline/results
    ├── export_ecg_hex.py          Exports $readmemh-style hex ECG test vectors for RTL sim
    └── noise_robustness_test.py   HDC vs. RF accuracy under BW/EM/MA/Gaussian noise (Fig. 2)
```

## Requirements

Python dependencies (see `requirements.txt`):

```
numpy
scipy
scikit-learn
wfdb
```

```bash
pip install -r requirements.txt
```

`struct`, `os`, `argparse` are Python standard library (no install needed).

For RTL simulation you also need **Icarus Verilog** (`iverilog`/`vvp`) on
your PATH. Synthesis results in `hdc_sky130/` and `rf_sky130/` were
produced with the open-source **sky130 PDK** via the **OpenROAD** flow
(only the resulting `config.json` / `metrics.json` are included here, not
the flow scripts themselves).

## Pipeline overview

```
raw ECG (250 Hz, 1 s windows)
   -> wavelet_ca4.v       14 CA4 wavelet coefficients (closed-form, multiplier-free)
   -> level_finder.v      quantize each coefficient to 1-of-20 levels (threshold ROM compares)
   -> hdc_core.v           bind(position, level) per feature, bundle -> 500-d hypervector H,
                            dot-product against class ROM (XNOR + add, no multiplier)
   -> decision             argmax(sim0, sim1) -> CHF / NSR
```

Controlled by `ensemble_top.v` (HDC) / `chf_top.v` (RF), each a small FSM
that sequences feature extraction → quantization loop → classification →
result-valid.

## Key parameters (`hdc_params.vh`)

| Parameter | Value | Notes |
|---|---|---|
| `DIM` (D)        | 500   | hypervector dimension |
| `LEVELS` (L)      | 20    | quantization levels per feature |
| `N_FEAT`          | 14    | CA4 wavelet coefficients per window |
| Feature/threshold scale | 128 (Q-format, 16-bit signed) | **not** 1024 (see note below) |
| Class hypervector width | 8-bit signed | |
| Total ROM          | 29,256 bits (~3.57 KB) | position + level + class + threshold |

> **SCALE mismatch pitfall:** `generate_rom.py` and `generate_test_vectors.py`
> must use the *same* `SCALE`. The HDC pipeline's CA4 output range is
> ~±127, so `SCALE=128` is used (not the RF side's `SCALE=1024`, which
> would overflow the 16-bit signed range and silently invert comparisons
> via two's-complement wraparound, this was the original cause of the
> "RTL always predicts the same class" bug). The RF pipeline
> (`train_and_export.py`, `export_ecg_hex.py`) uses `SCALE=1024` (Q6.10)
> and is self-consistent on its own side; don't mix ROMs/test vectors
> across the two designs.

## Reproducing the results

### 1. HDC design
```bash
cd software    # chf_hdc_implementation.py must be importable
python ../HDC_rtl/generate_rom.py            # trains + writes HDC_rtl/roms_500/*.mem
python ../HDC_rtl/generate_test_vectors.py   # writes ecg_samples.bin + labels.txt (DS2)

cd ../HDC_rtl
iverilog -g2012 -o sim_hdc hdc_params.vh wavelet_ca4.v level_finder.v \
    basis_and_class_rom.v hdc_core.v ensemble_top.v tb_chf.v
vvp sim_hdc
```

### 2. RF baseline
```bash
cd RF_rtl
python train_and_export.py
# paste the generated trained_trees.v generate-if body into rf_classifier.v's
# decision_tree module (replacing the placeholder TREE0/TREE1/TREE2 blocks)
iverilog -g2012 -o sim_rf wavelet_ca4.v rf_classifier.v chf_top.v tb_chf_full.v
vvp sim_rf
```

### 3. Noise robustness (software-level, Fig. 2)
```bash
cd software
python noise_robustness_test.py
```

### 4. Synthesis results
`hdc_sky130/` and `rf_sky130/` contain the OpenROAD/sky130 `config.json`
flow settings and `metrics.json` outputs (area, power, cell count,
DRC/LVS) used to populate Table II of the paper.

## Datasets

Lead-I ECG from PhysioNet **BIDMC-CHF** (`chfdb`) and **MIT-BIH Normal
Sinus Rhythm** (`nsrdb`), resampled to 250 Hz. Subject-oriented split:
- **DS1 (train):** chf01-chf10 (10 CHF subjects) + 12 NSR subjects
- **DS2 (test):** chf11-chf15 (5 CHF subjects) + 6 NSR subjects

Noise-robustness evaluation additionally uses the MIT-BIH Noise Stress
Test Database (BW, MA, EM) plus a synthetic Gaussian condition.

Signal caching/loading expects local `.npy` (or `.dat`/`.hea` for
`export_ecg_hex.py`) files under a `data/signals/`-style cache directory —
see `load_lead_i()` in `chf_hdc_implementation.py` / `train_and_export.py`
for the expected naming convention (`chfXX.npy`, `nsrdb_XXXXX.npy`).

## Headline results

- **92.57%** accuracy (90.30% sensitivity / 94.17% specificity) on the
  RTL-simulated DS2 test subset (100,000 windows).
- **8.1×** more accuracy-per-µW than the matched RF reimplementation
  (12.54%/µW vs. 1.54%/µW), driven by a power reduction (7.38 µW vs.
  59.78 µW) rather than an accuracy difference (the two designs are
  within 0.6 points of each other).
- HDC accuracy is never lower than the RF baseline across all four
  tested noise conditions (BW, MA, EM, Gaussian), with the largest gap
  under electrode motion at 6 dB SNR (86.1% vs. 84.2%).

## Status / limitations

Single-lead (Lead I) ECG only; small subject pool (10 CHF/12 NSR train,
5 CHF/6 NSR test); power/area are post-synthesis sky130 estimates, not
silicon measurements. See the paper's Discussion/Limitations section for
the full list.

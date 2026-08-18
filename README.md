# Bearing Fault Diagnosis: Cross-Domain Transfer & SHAP Interpretability

## 1. Overview

This project investigates why machine learning models trained on a single benchmark
bearing-vibration dataset (CWRU) fail to generalize to vibration data collected on a
different physical test rig (MFS), and tests whether a single, interpretable
preprocessing change — unit-variance normalization — can close that gap. A Random
Forest classifier is trained exclusively on CWRU Drive End data and evaluated in a
strict **zero-shot** setting on the MFS dataset, under two otherwise identical
pipelines that differ only in the presence of a segment-level unit-variance
normalization step. SHAP (SHapley Additive exPlanations) is used to explain *why*
accuracy changes, not just *that* it changes, by comparing feature attributions
before and after normalization.

**Core finding:** enforcing domain-consistent normalization raised mean three-class
(Healthy / Inner Race / Outer Race) transfer accuracy from **57.7% to 92.3%**.
SHAP analysis shows this is driven by scale-sensitive amplitude features (RMS,
Standard Deviation, Peak Amplitude) losing nearly all predictive contribution, while
scale-invariant, geometry-derived ratio features (BPFO-ratio, BPFI-ratio) become the
dominant contributors to Inner Race and Outer Race classification.

Full methodology, results, and discussion are in [`Final_report.pdf`](./Final_report.pdf).

---

## 2. System & Workspace Environment

This pipeline was developed and run inside a dedicated Python virtual environment
to keep dependencies isolated and reproducible.

### Workspace Specifications

- **Python Runtime:** 3.12.x (`./venv/bin/python3`)
- **Package Installer:** `./venv/bin/python3 -m pip`

### Project Dependencies

- **numpy** — signal array handling and feature matrix construction
- **pandas** — tabular feature CSVs and results summarization
- **scipy** — `sosfiltfilt` zero-phase high-pass filtering and Hilbert transform (envelope extraction)
- **scikit-learn** — `RandomForestClassifier`, `StratifiedGroupKFold` / group k-fold splitting, `MinMaxScaler`
- **shap** — TreeExplainer-based SHAP value computation and beeswarm plotting
- **matplotlib** — confusion matrices, SHAP beeswarm plots
- **joblib** — serializing trained model artifacts to disk

---

## 3. Pipeline Architecture & Execution Flow

The pipeline runs as a sequential set of stages from raw signal acquisition through
to paired SHAP interpretation, executed through a single entry point.

### Stage 1: Data Acquisition

- **Source domain (training only):** CWRU Drive End dataset — SKF 6205 bearing,
  Healthy + EDM-seeded Inner/Outer Race faults at three fault diameters
  (0.007″ / 0.014″ / 0.021″), ~1720–1797 RPM, 0–3 HP load, sampled at 48 kHz.
- **Target domain (zero-shot evaluation only):** MFS (Machinery Fault Simulator)
  dataset — matching SKF 6205 configuration, seven-class scheme (Normal, IR2/IR5/IR8,
  OR2/OR5/OR8 by arc-degree severity), sampled at 51.2 kHz across three fixed speeds
  (600 / 900 / 1200 RPM).

No MFS data is used during training or hyperparameter selection at any point.

### Stage 2: Signal Preprocessing (Two Parallel Pipelines)

Both pipelines share zero-phase filtering (`sosfiltfilt`) applied to the full-length
signal before segmentation, split into two branches: a 1000 Hz high-pass branch
feeding time/frequency-domain features, and a Hilbert-transform envelope branch
feeding fault-frequency ratio features. Signals are cut into 10,000-sample windows
with 50% overlap.

- **Baseline pipeline:** features extracted directly from the filtered, segmented window.
- **Normalized pipeline:** identical to baseline, plus one change — each window is
  zero-centered and rescaled to unit standard deviation, computed separately per
  domain, immediately before feature extraction.

This is the **only** variable that differs between the two pipelines — filtering,
segmentation, feature-extraction code, model architecture, and hyperparameters are
held identical, so any accuracy change is attributable to normalization alone.

### Stage 3: Feature Extraction

Eight hand-crafted features per segment:

| Category | Features |
|---|---|
| Time-domain (4) | RMS, Standard Deviation, Peak Amplitude, Kurtosis |
| Frequency-domain (2) | Dominant Frequency, Peak FFT Amplitude |
| Ratio (2) | BPFO-ratio, BPFI-ratio |

BPFO-ratio and BPFI-ratio are computed from the envelope spectrum as the proportion
of summed amplitude across the first three harmonics of each fault frequency,
recalculated per RPM condition from SKF 6205 geometry (see `Final_report.pdf`,
Annex A, for the full geometric parameter table). The 8-feature set is then
min-max scaled, fit independently per domain, identically in both pipelines.

### Stage 4: Model Training (CWRU only)

A single Random Forest configuration is used throughout:
`n_estimators=300, max_depth=10, max_features=0.8, random_state=9`, trained with
group k-fold cross-validation (leave-one-load-condition-out) on CWRU. Two models
are trained — one on baseline features, one on normalized features — with identical
hyperparameters and validation scheme, so model configuration is ruled out as an
explanation for any performance difference.

### Stage 5: Zero-Shot Cross-Domain Evaluation

Each trained model is applied, without any fine-tuning or adaptation, directly to
the correspondingly processed MFS features. Results are reported at two levels of
granularity: a 3-class collapse (Healthy / Inner Race / Outer Race) for the
headline accuracy comparison, and the full 7-class scheme (fault type × severity)
for detailed error analysis.

### Stage 6: SHAP Interpretability Analysis

SHAP values are computed via `TreeExplainer` for true-positive predictions only,
for both the baseline and normalized models, on both source and target domains.
Beeswarm plots are generated per fault category (Healthy, Inner Race, Outer Race)
so feature-attribution patterns can be directly compared before and after
normalization.

---

## 4. Results Summary

### Cross-Domain Accuracy (3-Class, MFS Target)

| Class | Before Normalization | After Normalization |
|---|---|---|
| Healthy | 1.00 | 1.00 |
| Inner Race | 0.34 | 1.00 |
| Outer Race | 0.39 | 0.77 |
| **Average** | **57.7%** | **92.3%** |

### SHAP Feature Importance Shift

| Feature | Before Normalization | After Normalization |
|---|---|---|
| RMS | ~0.14 | ~0.0 |
| Standard Deviation | ~0.12 | ~0.0 |
| Peak Amplitude | ~0.35 | ~0.02 |
| Kurtosis | ~0.05 | ~0.15 |
| BPFO-ratio | ~0.03 | ~0.20 |
| BPFI-ratio | ~0.03 | ~0.25 |

**Interpretation:** normalization does not make the ratio features more informative
in isolation — it removes the scale-driven shortcut (RMS / Std / Peak Amplitude)
the model was previously exploiting, which forces the classifier onto features that
carried genuine, physically-grounded fault information throughout. Fault *location*
(Inner vs. Outer Race) transfers well after normalization; fault *severity* does
not, since severity information is carried largely by the amplitude-scale features
that normalization suppresses.

---

## 5. Known Limitations

- Severity labels are assumed equivalent between CWRU's EDM-seeded fault diameters
  and MFS's arc-degree defects; this is not independently validated.
- Only Random Forest was evaluated; it is not established whether the normalization
  effect is specific to tree-based models.
- CWRU (1730–1797 RPM) and MFS (600–1200 RPM) operate in largely disjoint speed
  ranges, which limits the transferability of Hz-based features (Dominant Frequency)
  independent of normalization.
- Ball/Cage faults were excluded from cross-domain evaluation.
- The BPFO-ratio/BPFI-ratio harmonic count (first 3 harmonics) was not subjected to
  sensitivity analysis.

Full discussion in `Final_report.pdf`, Section 1.6 and Chapter 5.

---

## 6. Execution Commands

```bash
# Activate the isolated virtual environment
source venv/bin/activate

# Run the full pipeline: preprocessing -> training -> zero-shot eval -> SHAP
python3 main.py
```

---

## 7. Repository Structure

```
.
├── data/               # Dataset references / mapping configs
├── src/                # Pipeline stage scripts (preprocessing, training, eval, SHAP)
├── output/             # Generated confusion matrices, beeswarm plots, result tables
├── model/              # Serialized trained Random Forest artifacts (.joblib)
├── main.py             # Entry point — runs the full pipeline end to end
├── requirements.txt
└── README.md
```

---

## 8. Citation / Acknowledgements

Developed as a minor project for the Bachelor's degree in Mechanical Engineering,
Department of Mechanical and Aerospace Engineering, Pulchowk Campus, Institute of
Engineering, Tribhuvan University (2026), under the supervision of Asst. Prof.
Manoj Adhikari.

Vibration datasets courtesy of the Case Western Reserve University Bearing Data
Center and the Sensing and Measurement Laboratory, School of Mechatronics
Engineering, Harbin Institute of Technology.

**Authors:** Ashim Ghimire, Gaurab Man Joshi, Pranjal Khadka, Sambid Koirala

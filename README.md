# MUTAFormer Small Panel Design Tool

Research code for cohort-specific mutation modeling, permutation attribution,
candidate mutation selection, blocker design, and RF-based blocker screening.
This is a curated source release, not a distribution of patient data or a claim
that a fresh training run reproduces every historical manuscript result.

## Workflow

1. Upload a labeled mutation cohort and a mutation annotation table.
2. Retrain the original mutation-only Transformer on the submitted cohort.
3. Calculate Captum FeaturePermutation attribution on the original test split.
4. Select mutations whose exported attribution magnitude exceeds the threshold.
5. Retrieve hg38 flanking sequences and design blockers using NUPACK.
6. Predict **P(dCt > 8)** using a separately provisioned RF classifier and rank candidates.

The web workflow is mutation-only; it does **not** train the Gene+Protein model.
The original protein and Small-panel scripts are included as research references.
Those scripts require private workbooks, panel mapping, and experiment-specific
configuration; they are not a ready-to-run external validation suite.

## Source map

| Files under `gene_with_order/` | Purpose |
| --- | --- |
| `models.py`, `models_for_explain.py` | Transformer and explanation wrapper |
| `train.py`, `dataset_qPCR.py`, `load_as_dict.py` | Original training, splits, tokenization and VAF handling |
| `explain.py` | Original Captum attribution calculation |
| `retrain_with_plexfilter.py`, `dataset_pannel.py` | Original Small-panel training/filtering |
| `train_with_protein.py`, `dataset_pannel_protein.py` | Original Gene+Protein training |
| `dataset_pannel_blood.py`, `dataset_pannel_tissue.py` | Original external cohort loaders |
| `web/legacy_engine.py` | Isolated subprocess adapter; exports provenance and predictions |
| `web/pipeline.py` | Input checks, annotation, sequence, blocker and RF stages |
| `web/server.py` | Flask upload/job/status/download API |
| `web/index.html`, `web/app.js`, CSS files | English web interface |
| `web/data/demo_*.csv` | Synthetic examples, not patient data |

Keep the inner directory name `gene_with_order`: legacy imports depend on it.
The outer repository can have any name.

## Linux quick start

Python 3.10 is recommended for compatibility with the previously used NUPACK wheel.
From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r gene_with_order/web/requirements.txt
python gene_with_order/web/server.py
```

Open http://127.0.0.1:8000/ (do not open the HTML through `file://`).
Select **Upload my own dataset**, or use **Run demo with example data** for the
small synthetic demonstration. The public source edition defaults to uploads.
The private RFData option requires administrator-provided resources listed below.

For the supplied background launcher:

```bash
PANEL_PYTHON="$PWD/.venv/bin/python" PANEL_HOST=127.0.0.1 bash gene_with_order/web/start_server.sh
bash gene_with_order/web/stop_server.sh
```

Requirements define compatibility ranges, not a fully frozen environment.
Save `python -m pip freeze` alongside each experiment. Different PyTorch versions,
seeds and data order can change the selected panel.

## Required inputs

**Cohort database:** Excel or CSV, one sample/mutation per row.

```csv
Sample_ID,Class,mut_pos,VAF
EXAMPLE_001,HCC,chr17:7674216,0.125
EXAMPLE_002,nonHCC,chr5:1295113,0.032
EXAMPLE_003,nonHCC,,
```

**Mutation annotation:** Excel or CSV, required even when cohort columns overlap.

```csv
mut_pos,Ref,Allele
chr17:7674216,C,A
chr5:1295113,G,A
```

Coordinates are hg38, 1-based. VAF is a fraction in [0,1]; the historical `-`
sentinel is also retained. Use one blank mutation/VAF row for mutation-free
patients. Exactly two classes and at least 10 independent patients per class are
required. Vocabulary size cannot exceed the number of distinct mutations.
The upload limit is 50 MiB for the combined request. Templates are available in
the interface. Example rows above illustrate format, not a sufficient training set.

## Attribution and selection semantics

`explain.py` uses **FeaturePermutation**, not attention weights or SHAP.
The adapter preserves shuffled explanation batches of 4, target labels, RNG call
order and the original duplicate-index accumulation. It saves signed attribution,
then uses `abs(raw[1:])` for the exported mutation attribution.

Consequently the default `attribution > 0` selects **nonzero magnitude**, not
positive signed evidence for cancer. Frequency counts all cohort mutation
occurrences, not necessarily unique patients. A frequency table is merged with
attribution and sorted; no guarantee of the historical 74 mutations is made.

Downstream blocker screening accepts groups with maximum branch probability at
least 0.5 by default, keeps up to 50 by summed attribution, then sorts by summed
frequency and a tie score averaging mean RF probability with normalized summed
attribution. It returns top K (default 10). `P(dCt>8)` is a classification
probability, **not a predicted continuous dCt**.

## Optional/private resources (not committed)

| Resource | Expected location / requirement |
| --- | --- |
| RF classifier + feature schema | `gene_with_order/web/data/predict_dCt/predict_dCt/Data/processed/rf_model.joblib` and `feature_columns.json` |
| NUPACK | Install an authorized, platform-compatible distribution into the same Python environment; previously used version 4.0.2 |
| Private RFData example | `gene_with_order/RFdata_hg38_VAF_Revised.xlsx` and `gene_with_order/blocker_design(1).zip` containing its original annotation resources |
| Historical artifact replay | Same private resources plus original archived attribution/sequence/blocker tables |
| Original Small-panel studies | Private cohort workbooks, `plex_mapping.xlsx`, original checkpoints/settings as referenced in the scripts |

Only load trusted joblib/PyTorch model files: these formats may execute code.
Model weights, licensed binaries, patient workbooks and historical archives are
intentionally excluded. Provisioning them is separate from publishing source.
Without NUPACK, a sequence-completed partial result is expected. Without the RF
model/schema, final RF prediction cannot run. The synthetic demo therefore is
not an unconditional end-to-end design test on a bare installation.
The current health endpoint's RF flag is not an artifact-integrity check.

## Outputs and reproducibility

Each new job writes to `gene_with_order/web/runs/<job_id>/` (git-ignored).
Artifacts may include model weights, split assignments, training history,
patient probabilities/logits, raw attribution, candidate sequences, blockers,
RF branch predictions and panel recommendations. `run_manifest.json` records
settings, source hashes, versions and model hash.

Defaults: 10 layers, 16 heads, embedding 64, dropout 0.1, seed 11, 20 epochs,
learning rate 0.0001, vocabulary 250, sequence length 15, train/test batch 64.
The synthetic demo uses smaller settings. These are not the four-layer
Small-panel experiments, and the web learning-rate default differs from saved
historical 0.001 configurations.

**Important retained limitations:**

- The original Transformer layout permits cross-patient interaction within a
  batch. Predictions can depend on batch composition and ordering.
- Vocabulary uses the full cohort; attribution uses the test split for panel
  discovery. This test set is not an untouched final panel validation cohort.
- Positive training cases are duplicated; cross-entropy consumes probabilities;
  the original live `state_dict` snapshot behavior is retained.
- The original unknown-token attribution alias is retained.
- Legacy micro-AUC flattens two-class one-hot labels and both output columns.
  Patient AUROC uses HCC labels and the HCC output column. They are not interchangeable.
- A historical chr17:7674194 annotation correction (GT to deletion) is retained.
- NUPACK exceptions currently warn and fall back to zero energy. This occurred
  in Linux diagnostics; inspect logs and do not interpret affected RF predictions
  as validated thermodynamic results.

These behaviors are documented for compatibility, not endorsed as best practice.
No clinical diagnostic claim is made. Independent, leakage-controlled validation
is needed for new panels. Historical notes in `web/README.md` describe earlier
experiments, **not tests rerun for this source release**.

## Public deployment and privacy

Bind the local service to loopback and place a reviewed HTTPS reverse proxy or
Cloudflare Tunnel in front of it. Tunnel credentials and tokens must never be
committed. Do not expose the project directory as a static file server.
The Flask development server is not a hardened production deployment.
Add authentication, rate/concurrency limits, retention/deletion policies and
upload safeguards before handling sensitive data publicly.

Jobs are in process memory; restarting loses status/download registrations.
Multiple independent web workers are not supported by this job store.
Private RFData jobs suppress downloads; uploaded-cohort jobs export patient-level
artifacts through job URLs, which must be treated as sensitive bearer links.
Mutational coordinates are sent to the UCSC API for sequence retrieval.

The source includes JSON API error handling. An `Unexpected token '<'` browser
error means HTML was parsed as JSON, often from an error page/proxy. Inspect the
actual HTTP status/body. Copying this directory does not update a running Linux
deployment; deploy and restart separately. Public large-upload timeouts remain
an unresolved deployment issue, not a certified fix in this release.

## Checks and publication

```bash
python -m unittest discover -s tests -v
```

These are source-release smoke tests, not a historical numerical parity claim.
Review `SOURCE_MANIFEST.json` for included files and original hashes.
Only push the contents of this release directory, **not the parent research
workspace**. No remote repository is created or updated by this export.
Before publication, select a license and add the verified manuscript citation;
neither a redistribution license nor a publication identifier is assumed here.

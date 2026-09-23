# Closed-loop Transformer-guided thermodynamic landscapes for programmable mutation detection
### Transformer-guided small-panel design

**From cohort mutation profiles to interpretable candidate selection and blocker screening.**

MUTAFormer brings together mutation-based classification, feature attribution,
and sequence-aware blocker design in a single research workflow. Researchers can
train a cohort-specific Transformer, identify candidate mutations, and prioritize
blockers using predicted P(dCt > 8).

**[Launch the web tool →](https://dna-panel.mmspectra.cn/)** ·
[Quick start](#quick-start) · [Input format](#input-format) ·
[Reproducing an analysis](#reproducing-an-analysis)

## Overview

MUTAFormer connects model interpretation with small-panel development:

- **Cohort-specific modeling:** train a Transformer using labeled mutation and
  variant allele frequency (VAF) data.
- **Attribution-guided selection:** quantify mutation contributions with Captum
  FeaturePermutation and select candidates by attribution magnitude.
- **Sequence-aware design:** retrieve hg38 flanking sequences and design blockers
  using NUPACK thermodynamic calculations.
- **RF screening:** estimate the probability that a blocker achieves dCt > 8
  and rank candidates using attribution, mutation frequency and predicted performance.
- **Traceable analysis:** export model settings, source hashes, patient predictions
  and intermediate results for review and reproducibility.

```text
Labeled cohort + mutation annotations
                  ↓
       Transformer training
                  ↓
 Permutation attribution + frequency
                  ↓
 Candidate selection → hg38 sequences
                  ↓
    Blocker design → RF screening
                  ↓
    Small-panel recommendations
```

This repository provides the model, attribution and web workflow source code
supporting the study. Gene+Protein and Small-panel training scripts are also
included. The interactive panel-design workflow currently trains a
**mutation-only** model; it is separate from the Gene+Protein evaluation scripts.

## Try it online

Visit **[dna-panel.mmspectra.cn](https://dna-panel.mmspectra.cn/)** to explore the
workflow without installing the software.

Choose the server-held **RFData example**, or select **Upload my own dataset**
to analyze a labeled cohort with its mutation annotations. The RFData example
displays aggregate results and mutation recommendations; its source data and
patient-level outputs are not available for download.

For sensitive cohorts, use a local installation and follow your institution's
data-governance requirements. Upload only data you are authorized to process on
the hosted service. A new training run may select a different panel from the
one reported in the study.

## Quick start

Linux and Python 3.10 are recommended for the complete sequence/blocker workflow.
From the root of your downloaded repository:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r gene_with_order/web/requirements.txt
python gene_with_order/web/server.py
```

Open **http://127.0.0.1:8000/** in your browser. The source distribution defaults
to uploading your own data. Synthetic examples are included in
[`gene_with_order/web/data/`](gene_with_order/web/data), and the interface offers
a small demonstration run.

**For complete blocker screening**, install an authorized NUPACK distribution
and supply the RF classifier and feature schema. These resources are not bundled;
see [resources and reproducibility notes](docs/REPRODUCIBILITY.md).
Training and attribution do not require the private RFData workbook.
Keep the inner directory name `gene_with_order`, which is used by package imports.

## Input format

Both tables accept CSV or Excel. Downloadable templates are available in the web
interface.

### 1. Cohort mutation database

One sample–mutation pair per row:

| Sample_ID | Class | mut_pos | VAF |
| --- | --- | --- | --- |
| EXAMPLE_001 | HCC | chr17:7674216 | 0.125 |
| EXAMPLE_002 | nonHCC | chr5:1295113 | 0.032 |
| EXAMPLE_003 | nonHCC | | |

Use hg38, 1-based mutation coordinates and VAF fractions between 0 and 1.
Represent a mutation-free patient with one row containing blank `mut_pos` and
`VAF`. Provide exactly two classes and select the positive class label in the UI.

### 2. Mutation annotation table

| mut_pos | Ref | Allele |
| --- | --- | --- |
| chr17:7674216 | C | A |
| chr5:1295113 | G | A |

Mutation identifiers must match the cohort database. Use `-` for an empty allele
in an insertion/deletion. The annotation table is required.

At least 10 independent patients per class are required by the implemented
70/10/20 split. Vocabulary size must not exceed the number of distinct mutations.
These are software constraints, not recommendations for adequate study size.
The combined upload limit is 50 MiB.

## How candidates are selected

1. Compute permutation attribution and retain its signed values for inspection.
2. Use **absolute attribution magnitude** for candidate selection; the default
   threshold `attribution > 0` means nonzero magnitude, not positive signed evidence.
3. Attach mutation occurrence counts from the cohort and retrieve reference sequences.
4. Design blockers and predict each mutation branch's **P(dCt > 8)**.
5. Retain blocker groups whose maximum branch probability meets the threshold
   (default 0.5); keep up to 50 by summed attribution, then rank by summed frequency
   and a tie score combining mean probability and normalized attribution.
6. Return the requested number of recommendations (default 10).

The RF output is a **classification probability**, not a continuous dCt estimate.
Recommendations require experimental validation.

## Reproducing an analysis

For a reproducible run, preserve:

- The exact cohort and annotation files, including row order.
- Panel mapping, class encoding, split assignments and random seed.
- Model configuration, checkpoint and software environment.
- Training, inference and explanation batch settings.

The web workflow saves outputs under `gene_with_order/web/runs/<job_id>/`.
Depending on completed stages, these include:

| Output | Contents |
| --- | --- |
| `run_manifest.json` | Configuration, source hashes, model hash and runtime versions |
| `sample_splits.csv` | Patient split assignments |
| `diagnosis_results.xlsx` | Test-set predictions and probabilities |
| `test_predictions.npz` | Test labels, probabilities, logits and batch information |
| `attribution_frequency.xlsx` | Mutation attribution and occurrence frequency |
| `mutation_candidate_withseq.xlsx` | Candidate mutations and flanking sequences |
| `design_blockers_with_energy.xlsx` | Blocker designs and calculated energies |
| `small_panel_recommendations.csv` | Ranked blocker recommendations |

Web defaults are 10 Transformer layers, 16 attention heads, embedding dimension
64, dropout 0.1, seed 11, 20 epochs, learning rate 0.0001, vocabulary size 250,
sequence length 15 and training/evaluation batch size 64. Explanation uses batches
of 4. The synthetic demonstration uses smaller settings.

Reproducing a specific study result requires its matching data, checkpoint and
configuration—not just the web defaults. Private cohorts, trained weights and
experiment archives are not included in this source release. Small-panel and
Gene+Protein scripts require their corresponding datasets and panel mappings;
a complete, one-command reproduction of all study experiments is not provided.

See [reproducibility notes](docs/REPRODUCIBILITY.md) for resource locations,
implementation details and interpretation limits.

## Code guide

| Location under `gene_with_order/` | Function |
| --- | --- |
| `models.py`, `models_for_explain.py` | Transformer architecture and attribution wrapper |
| `train.py`, `dataset_qPCR.py`, `load_as_dict.py` | Training, data splits and mutation encoding |
| `explain.py` | Captum FeaturePermutation attribution |
| `retrain_with_plexfilter.py`, `dataset_pannel.py` | Small-panel modeling |
| `train_with_protein.py`, `dataset_pannel_protein.py` | Gene+Protein modeling |
| `dataset_pannel_blood.py`, `dataset_pannel_tissue.py` | External cohort loaders |
| `web/` | Browser interface, job API and panel-design pipeline |

## Verification

Run the data-free release checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These checks cover syntax, imports, synthetic inputs, JSON API errors and exclusion
of private workbook/model artifacts. They do not replace scientific validation or
a full end-to-end test. `SOURCE_MANIFEST.json` records the release file hashes.

## Research use and data availability

MUTAFormer is a research tool, not a clinical diagnostic service. Candidate panels
require independent evaluation and experimental validation. The implementation
has batch-dependent predictions and uses test-split attribution for panel discovery;
that split must not be treated as an untouched final validation cohort.
Thermodynamic calculation failures also require review before interpreting RF results.

Patient-level data and licensed dependencies are not distributed with this
repository. The hosted example does not grant access to its underlying cohort.
For implementation details and deployment precautions, see the
[technical notes](docs/REPRODUCIBILITY.md) and [web documentation](gene_with_order/web/README.md).

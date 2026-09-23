# MUTAFormer web tool: original-script compatibility

This web directory depends on the unchanged Python scripts and historical archive in the project root. It is not a standalone replacement model implementation.

## Two explicitly different modes

**Legacy retraining (default)** runs the root `train.py`, `dataset_qPCR.py`, `models.py`, `load_as_dict.py` and `explain.py` through `legacy_engine.py`. Every job runs in a separate Python process. It never loads a historic checkpoint instead of performing requested training.

Defaults: 10 layers, 16 heads, embedding 64, dropout 0.1, vocabulary 250, sequence length 15, seed 11, 20 epochs, Adam learning rate 0.0001, training/evaluation batch 64, explanation batch 4. The web learning-rate default (including RFData and synthetic examples) was changed at the user's request from the saved paper configuration's 0.001. Historical verification results below used 0.001 and have not been rerun. Root scripts are unchanged. This is not the four-layer Small-panel evaluation model.

**Historical artifact replay** checks that the upload matches the current project RFData reference in original row order. It reuses the archived attribution table (1218 rows, 74 nonzero entries), archived sequence/blocker tables, and archived predict_68 input. RF probabilities and ranking are recomputed. Upload annotations must match the archived candidates after the original chr17:7674194 correction. It does not train, recompute attribution, or report a new Transformer AUC. Matching the project reference does NOT prove which historical data/checkpoint/RNG originally generated these artifacts.

Never compare a newly trained result and an archived result as though they were the same experiment.

## Exact historical behaviors retained

- Mutation-free patients remain as a single row with blank mut_pos and VAF.
- The original literal "-" VAF (present in RFData) is retained for original sorting behavior. Numeric invalid values are not silently converted to zero.
- The full cohort builds the vocabulary, including its original frequency tie ordering. It is not rebuilt from Train only.
- Patient splitting uses the original class-wise torch.randperm and floor-based 70/10/20 split, not sklearn train_test_split. The positive class maps to 0; the other class maps to 1.
- The original positive-class training duplication is retained. Unique patients never cross splits.
- The original unweighted CrossEntropyLoss receives model probabilities, not logits.
- Validation accuracy, not AUROC, controls the original state_dict reference assignment. Because the old snapshot is a live reference, the resulting checkpoint ordinarily contains final-epoch weights. No deepcopy repair is silently introduced.
- The original repeated test evaluation and save_everything call order are retained, including their RNG consumption.
- The original Transformer batch_first behavior is retained. Consequently patients interact within an inference batch.
- Explanation uses real Captum FeaturePermutation, original model construction/RNG, shuffled batches of 4, target labels, and original advanced-index accumulation. Repeated indices do not have index_add semantics.
- Raw signed attribution is saved. gene_list and the downstream candidate table use the original absolute magnitude. This is not the same as positive signed evidence.
- The original unknown-token attribution naming alias is preserved and identified in the manifest.
- Frequency is the original full-cohort count of mutation occurrences, not a new unique-patient/log-weighted score. The undocumented historical display score is not invented; blocker/RF stages do not consume it.
- Each job exports settings, source hashes, model hash, runtime versions, patient splits, test probabilities/logits, raw attribution and explanation RNG state.

These are compatibility choices, NOT endorsements of the old methodology. The full-cohort vocabulary and use of test attribution for panel discovery mean that the test set is not an untouched final panel validation cohort. Independent validation is still required. Cross-patient attention and imputation make predictions batch/data-pattern dependent.

## Sequence, blocker and RF parity

- Sequence extraction follows get_sequence.R coordinates: 100 bases on each side of the reference allele. SNVs have 201 reference bases; indels need not.
- Annotation selection mirrors distinct triples followed by sorted tie selection. The original chr17:7674194 Ref=GT / Allele=- correction is retained. Other cohorts' annotations are never silently borrowed.
- UCSC hg38 API supplies reference bases equivalent to the R hg38 coordinate definition. A local frozen BSgenome build has not been numerically compared on this Windows host.
- NUPACK uses the original Tube/tube_analysis setup: DNA04.2, stacking, 60 C, sodium 0.06, magnesium 0, strand concentration 4e-7, max_size 2, pairs+mfe.
- Local blocker growth alternates left/right as in blockerdesign.ipynb, targeting -13; global growth extends left, targeting -18. Group splitting, initial over-stable singleton skipping, mutation branch construction and rounding follow that notebook.
- Thermodynamic errors retain the original 0-energy fallback, with warnings. Missing NUPACK is detected before design. Two intentional fail-closed guards stop an otherwise infinite boundary-extension loop or a cluster extending beyond its reference window; no fabricated energy or truncated design is returned.
- All mutation branches are exported, not just branches 1–9.
- RF features match the archived feature_extraction.py, including division-by-zero returning 0. The classifier predicts P(dCt > 8), not continuous dCt.
- Branch sorting, max-probability group acceptance, attribution top 50, frequency/tie-score sorting and top K follow predict_transfer_68.py.
- sklearn 1.3.0 is the archived RF version. A compatibility fallback for newer versions is tested against the archived output to its recorded precision, but a pinned environment remains preferable.

## Input

The default input option is server-held RFData with its project archive annotations.
It runs fixed legacy retraining settings (20 epochs, seed 11, vocabulary 250).
Users can instead select their own uploads. RFData jobs expose aggregate metrics
and mutation recommendations only: all job downloads are denied server-side,
including inputs, patient predictions, splits, logs and model weights. The status
endpoint excludes detailed internal errors and file links. This is not a formal
privacy guarantee: aggregate outputs reveal dataset characteristics. Synthetic
example downloads remain synthetic. Deploy behind an authenticated/rate-limited
service to control repeated compute requests. No RFData download route is provided.

Required cohort columns: Sample_ID, Class, mut_pos, VAF. Exactly two classes; choose the exact positive label. At least 10 independent patients per class are needed for nonempty original Train/Validation/Test splits. The original explanation also requires at least vocab_size distinct mutations. The 20-sample synthetic demo therefore uses vocabulary 4 and 3 epochs; it is not a scientific validation.

A mutation-free patient must have one row with both mutation and VAF blank. Do not mix blank placeholders and mutation rows for the same patient. Conflicting patient labels and malformed mutations fail validation without dropping patients. Age/AFP/Gender are optional here because apply_ehr and if_protein are false; unused defaults are only adapters for the old loader.

Annotation table is required: mut_pos, Ref, Allele. "-" denotes an insertion/deletion empty allele.

## Run and deploy

From the project root:

```text
python -m pip install -r web/requirements.txt
python web/server.py
python web/test_legacy_parity.py
```

Local URL: http://127.0.0.1:8000/. Linux launch: `bash web/start_server.sh`; stop: `bash web/stop_server.sh`. Set PANEL_PYTHON to override the default .venv-panel/bin/python. Use one Python environment; the launcher no longer injects another Conda environment's site-packages. Install the appropriate licensed NUPACK wheel into that same environment. The bundled cp310 Linux wheel requires Python 3.10.

Existing servers must be restarted after copying the changed web files. No remote server was modified by the local repair. Install requirements on the server before restart; the optional local web/vendor/python Captum installation is git-ignored.

The Flask development server is not a hardened public deployment. Patient files and source code are no longer exposed through a catch-all static directory, but production authentication, retention, resource limits and deployment hardening still require review.

## Verification (2026-09-22)

- Direct original-vs-web comparison: every model tensor, signed Captum attribution and gene_list value is exactly equal on a fixed synthetic cohort containing a mutation-free patient.
- Automated tests cover zero-mutation retention, input rejection, literal "-" VAF, SNV/indel coordinates, notebook blocker algorithm parity under a deterministic energy oracle, unreachable-energy guard, RF feature parity, branches beyond 9, archived RF ranking, archive input rejection and HTTP/static behavior.
- Full RFData default 20-epoch legacy retraining completed: 1066 Train / 152 Validation / 307 Test patients; 1411 training rows after legacy duplication. Legacy micro-AUC 0.97985655; patient HCC AUROC 0.97596618. Attribution has 66 nonzero mutations, overlapping 57 of the archived 74. This is a new run, not an exact historical experiment.
- The current RFData frequency list has 1288 mutations, whereas the historical attribution table has 1218. Historical source identity remains unresolved; no claim is made that the tables originated from the identical workbook.
- Historical-mode HTTP run completed: 74 mutation branches, 68 blocker inputs, exact same top-10 names/order as archived RF recommendations; numeric values agree within archived CSV rounding.
- Windows has no working NUPACK installation. Real thermodynamic calculations are NOT certified by the mocked-energy algorithm test. Fresh full design requires a Linux NUPACK integration test; archived replay works without it.
- Browser checks verified mode switching, disabled archive training controls and synthetic-demo submission/status rendering.

Historical runs and original scripts are not overwritten. New artifacts are under web/runs/<job_id>/.

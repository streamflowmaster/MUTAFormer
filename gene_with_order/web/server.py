from __future__ import annotations

import csv
import io
import os
import shutil
import threading
import traceback
import uuid
import json
import math
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import (PipelineError, blocker_to_predict, design_blockers, generate_sequences,
                      nupack_available, predict_dct, read_table, resolve_annotations,
                      train_disease_model, validate_database, write_xlsx)

WEB_DIR = Path(__file__).resolve().parent
RUNS_DIR = WEB_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
ALLOWED = {".xlsx", ".xls", ".csv"}
DEMO_DATABASE = WEB_DIR / "data" / "demo_cohort.csv"
DEMO_ANNOTATION = WEB_DIR / "data" / "demo_annotation.csv"
JOBS = {}
LOCK = threading.Lock()

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.errorhandler(Exception)
def api_error(exc):
    from werkzeug.exceptions import HTTPException
    if not request.path.startswith('/api/'):
        if isinstance(exc, HTTPException):
            return exc
        raise exc
    status = exc.code if isinstance(exc, HTTPException) else 500
    messages = {
        413: 'Upload exceeds the 50 MiB total request limit. Reduce the combined size of the cohort and annotation files.',
        404: 'API endpoint not found. Refresh the page and check that the server has been updated.',
        405: 'This API endpoint does not accept this request method.',
    }
    if status == 500:
        app.logger.exception('API request failed')
    return jsonify(error=messages.get(status, 'The server could not process this request. Check the server logs.'),
                   status=status), status


def update(job_id, **values):
    with LOCK:
        JOBS[job_id].update(values)


def output_file(job_id, path: Path, label: str):
    return {"label": label, "url": f"/api/jobs/{job_id}/files/{path.name}"}


@app.post('/api/jobs/rfdata')
def create_rfdata_job():
    """Fixed server-side example. Never expose cohort or patient-level artifacts."""
    from pipeline import BLOCKER_ARCHIVE
    database = WEB_DIR.parent / 'RFdata_hg38_VAF_Revised.xlsx'
    if not database.is_file() or not BLOCKER_ARCHIVE.is_file():
        return jsonify({'error': 'The server RFData example or its annotation resource is unavailable.'}), 503
    job_id = uuid.uuid4().hex
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir()
    annotation = job_dir / 'internal_annotation.csv'
    settings = dict(analysis_protocol='legacy_retrain', attr_threshold=0.0,
                    cluster_distance=15, prob_threshold=0.5, top_k=10,
                    positive_class='HCC', epochs=20, seed=11, vocab_size=250,
                    block_size=15, learning_rate=0.0001)
    JOBS[job_id] = dict(job_id=job_id, private_example=True, status='running',
                        progress=2, stage='RFData example created',
                        detail='Training on server-held RFData. Downloads are disabled for this example.')
    def prepare_and_run():
        from pipeline import _rename_aliases
        import pandas as pd
        try:
            update(job_id, stage='Loading RFData annotations')
            with zipfile.ZipFile(BLOCKER_ARCHIVE) as archive:
                name = next(n for n in archive.namelist() if 'Hg38' in n and n.lower().endswith('.xlsx'))
                annotations = _rename_aliases(pd.read_excel(io.BytesIO(archive.read(name))))
                if 'mutpos' in annotations and 'mut_pos' not in annotations:
                    annotations = annotations.rename(columns={'mutpos': 'mut_pos'})
                annotations[['mut_pos', 'Ref', 'Allele']].drop_duplicates().to_csv(annotation, index=False)
            run_job(job_id, 'full', database, annotation, None, settings)
        except Exception:
            traceback.print_exc()
            update(job_id, status='failed', progress=100, stage='Example initialization failed',
                   error='The server RFData example could not be initialized.')
    threading.Thread(target=prepare_and_run, daemon=True).start()
    return jsonify({'job_id': job_id}), 202


def run_job(job_id, mode, database_path, annotation_path, blocker_path, settings):
    job_dir = RUNS_DIR / job_id
    files, warnings = [], []
    try:
        if mode == "dct":
            update(job_id, progress=35, stage="Extracting blocker features", detail="Identifying WT/MUT sequences and energy fields.")
            blocker_df = read_table(blocker_path)
        else:
            update(job_id, progress=8, stage="Validating cohort database", detail="Checking samples, classes, mutations, and VAF values.")
            database = validate_database(read_table(database_path))
            diagnosis, attribution, training_metrics = train_disease_model(
                database, settings, job_dir, lambda p, s, d: update(job_id, progress=p, stage=s, detail=d))
            warnings.extend(training_metrics.get('warnings', []))
            diagnosis_path = job_dir / "diagnosis_results.xlsx"
            if training_metrics.get('attribution_recomputed'):
                write_xlsx(diagnosis, diagnosis_path)
            attr_path = job_dir / "attribution_frequency.xlsx"; write_xlsx(attribution, attr_path)
            files += [output_file(job_id, job_dir / "trained_transformer.pth", "Trained model"),
                      output_file(job_id, job_dir / "gene_vocabulary.json", "Mutation vocabulary"),
                      output_file(job_id, job_dir / "training_history.csv", "Training history"),
                      output_file(job_id, job_dir / "training_metrics.json", "Model metrics"),
                      output_file(job_id, diagnosis_path, "Diagnosis results"), output_file(job_id, attr_path, "Attribution table")]
            files = [f for f in files if (job_dir / f['url'].rsplit('/',1)[-1]).exists()]
            for name,label in [('sample_splits.csv','Patient split assignments'),('run_manifest.json','Protocol and provenance'),
                               ('test_predictions.npz','Test probabilities and logits'),('attribution_unormed.pt','Signed raw attribution'),
                               ('gene_list.xlsx','Original absolute attribution export'),('roc_auc.svg','Legacy micro ROC'),
                               ('pr_curve.svg','Legacy micro PR'),('log.txt','Original training log')]:
                if (job_dir/name).exists(): files.append(output_file(job_id,job_dir/name,label))
            auc = training_metrics.get('test_auc')
            displayed_auc = round(auc,3) if auc is not None else 'Archive only; not recomputed'
            candidates = attribution[attribution["attribution"] > settings["attr_threshold"]].copy()
            if candidates.empty:
                update(job_id,status='partial',progress=100,stage='Model analysis completed',detail='No mutations passed the selected threshold.',
                       result={'metrics':{'Legacy micro-AUC':displayed_auc,'Positive attribution':0},'rows':[],'files':files,'warnings':warnings})
                return
            candidate_path = job_dir / "positive_attribution_candidates.xlsx"; write_xlsx(candidates, candidate_path)
            files.append(output_file(job_id, candidate_path, "Candidate mutations"))
            annotations = resolve_annotations(database, annotation_path)
            if settings.get('analysis_protocol') == 'paper_archive':
                from pipeline import BLOCKER_ARCHIVE
                import pandas as pd
                with zipfile.ZipFile(BLOCKER_ARCHIVE) as archive:
                    expected = pd.read_excel(io.BytesIO(archive.read('mutation_candidate_withseq.xlsx')))
                    expected = expected.merge(annotations, left_on='mutpos',right_on='mut_pos',how='left',suffixes=('_old','_upload'))
                    for c in ['Ref','Allele']:
                        if not expected[c+'_old'].astype(str).str.upper().equals(expected[c+'_upload'].astype(str).str.upper()):
                            raise ValueError('Historical replay requires annotations matching the archived 74 candidates (after the original chr17:7674194 correction).')
                    for name,label in [('mutation_candidate_withseq.xlsx','Archived sequences'),('design_blockers_with_energy.xlsx','Archived blocker designs')]:
                        (job_dir/name).write_bytes(archive.read(name))
                        files.append(output_file(job_id,job_dir/name,label))
                    blocker_df = pd.read_csv(io.BytesIO(archive.read('predict_dCt/Data/raw/predict_68.csv')))
                warnings.append('Sequence and blocker tables are historical artifacts, not newly generated designs. RF screening below is recomputed from the archived predict_68 input; filter changes may alter the final selection.')
                finish_dct(job_id,job_dir,blocker_df,settings,files,warnings)
                return
            try:
                sequences = generate_sequences(candidates, annotations, lambda p, s, d: update(job_id, progress=p, stage=s, detail=d))
            except PipelineError as exc:
                warnings.append(str(exc))
                result = {
                    "metrics": {"Legacy micro-AUC": displayed_auc, "Attribution mutations": attribution.shape[0],
                                "Positive attribution": candidates.shape[0], "Stages completed": "2 / 5"},
                    "rows": [{"name": r.mutpos, "mut_count": 1, "probability": None, "frequency": r.freq,
                              "attribution": r.attribution, "pass": True} for r in candidates.head(settings["top_k"]).itertuples()],
                    "files": files, "warnings": warnings}
                update(job_id, status="partial", progress=100, stage="Model analysis completed", detail="Check the mutation annotations or hg38 network access before continuing.", result=result)
                return
            seq_path = job_dir / "mutation_candidate_withseq.xlsx"; write_xlsx(sequences, seq_path)
            files.append(output_file(job_id, seq_path, "Candidate sequences"))
            if not nupack_available():
                warnings.append("NUPACK is unavailable. Sequence generation is complete; run in a Linux/Docker environment or continue from a blocker table.")
                result = {"metrics": {"Legacy micro-AUC": displayed_auc, "Candidate mutations": len(candidates),
                                      "Sequences generated": len(sequences), "Stages completed": "3 / 5"},
                          "rows": [{"name": r.mutpos, "mut_count": 1, "probability": None, "frequency": r.freq,
                                    "attribution": r.attribution, "pass": True} for r in sequences.head(settings["top_k"]).itertuples()],
                          "files": files, "warnings": warnings}
                update(job_id, status="partial", progress=100, stage="Sequence generation completed", detail="The NUPACK energy environment is unavailable.", result=result)
                return
            update(job_id, progress=62, stage="Designing blockers", detail="NUPACK is calculating WT/MUT binding energies.")
            blocker_df = design_blockers(sequences, settings["cluster_distance"], lambda p, s, d: update(job_id, progress=p, stage=s, detail=d))
            blocker_xlsx = job_dir / "design_blockers_with_energy.xlsx"; write_xlsx(blocker_df, blocker_xlsx)
            files.append(output_file(job_id, blocker_xlsx, "Blocker designs"))

        finish_dct(job_id,job_dir,blocker_df,settings,files,warnings)
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="failed", progress=100, stage="Job failed", detail="The analysis did not complete.", error=str(exc))


def finish_dct(job_id,job_dir,blocker_df,settings,files,warnings):
        update(job_id, progress=84, stage="Predicting dCt", detail="RF classification: estimating P(dCt > 8), not a continuous dCt value.")
        predict_input = blocker_to_predict(blocker_df)
        predict_path = job_dir / "predict_input.csv"; predict_input.to_csv(predict_path, index=False, encoding="utf-8-sig")
        expanded, summary, top = predict_dct(predict_input, settings["prob_threshold"], settings["top_k"])
        expanded_path = job_dir / "dct_branch_results.csv"; expanded.to_csv(expanded_path, index=False, encoding="utf-8-sig")
        summary_path = job_dir / "dct_passed_candidates.csv"; summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        top_path = job_dir / "small_panel_recommendations.csv"; top.to_csv(top_path, index=False, encoding="utf-8-sig")
        files += [output_file(job_id, expanded_path, "Branch predictions"), output_file(job_id, summary_path, "Passed candidates"), output_file(job_id, top_path, "Panel recommendations")]
        rows = [{"name": r.primer_WT_name, "mut_count": int(r.mut_count),
                 "probability": float(r.max_pred_prob_dCt_gt_8), "frequency": float(r.sum_Freq),
                 "attribution": float(r.sum_Attribution), "pass": True} for r in top.itertuples()]
        result = {"metrics": {"Blocker inputs": len(predict_input), "Mutation branches": len(expanded),
                              "dCt>8 candidates": len(summary), "Panel output": len(top)},
                  "rows": rows, "files": files, "warnings": warnings}
        update(job_id, status="done", progress=100, stage="Panel design completed", detail=f"Generated {len(top)} recommended designs.", result=result)


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get('/<filename>')
def assets(filename):
    if filename not in {'styles.css','training.css','app.js'}:
        return jsonify({'error':'Not found'}),404
    return send_from_directory(WEB_DIR,filename)


@app.get("/api/health")
def health():
    import importlib.util
    import sys
    vendor=str(WEB_DIR/'vendor/python')
    if vendor not in sys.path: sys.path.append(vendor)
    captum=importlib.util.find_spec('captum') is not None
    thermo=nupack_available()
    return jsonify({"status": "ok", "transformer": captum, "captum":captum,"rf": True,
                    "protocol":'root_scripts_v1',"nupack":thermo,"full_pipeline":thermo and captum})


@app.get("/api/template/database")
def database_template():
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(["Sample_ID", "Class", "mut_pos", "VAF"])
    writer.writerow(["SAMPLE_001", "HCC", "chr17:7674216", "0.125"])
    writer.writerow(["SAMPLE_002", "nonHCC", "chr5:1295113", "0.032"])
    writer.writerow(["SAMPLE_003", "nonHCC", "", ""])
    return app.response_class(stream.getvalue(), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=mutation_database_template.csv"})


@app.get("/api/template/annotation")
def annotation_template():
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(["mut_pos", "Ref", "Allele"])
    writer.writerow(["chr17:7674216", "C", "A"])
    writer.writerow(["chr5:1295113", "G", "A"])
    return app.response_class(stream.getvalue(), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=mutation_annotation_template.csv"})


@app.get("/api/demo/<kind>")
def download_demo(kind):
    files = {"database": DEMO_DATABASE, "annotation": DEMO_ANNOTATION}
    path = files.get(kind)
    if not path:
        return jsonify({"error": "Unknown demo file."}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.post("/api/jobs/demo")
def create_demo_job():
    job_id = uuid.uuid4().hex
    job_dir = RUNS_DIR / job_id; job_dir.mkdir()
    database_path = job_dir / "demo_cohort.csv"
    annotation_path = job_dir / "demo_annotation.csv"
    shutil.copyfile(DEMO_DATABASE, database_path)
    shutil.copyfile(DEMO_ANNOTATION, annotation_path)
    settings = {"attr_threshold": 0.0, "cluster_distance": 15, "prob_threshold": 0.5,
                "top_k": 10, "positive_class": "HCC", "epochs": 3, "seed": 11,
                "vocab_size": 4, "block_size": 15, "learning_rate": 0.0001,'analysis_protocol':'legacy_retrain'}
    JOBS[job_id] = {"job_id": job_id, "status": "running", "progress": 2,
                    "stage": "Demo job created", "detail": "Loading the built-in example dataset."}
    threading.Thread(target=run_job,
                     args=(job_id, "full", database_path, annotation_path, None, settings),
                     daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.post("/api/jobs")
def create_job():
    mode = request.form.get("mode", "full")
    if mode not in {'full','dct'}: return jsonify({'error':'Unknown input mode.'}),400
    upload = request.files.get("database" if mode == "full" else "blockers")
    if not upload or not upload.filename:
        return jsonify({"error": "Please select an input file."}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": "Only .xlsx, .xls, and .csv files are supported."}), 400
    job_id = uuid.uuid4().hex
    job_dir = RUNS_DIR / job_id; job_dir.mkdir()
    input_path = job_dir / ("database" + suffix if mode == "full" else "blockers" + suffix)
    upload.save(input_path)
    annotation_path = None
    annotation = request.files.get("annotation")
    if mode == "full" and (not annotation or not annotation.filename):
        return jsonify({"error": "A mutation annotation table is required. Required columns: mut_pos, Ref, Allele."}), 400
    if mode == "full":
        ann_suffix = Path(annotation.filename).suffix.lower()
        if ann_suffix not in ALLOWED:
            return jsonify({"error": "The mutation annotation table must be an .xlsx, .xls, or .csv file."}), 400
        annotation_path = job_dir / ("annotation" + ann_suffix); annotation.save(annotation_path)
    try:
        settings = {"analysis_protocol":request.form.get('analysis_protocol','legacy_retrain'),
                    "attr_threshold": float(request.form.get("attr_threshold", 0)),
                    "cluster_distance": max(0, min(100, int(request.form.get("cluster_distance", 15)))),
                    "prob_threshold": max(0.0, min(1.0, float(request.form.get("prob_threshold", .5)))),
                    "top_k": max(1, min(100, int(request.form.get("top_k", 10)))),
                    "positive_class": request.form.get("positive_class", "HCC").strip(),
                    "epochs": max(1, min(500, int(request.form.get("epochs", 20)))),
                    "seed": int(request.form.get("seed", 11)),
                    "vocab_size": max(4, min(5000, int(request.form.get("vocab_size", 250)))),
                    "block_size": max(1, min(256, int(request.form.get("block_size", 15)))),
                    "learning_rate": float(request.form.get("learning_rate", 0.0001))}
        if settings['analysis_protocol'] not in {'legacy_retrain','paper_archive'}: raise ValueError('Unknown protocol')
        if not all(math.isfinite(settings[k]) for k in ['attr_threshold','prob_threshold','learning_rate']): raise ValueError('Nonfinite setting')
        if settings['learning_rate']<=0: raise ValueError('Learning rate must be positive')
        if settings['analysis_protocol']=='paper_archive' and (settings['attr_threshold']!=0 or settings['cluster_distance']!=15):
            raise ValueError('Historical archive replay requires attribution threshold 0 and clustering distance 15.')
    except ValueError:
        return jsonify({"error": "One or more parameter values are invalid."}), 400
    JOBS[job_id] = {"job_id": job_id, "status": "running", "progress": 2,
                    "stage": "Job created", "detail": "Reading uploaded files."}
    args = (job_id, mode, input_path if mode == "full" else None, annotation_path,
            input_path if mode == "dct" else None, settings)
    threading.Thread(target=run_job, args=args, daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = JOBS.get(job_id)
    if job and job.get('private_example'):
        public = {k: job[k] for k in ['job_id', 'status', 'progress', 'stage'] if k in job}
        public['detail'] = 'Server-held RFData example; raw data and patient-level outputs cannot be downloaded.'
        if job.get('status') == 'failed':
            public['error'] = 'RFData example failed. Ask the administrator to inspect the server logs.'
        if 'result' in job:
            result = job['result']
            public['result'] = dict(metrics=result.get('metrics', {}), rows=result.get('rows', []), files=[],
                warnings=['RFData is processed on the server. Downloads are disabled. Aggregate results and mutation recommendations remain visible.',
                          'New training does not guarantee the historical panel. Legacy modeling limitations are retained.'])
        return jsonify(public)
    return jsonify(job) if job else (jsonify({"error": "The job does not exist or the service has restarted."}), 404)


@app.get("/api/jobs/<job_id>/files/<filename>")
def download(job_id, filename):
    job_dir = RUNS_DIR / job_id
    if job_id not in JOBS or not job_dir.exists():
        return jsonify({"error": "The job does not exist."}), 404
    if JOBS[job_id].get('private_example'):
        return jsonify({'error': 'Downloads are disabled for the server-held RFData example.'}), 403
    allowed = {f['url'].rsplit('/', 1)[-1] for f in JOBS[job_id].get('result', {}).get('files', [])}
    if filename not in allowed:
        return jsonify({'error': 'This file is not available for download.'}), 403
    return send_from_directory(job_dir, secure_filename(filename), as_attachment=True)


if __name__ == "__main__":
    host = os.environ.get("PANEL_HOST", "127.0.0.1")
    port = int(os.environ.get("PANEL_PORT", "8000"))
    app.run(host=host, port=port, debug=False, threaded=True)

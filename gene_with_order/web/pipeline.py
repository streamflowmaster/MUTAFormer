from __future__ import annotations

import io
import json
import math
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from copy import deepcopy
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
MODEL_PATH = ROOT_DIR / "final_model_v0" / "model.pth"
GENE_MAP_PATH = WEB_DIR / "data" / "gene_map.json"
BLOCKER_ARCHIVE = ROOT_DIR / "blocker_design(1).zip"
RF_DIR = WEB_DIR / "data" / "predict_dCt" / "predict_dCt" / "Data" / "processed"


class PipelineError(ValueError):
    pass


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise PipelineError("Only .xlsx, .xls, and .csv files are supported.")


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "sample_id": "Sample_ID", "sampleid": "Sample_ID", "sample": "Sample_ID",
        "class": "Class", "label": "Class", "group": "Class",
        "mut_pos": "mut_pos", "mutpos": "mut_pos", "mutation": "mut_pos",
        "vaf": "VAF", "ref": "Ref", "reference": "Ref",
        "allele": "Allele", "alt": "Allele", "alternate": "Allele",
    }
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(col).strip().lower())
        if key in aliases:
            rename[col] = aliases[key]
    return df.rename(columns=rename)


def validate_database(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_aliases(df.copy())
    required = ["Sample_ID", "Class", "mut_pos", "VAF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PipelineError("The cohort database is missing required columns: " + ", ".join(missing))
    df = df.dropna(how="all").copy()
    if df[["Sample_ID", "Class"]].isna().any().any():
        raise PipelineError("Every patient row needs Sample_ID and Class; no patient is silently removed.")
    df["Sample_ID"] = df["Sample_ID"].astype(str).str.strip()
    df["Class"] = df["Class"].astype(str).str.strip()
    if df.Sample_ID.eq("").any() or df.Class.eq("").any():
        raise PipelineError("Sample_ID and Class cannot be blank.")
    df["mut_pos"] = df["mut_pos"].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).replace("", np.nan)
    parsed_vaf = pd.to_numeric(df["VAF"], errors="coerce")
    legacy_dash = df.VAF.eq('-')
    if (df.VAF.notna() & parsed_vaf.isna() & ~legacy_dash).any():
        raise PipelineError("VAF contains non-numeric values; they cannot be silently replaced by zero.")
    df["VAF"] = parsed_vaf.astype(object) if legacy_dash.any() else parsed_vaf
    if legacy_dash.any():
        df.loc[legacy_dash, 'VAF'] = '-'
    if (df.mut_pos.notna() & ~legacy_dash & (~np.isfinite(parsed_vaf) | (parsed_vaf < 0) | (parsed_vaf > 1))).any():
        raise PipelineError("Each mutation needs a finite VAF between 0 and 1. Leave both fields blank for a mutation-free patient.")
    invalid = df.mut_pos.notna() & ~df["mut_pos"].str.match(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y):\d+$", na=False)
    if invalid.any():
        examples = ", ".join(df.loc[invalid, "mut_pos"].head(3))
        raise PipelineError(f"mut_pos must use the chr1:12345 format. Invalid examples: {examples}")
    if df.empty:
        raise PipelineError("The uploaded file contains no valid records.")
    if df.groupby('Sample_ID').Class.nunique().gt(1).any():
        raise PipelineError("A Sample_ID has conflicting class labels.")
    for sid, group in df.groupby('Sample_ID', sort=False):
        if group.mut_pos.isna().any() and len(group) != 1:
            raise PipelineError(f"{sid}: a mutation-free patient must have exactly one empty mutation row. Do not mix placeholder and mutation rows.")
    return df.reset_index(drop=True)


def _gene_map() -> dict[str, int]:
    """Legacy HCC map retained only for checkpoint compatibility tests."""
    return {str(gene): int(token) for gene, token in json.loads(GENE_MAP_PATH.read_text(encoding="utf-8")).items()}


def _load_transformer():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from models import GPTConfig, transformer_cls

    config = GPTConfig(block_size=15, vocab_size=250, n_layer=10, n_head=16,
                       n_embd=64, dropout=0.1, bias=True, cls_num=2,
                       apply_ehr=False, if_vaf_sort=True, if_qpcr=False)
    model = transformer_cls(config)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def train_disease_model(df: pd.DataFrame, settings: dict, output_dir: Path, progress=None):
    """Run unmodified root scripts in a job-isolated process, never Flask global RNG."""
    import os
    import subprocess
    progress = progress or (lambda *_: None)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Internal, server-created pickle preserves exact float values; uploads never accept pickle.
    df.to_pickle(output_dir / "engine_input.pkl")
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    with (output_dir / "worker.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen([sys.executable, "-u", str(WEB_DIR / "legacy_engine.py"), str(output_dir)],
                                  cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        tail = []
        for line in worker.stdout:
            log.write(line); log.flush()
            tail = (tail + [line])[-12:]
            if line.startswith("WEB_PROGRESS "):
                progress(*json.loads(line[len("WEB_PROGRESS "):]))
        worker.stdout.close()
        if worker.wait() != 0:
            raise PipelineError("Original-script worker failed: " + "".join(tail)[-2200:])
    return (pd.read_csv(output_dir / "engine_diagnosis.csv"),
            pd.read_csv(output_dir / "engine_attribution.csv"),
            json.loads((output_dir / "training_metrics.json").read_text(encoding="utf-8")))


def _builtin_annotation() -> pd.DataFrame:
    if not BLOCKER_ARCHIVE.exists():
        return pd.DataFrame(columns=["mut_pos", "Ref", "Allele"])
    with zipfile.ZipFile(BLOCKER_ARCHIVE) as archive:
        names = [n for n in archive.namelist() if "Hg38" in n and n.lower().endswith(".xlsx")]
        if not names:
            return pd.DataFrame(columns=["mut_pos", "Ref", "Allele"])
        df = pd.read_excel(io.BytesIO(archive.read(names[0])))
    df = _rename_aliases(df)
    if "mutpos" in df.columns and "mut_pos" not in df.columns:
        df = df.rename(columns={"mutpos": "mut_pos"})
    keep = [c for c in ["mut_pos", "Ref", "Allele"] if c in df.columns]
    return df[keep].drop_duplicates("mut_pos") if len(keep) == 3 else pd.DataFrame(columns=["mut_pos", "Ref", "Allele"])


def resolve_annotations(database: pd.DataFrame, annotation_path: Path | None) -> pd.DataFrame:
    frames = []
    if {"mut_pos", "Ref", "Allele"}.issubset(database.columns):
        frames.append(database[["mut_pos", "Ref", "Allele"]])
    if annotation_path:
        ann = _rename_aliases(read_table(annotation_path))
        if "mutpos" in ann.columns and "mut_pos" not in ann.columns:
            ann = ann.rename(columns={"mutpos": "mut_pos"})
        missing = [c for c in ["mut_pos", "Ref", "Allele"] if c not in ann.columns]
        if missing:
            raise PipelineError("The mutation annotation table is missing required columns: " + ", ".join(missing))
        frames.append(ann[["mut_pos", "Ref", "Allele"]])
    if not frames:
        raise PipelineError("A mutation annotation table is required; annotations are never silently borrowed from another cohort.")
    ann = pd.concat(frames, ignore_index=True).dropna(subset=["mut_pos"])
    ann = ann.drop_duplicates(['mut_pos','Ref','Allele'])
    # get_sequence.R: distinct triples, then count/slice_max. Counts are all 1
    # after distinct; R count sorts the grouping keys, so ties take the first.
    ann = ann.sort_values(['mut_pos','Ref','Allele'],kind='stable').drop_duplicates('mut_pos')
    # Original R applies this correction AFTER a left join, so it also fills
    # a candidate absent from the annotation workbook, not only existing rows.
    legacy_site = 'chr17:7674194'
    if database.mut_pos.eq(legacy_site).any() and not ann.mut_pos.eq(legacy_site).any():
        ann = pd.concat([ann, pd.DataFrame([{'mut_pos': legacy_site, 'Ref': 'GT', 'Allele': '-'}])], ignore_index=True)
    ann.loc[ann.mut_pos.eq('chr17:7674194'), ['Ref','Allele']] = ['GT','-']
    for c in ['Ref','Allele']:
        if ann[c].isna().any() or not ann[c].astype(str).str.fullmatch(r'[ACGTacgt]+|-').all():
            raise PipelineError(f"Annotation {c} must contain A/C/G/T bases or '-'.")
    return ann


def _fetch_hg38(mutpos: str, ref: str, alt: str):
    chrom, pos_text = mutpos.split(":", 1)
    pos = int(pos_text)
    clean_ref = "" if str(ref) == "-" else str(ref).upper()
    clean_alt = "" if str(alt) == "-" else str(alt).upper()
    start = pos - 101
    end = pos - 1 + len(clean_ref) + 100
    query = urllib.parse.urlencode({"genome": "hg38", "chrom": chrom, "start": start, "end": end})
    req = urllib.request.Request("https://api.genome.ucsc.edu/getData/sequence?" + query,
                                 headers={"User-Agent": "MUTAFormer-panel-designer/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        dna = json.loads(response.read().decode("utf-8"))["dna"]
    left = dna[:100].lower()
    real_ref = dna[100:100 + len(clean_ref)].upper()
    right = dna[100 + len(clean_ref):].lower()
    return left + real_ref + right, left + clean_alt + right, clean_ref == real_ref


def generate_sequences(candidates: pd.DataFrame, annotations: pd.DataFrame, progress=None) -> pd.DataFrame:
    progress = progress or (lambda *_: None)
    merged = candidates.merge(annotations, left_on="mutpos", right_on="mut_pos", how="left")
    missing = merged[merged[["Ref", "Allele"]].isna().any(axis=1)]["mutpos"].tolist()
    if missing:
        preview = "、".join(missing[:5])
        raise PipelineError(f"{len(missing)} candidates are missing Ref/Allele annotations. Examples: {preview}")
    rows = []
    for i, row in merged.iterrows():
        try:
            ref_seq, alt_seq, matched = _fetch_hg38(row["mutpos"], row["Ref"], row["Allele"])
        except Exception as exc:
            raise PipelineError(f"Unable to retrieve the hg38 sequence for {row['mutpos']}: {exc}") from exc
        rows.append({"mutpos": row["mutpos"], "attribution": row["attribution"], "freq": row["freq"],
                     "Ref": row["Ref"], "Allele": row["Allele"], "is_ref_match": matched,
                     "ref_seq_full": ref_seq, "alt_seq_full": alt_seq})
        progress(42 + int(18 * (i + 1) / len(merged)), "Extracting hg38 sequences",
                 f"Retrieved 201 bp windows for {i + 1}/{len(merged)} candidates.")
    return pd.DataFrame(rows)


def nupack_available() -> bool:
    try:
        import nupack  # noqa: F401
        return True
    except Exception:
        return False


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _energy(seq1: str, seq2: str, temperature=60.0) -> float:
    from nupack import Model, Strand, Tube, SetSpec, tube_analysis
    model = Model(material="dna04.2", ensemble="stacking", celsius=temperature, sodium=0.06, magnesium=0.0)
    a, b = Strand(seq1, name="a"), Strand(seq2, name="b")
    tube = Tube(strands={a:4e-7,b:4e-7},complexes=SetSpec(max_size=2),name='Tube t1')
    result = tube_analysis(tubes=[tube],compute=['pairs','mfe'],model=model)
    duplex = result['(a+b)'] if '(a+b)' in result else result['(b+a)']
    return float(duplex.mfe[0][1])


def _safe_energy(seq1, seq2, temperature=60):
    # notebook safe_calculate_energy2 returns 0 for thermodynamic exceptions.
    # Check availability BEFORE entering design, not by disguising ImportError as energy.
    import warnings
    try:
        return _energy(seq1,seq2,temperature)
    except Exception as exc:
        warnings.warn(f"Legacy energy fallback to 0.0: {type(exc).__name__}: {exc}")
        return 0.0


def design_blockers(seq_df: pd.DataFrame, max_cluster_dist=15, progress=None) -> pd.DataFrame:
    progress = progress or (lambda *_: None)
    if not nupack_available():
        raise PipelineError("NUPACK is required for blocker design.")
    if seq_df.empty: raise PipelineError("No sequences to design.")
    df = seq_df.copy()
    df[["chr", "pos"]] = df["mutpos"].str.split(":", n=1, expand=True)
    df["pos"] = df["pos"].astype(int)
    df = df.sort_values(["chr", "pos"]).reset_index(drop=True)
    groups, current = [], [0]
    for i in range(1, len(df)):
        prev = current[-1]
        if df.loc[i, "chr"] == df.loc[prev, "chr"] and df.loc[i, "pos"] - df.loc[prev, "pos"] <= max_cluster_dist:
            current.append(i)
        else:
            groups.append(current); current = [i]
    if current:
        groups.append(current)
    results = []
    queue = list(groups)
    while queue:
        group = queue.pop(0)
        base_idx, base_pos = group[0], int(df.loc[group[0], "pos"])
        ref_seq = str(df.loc[base_idx, "ref_seq_full"]).upper()
        rel = [100 + (int(df.loc[idx, "pos"]) - base_pos) for idx in group]
        left, right = min(rel) - 1, max(rel)
        local = ref_seq[left:right + 1]
        if max(rel) >= len(ref_seq):
            raise PipelineError("A chained mutation cluster exceeds its reference window; original notebook would truncate it.")
        if _safe_energy(_reverse_complement(local), local) < -13:
            if len(group) > 1:
                mid = len(group) // 2; queue.extend([group[:mid], group[mid:]])
            continue
        while _safe_energy(_reverse_complement(ref_seq[left:right + 1]), ref_seq[left:right + 1]) > -13:
            if left == 0 and right == len(ref_seq)-1:
                raise PipelineError("Local energy target is unreachable within the sequence window; stopped instead of the notebook's infinite loop.")
            if left > 0:
                left -= 1
                target_left = ref_seq[left:right+1]
                if _safe_energy(_reverse_complement(target_left),target_left) <= -13: break
            if right < len(ref_seq)-1: right += 1
        local = ref_seq[left:right + 1]
        global_left = left
        while _safe_energy(_reverse_complement(ref_seq[global_left:right + 1]), ref_seq[global_left:right + 1]) > -18 and global_left > 0:
            global_left -= 1
        target = ref_seq[global_left:right + 1]
        wt_dg = _safe_energy(_reverse_complement(target), target)
        result = {"Target_Mutations": ", ".join(df.loc[group, "mutpos"]), "Blocker_Sequence": target,
                  "Target_WT": target, "Red_Region_Blocker": _reverse_complement(local),
                  "WT_dG": round(wt_dg, 2), "Local_dG": round(_safe_energy(_reverse_complement(local), local), 2),
                  "Mutation_Count": len(group)}
        for branch, idx in enumerate(group, 1):
            ref = "" if str(df.loc[idx, "Ref"]) == "-" else str(df.loc[idx, "Ref"])
            alt = "" if str(df.loc[idx, "Allele"]) == "-" else str(df.loc[idx, "Allele"])
            local_idx = 100 + (int(df.loc[idx, "pos"]) - base_pos) - global_left
            mutated = target[:local_idx] + alt + target[local_idx + len(ref):]
            mut_dg = _safe_energy(target, _reverse_complement(mutated))
            result.update({f"Ref_{branch}": df.loc[idx, "Ref"], f"Alt_{branch}": df.loc[idx, "Allele"],
                           f"Target_MUT_{branch}": mutated, f"MUT_dG_{branch}": round(mut_dg, 2),
                           f"ddG_{branch}": round(mut_dg - wt_dg, 2),
                           f"Attribution_{branch}": df.loc[idx, "attribution"], f"Freq_{branch}": df.loc[idx, "freq"]})
        results.append(result)
        progress(62 + int(18 * len(results) / max(1, len(groups))), "Designing blockers",
                 f"Completed {len(results)} blocker groups.")
    return pd.DataFrame(results)


def blocker_to_predict(df: pd.DataFrame) -> pd.DataFrame:
    if {"primer_WT_name", "primer_seq", "WT_target_seq", "d_primer_WT"}.issubset(df.columns):
        return df.copy()
    required = {"Target_Mutations", "Blocker_Sequence", "Target_WT", "WT_dG"}
    if not required.issubset(df.columns):
        raise PipelineError("The blocker table matches neither the predict_*.csv nor the design_blockers_with_energy.xlsx format.")
    out = pd.DataFrame({"primer_WT_name": df["Target_Mutations"], "primer_seq": df["Blocker_Sequence"],
                        "WT_target_seq": df["Target_WT"], "d_primer_WT": df["WT_dG"]})
    branches=sorted(int(m.group(1)) for c in df.columns if (m:=re.fullmatch(r'Target_MUT_(\d+)',c)))
    for i in branches:
        src = [f"Target_MUT_{i}", f"MUT_dG_{i}", f"ddG_{i}", f"Attribution_{i}", f"Freq_{i}"]
        if src[0] not in df.columns: break
        for source, target in zip(src, [f"MUT_target_seq_{i}", f"d_primer_MUT_{i}", f"ddG_{i}", f"Attribution_{i}", f"Freq_{i}"]):
            out[target] = df[source]
    return out


def _clean_seq(value) -> str:
    return re.sub(r"[^ACGT]", "", "" if pd.isna(value) else str(value).upper())


def _max_comp(a: str, b: str) -> int:
    b = _reverse_complement(b); best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]: k += 1
            best = max(best, k)
    return best


def _seq_features(seq: str, prefix: str):
    seq = _clean_seq(seq); n = len(seq); pairs = Counter(seq[i:i + 2] for i in range(max(0, n - 1)))
    total_pairs = max(1, n - 1)
    longest = max((len(x.group()) for x in re.finditer(r"(A+|C+|G+|T+)", seq)), default=0)
    out = {f"{prefix}_len": n, f"{prefix}_gc": (seq.count('G') + seq.count('C')) / n if n else 0,
           f"{prefix}_at": (seq.count('A') + seq.count('T')) / n if n else 0,
           f"{prefix}_tm_simple": 2 * (seq.count('A') + seq.count('T')) + 4 * (seq.count('G') + seq.count('C')),
           f"{prefix}_longest_homopolymer": longest}
    out.update({f"{prefix}_{base}_frac": seq.count(base) / n if n else 0 for base in "ACGT"})
    out.update({f"{prefix}_dinuc_{a+b}": pairs.get(a+b, 0) / total_pairs for a in "ACGT" for b in "ACGT"})
    return out


def _candidate_features(wt, mut, primer, dwt, dmut, ddg):
    wt, mut, primer = map(_clean_seq, [wt, mut, primer]); out = {}
    out.update(_seq_features(wt, "wt")); out.update(_seq_features(mut, "mut")); out.update(_seq_features(primer, "primer"))
    gc = lambda s: (s.count("G") + s.count("C")) / len(s) if s else 0
    dwt, dmut, ddg = [pd.to_numeric(x, errors="coerce") for x in [dwt, dmut, ddg]]
    out.update({"wt_mut_same_len": int(len(wt) == len(mut)), "wt_mut_len_diff": abs(len(wt) - len(mut)),
                "primer_wt_len_diff": abs(len(primer) - len(wt)), "primer_mut_len_diff": abs(len(primer) - len(mut)),
                "wt_mut_gc_diff": abs(gc(wt) - gc(mut)), "primer_wt_gc_diff": abs(gc(primer) - gc(wt)),
                "primer_mut_gc_diff": abs(gc(primer) - gc(mut)), "primer_wt_max_comp_match": _max_comp(primer, wt),
                "primer_mut_max_comp_match": _max_comp(primer, mut), "primer_self_comp_match": _max_comp(primer, primer),
                "d_primer_WT": dwt, "d_primer_MUT": dmut, "ddG": ddg,
                "d_primer_diff_MUT_minus_WT": dmut - dwt if pd.notna(dmut) and pd.notna(dwt) else np.nan,
                "abs_ddG": abs(ddg) if pd.notna(ddg) else np.nan,
                "ratio_d_primer_MUT_WT": (dmut / dwt if dwt != 0 else 0.0) if pd.notna(dmut) and pd.notna(dwt) else np.nan})
    return out


def predict_dct(df: pd.DataFrame, probability_threshold=0.5, top_k=10):
    df = blocker_to_predict(df)
    mut_indices = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"MUT_target_seq_(\d+)$", c))})
    if not mut_indices: raise PipelineError("No mutation branch columns such as MUT_target_seq_1 were found.")
    expanded = []
    for row_idx, row in df.iterrows():
        for branch in mut_indices:
            mut = row.get(f"MUT_target_seq_{branch}")
            if not _clean_seq(mut): continue
            feats = _candidate_features(row.get("WT_target_seq"), mut, row.get("primer_seq"), row.get("d_primer_WT"),
                                        row.get(f"d_primer_MUT_{branch}"), row.get(f"ddG_{branch}"))
            expanded.append({"source_type": "predict", "original_row_index": row_idx, "csv_row_number": row_idx + 2,
                             "primer_WT_name": row.get("primer_WT_name") if pd.notna(row.get("primer_WT_name")) and str(row.get("primer_WT_name")).strip() else f"row_{row_idx}", "mut_branch": branch,
                             "Attribution": pd.to_numeric(row.get(f"Attribution_{branch}"), errors="coerce"),
                             "Freq": pd.to_numeric(row.get(f"Freq_{branch}"), errors="coerce"), **feats})
    if not expanded: raise PipelineError("The blocker table contains no mutation branches that can be predicted.")
    expanded_df = pd.DataFrame(expanded)
    model = joblib.load(RF_DIR / "rf_model.joblib")
    # The bundled model was trained with scikit-learn 1.3. Newer sklearn
    # versions expect this attribute on every restored decision tree.
    forest = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    for tree in getattr(forest, "estimators_", []):
        if not hasattr(tree, "monotonic_cst"):
            tree.monotonic_cst = None
    feature_cols = json.loads((RF_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    for col in feature_cols:
        if col not in expanded_df: expanded_df[col] = np.nan
    features = expanded_df[feature_cols]
    try:
        prob = model.predict_proba(features)[:, 1]
        if np.nanmin(prob) < 0 or np.nanmax(prob) > 1:
            raise ValueError("legacy tree values require normalization")
    except (AttributeError, ValueError):
        # sklearn 1.4+ changed persisted tree leaf values from counts to
        # proportions. A model saved by 1.3 therefore needs the historical
        # per-leaf normalization before averaging trees.
        transformed = model.named_steps["imputer"].transform(features)
        tree_probs = []
        for tree in forest.estimators_:
            leaf = tree.apply(transformed)
            counts = tree.tree_.value[leaf, 0, :]
            totals = counts.sum(axis=1, keepdims=True)
            tree_probs.append(np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0))
        prob = np.mean(tree_probs, axis=0)[:, 1]
    expanded_df["pred_prob_dCt_gt_8"] = prob
    expanded_df["pred_label_dCt_gt_8"] = (prob >= probability_threshold).astype(int)
    expanded_df = expanded_df.sort_values(['pred_label_dCt_gt_8','pred_prob_dCt_gt_8','Freq','Attribution'],ascending=False).reset_index(drop=True)
    summary = expanded_df.groupby("original_row_index", as_index=False).agg(
        csv_row_number=("csv_row_number", "first"), primer_WT_name=("primer_WT_name", "first"),
        mut_count=("mut_branch", "count"), max_pred_prob_dCt_gt_8=("pred_prob_dCt_gt_8", "max"),
        mean_pred_prob_dCt_gt_8=("pred_prob_dCt_gt_8", "mean"), sum_Freq=("Freq", "sum"),
        sum_Attribution=("Attribution", "sum"))
    summary["pred_label_dCt_gt_8"] = (summary["max_pred_prob_dCt_gt_8"] >= probability_threshold).astype(int)
    eligible = summary[summary["pred_label_dCt_gt_8"] == 1].copy()
    # Preserve the original RF script's second-stage selection rule.
    eligible = eligible.nlargest(50, "sum_Attribution").copy()
    if not eligible.empty:
        values = eligible["sum_Attribution"].fillna(eligible["sum_Attribution"].median())
        amin, amax = values.min(), values.max()
        eligible["norm_sum_Attribution"] = 0.5 if amin == amax else (values - amin) / (amax - amin)
        eligible["tie_score"] = .5 * eligible["mean_pred_prob_dCt_gt_8"] + .5 * eligible["norm_sum_Attribution"]
        eligible = eligible.sort_values(["sum_Freq", "tie_score"], ascending=False)
    return expanded_df, eligible.reset_index(drop=True), eligible.head(top_k).reset_index(drop=True)


def write_xlsx(df: pd.DataFrame, path: Path):
    df.to_excel(path, index=False)

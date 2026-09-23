"""Isolated adapter around the unmodified root training and Captum scripts.

This is deliberately historical behavior, including defects documented in README.
Never run the pandas reader patch / global RNG in a Flask request thread.
"""
from pathlib import Path
from unittest.mock import patch
import hashlib
import io
import json
import os
import sys
import zipfile
import numpy as np
import pandas as pd
import torch

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path[:0] = [str(WEB/'vendor/python'), str(ROOT.parent), str(ROOT)]
import matplotlib
matplotlib.use('Agg')
from gene_with_order.train import process
from gene_with_order.models import GPTConfig
from gene_with_order.load_as_dict import mutation_gene_freq, load_as_dict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(progress, stage, detail):
    print('WEB_PROGRESS '+json.dumps([progress,stage,detail]), flush=True)


def archive_table(name):
    with zipfile.ZipFile(ROOT/'blocker_design(1).zip') as z:
        return pd.read_excel(io.BytesIO(z.read(name)))


def same_cohort(a, b):
    """Order-sensitive: original sample and within-sample ordering affects RNG/tokens."""
    cols=['Sample_ID','Class','mut_pos','VAF']
    a=a[cols].reset_index(drop=True); b=b[cols].reset_index(drop=True)
    for col in cols[:-1]:
        if not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):
            return False
    if not a.VAF.eq('-').equals(b.VAF.eq('-')): return False
    return bool(np.allclose(pd.to_numeric(a.VAF,errors='coerce').to_numpy(float),pd.to_numeric(b.VAF,errors='coerce').to_numpy(float),
                            atol=1e-14,rtol=0,equal_nan=True))


def ids_for(ds, frame, seed):
    first=frame.drop_duplicates('Sample_ID')
    rng=torch.Generator().manual_seed(seed)
    groups=[]
    for label in ['HCC','nonHCC']:
        names=first.loc[first.Class.eq(label),'Sample_ID'].to_numpy(str)
        ix=torch.randperm(len(names),generator=rng).numpy()
        n,nv=int(len(names)*.7),int(len(names)*.1)
        take=ix[:n] if ds.type=='train' else ix[n:n+nv] if ds.type=='val' else ix[n+nv:]
        groups.append(names[take])
    result=np.concatenate(groups+([groups[0]] if ds.type=='train' else []))
    assert len(result)==len(ds)
    truth=first.set_index('Sample_ID').Class
    assert np.array_equal(np.asarray(ds.select_label),[0 if truth[n]=='HCC' else 1 for n in result])
    return result


class Indexed(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds=ds
    def __len__(self): return len(self.ds)
    def __getitem__(self,i): return i,*self.ds[i]


def predict(loader,model,ids):
    indices=[]; labels=[]; scores=[]; logits=[]; batches=[]
    handle=model.fc.register_forward_hook(lambda m,i,o:logits.append(o.detach().numpy().copy()))
    model.eval()
    with torch.no_grad():
        for bid,(ix,g,e,y) in enumerate(torch.utils.data.DataLoader(Indexed(loader.dataset),batch_sampler=loader.batch_sampler)):
            indices.extend(ix.tolist());labels.extend(y.tolist());scores.extend(model(g,e).numpy());batches.extend([bid]*len(ix))
    handle.remove()
    return dict(sample_id=ids[indices],y_true=np.array(labels),y_output=np.array(scores),logits=np.concatenate(logits),batch_id=np.array(batches))


def run(frame, settings, out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    mode=settings.get('analysis_protocol','legacy_retrain')
    if mode=='paper_archive':
        # A reference result, NEVER masquerading as a newly computed explanation.
        reference=pd.read_excel(ROOT/'RFdata_hg38_VAF_Revised.xlsx')
        if not same_cohort(frame,reference):
            raise ValueError('Historical archive replay requires the original RFData rows, labels, mutations and VAF in original order. Use Legacy retraining for new cohorts.')
        attr=archive_table('attribution_frequency.xlsx')
        attr.to_csv(out/'engine_attribution.csv',index=False)
        pd.DataFrame(columns=['Sample_ID','actual_class','predicted_class','probability_positive']).to_csv(out/'engine_diagnosis.csv',index=False)
        metrics=dict(protocol=mode,training_performed=False,attribution_recomputed=False,test_auc=None,
            archived_positive_mutations=int(sum(attr.attribution>0)),
            warnings=['Historical artifact replay only: no training, no newly computed test performance or attribution. Historical model/RNG/data provenance is incomplete.',
                      'Upload matching only confirms the current project RFData reference. It does not establish that the historical attribution table was generated from this exact workbook.'])
        (out/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        (out/'run_manifest.json').write_text(json.dumps(dict(settings=settings,mode=mode,archive_sha256=sha(ROOT/'blocker_design(1).zip'),
            reference_sha256=sha(ROOT/'RFdata_hg38_VAF_Revised.xlsx'),matches_current_project_reference=True,
            historical_dataset_identity_verified=False),indent=2),encoding='utf-8')
        return
    if mode!='legacy_retrain': raise ValueError('Unknown analysis protocol')
    from gene_with_order.explain import Explain
    import captum
    import sklearn
    from sklearn.metrics import roc_auc_score
    positive=str(settings.get('positive_class','HCC')).strip()
    classes=frame.Class.unique().tolist()
    if len(classes)!=2 or positive not in classes:
        raise ValueError('Legacy training requires exactly two classes and an exact positive class label.')
    original_classes=frame.drop_duplicates('Sample_ID').set_index('Sample_ID').Class
    normalized=frame.copy()
    normalized.Class=np.where(normalized.Class.eq(positive),'HCC','nonHCC')
    # Metadata is unused by the original model (apply_ehr=False).
    for c,default in [('Age',0),('AFP',0),('Gender','Male')]:
        if c not in normalized: normalized[c]=default
    normalized.Gender=normalized.Gender.replace({0:'Male',1:'Female'}).fillna('Male')
    normalized=normalized.reset_index(drop=True)
    counts=normalized.drop_duplicates('Sample_ID').Class.value_counts()
    if counts.min()<10: raise ValueError('At least 10 patients per class are required for the original 70/10/20 split.')
    seed=int(settings.get('seed',11)); vocab=int(settings.get('vocab_size',250))
    cfg=GPTConfig(block_size=int(settings.get('block_size',15)),vocab_size=vocab,n_layer=10,n_head=16,n_embd=64,
        dropout=.1,bias=True,cls_num=2,apply_ehr=False,if_vaf_sort=True,if_qpcr=False,if_protein=False)
    reader=pd.read_excel
    def read(path,*args,**kwargs):
        if Path(path).name=='RFdata_hg38_VAF_Revised.xlsx': return normalized.copy()
        return reader(path,*args,**kwargs)
    with patch('pandas.read_excel',side_effect=read):
        genes,_,_,_=load_as_dict(if_vaf_sort=True)
        freqs,ranked=mutation_gene_freq(genes)
        if len(ranked)<vocab:
            raise ValueError(f'Original Explain requires at least vocab_size distinct mutations ({vocab}); this cohort has {len(ranked)}. Reduce vocabulary size explicitly.')
        # Each Dataset resets torch.manual_seed, as in the original constructor.
        runner=process(cfg,seg_seed=seed,lr=float(settings.get('learning_rate',.001)),batch_size=64,save_path=str(out))
        assignments=[]
        for name,loader in [('train',runner.train_dataset),('validation',runner.val_dataset),('test',runner.test_dataset)]:
            ids=ids_for(loader.dataset,normalized,seed)
            assignments.extend(dict(Sample_ID=n,split=name,training_occurrence=i) for i,n in enumerate(ids))
        split=pd.DataFrame(assignments)
        assert split.groupby('Sample_ID').split.nunique().max()==1
        assert split.Sample_ID.nunique()==normalized.Sample_ID.nunique()
        split.to_csv(out/'sample_splits.csv',index=False)
        last_test_state=[]; original_test=runner.test_one_step
        def test(*args,**kwargs):
            last_test_state[:]=[torch.get_rng_state().clone()]
            return original_test(*args,**kwargs)
        runner.test_one_step=test
        original_step=runner.train_one_step; epoch=[0]
        def step():
            value=original_step();epoch[0]+=1
            emit(10+int(30*epoch[0]/int(settings.get('epochs',20))),'Legacy Transformer training',f'Epoch {epoch[0]}: original loss, duplication and call order')
            return value
        runner.train_one_step=step
        runner.train(int(settings.get('epochs',20)))
        runner.save_everything()
        import shutil
        shutil.copy2(out/'model.pth',out/'trained_transformer.pth')
        final_rng=torch.get_rng_state().clone()
        torch.set_rng_state(last_test_state[0])
        z=predict(runner.test_dataset,runner.model,ids_for(runner.test_dataset.dataset,normalized,seed))
        assert torch.equal(torch.get_rng_state(),final_rng)
        np.savez_compressed(out/'test_predictions.npz',**z)
        diag=pd.DataFrame(dict(Sample_ID=z['sample_id'],actual_class=original_classes.loc[z['sample_id']].to_numpy(),
            predicted_class=np.where(z['y_output'].argmax(1)==0,positive,next(c for c in classes if c!=positive)),
            probability_positive=z['y_output'][:,0],logit_positive=z['logits'][:,0],logit_control=z['logits'][:,1],batch_id=z['batch_id']))
        diag.to_csv(out/'engine_diagnosis.csv',index=False)
        history=[item for item in runner.log if isinstance(item,dict)]
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        (out/'gene_vocabulary.json').write_text(json.dumps(runner.test_dataset.dataset.map,indent=2),encoding='utf-8')
        # Instantiate the ACTUAL root Explain, preserving model construction RNG,
        # DataLoader shuffle, Captum permutation and duplicate-index accumulation.
        explainer=Explain(cfg,device='cpu',seg_seed=seed,model_path=str(out/'model.pth'))
        explain_ids=ids_for(explainer.dataset,normalized,seed)
        np.save(out/'explanation_rng_state.npy',torch.get_rng_state().numpy())
        captured={}
        old_draw=explainer.draw_attribution
        def draw(attr,path=None):
            captured['raw']=np.asarray(attr).copy()
            return old_draw(attr,path)
        explainer.draw_attribution=draw
        emit(42,'Legacy Captum explanation','Original test split, shuffled batches of 4, absolute attribution export')
        explainer.explain(path=str(out))
        raw=captured['raw']; values=np.abs(raw[1:])
        # Exact old gene_list naming, including the legacy unknown-token alias.
        explanation=pd.DataFrame({'mutpos':ranked[:len(values)],'attribution':values})
        freq_table=pd.DataFrame({'mutpos':ranked,'freq':freqs})
        attr=freq_table.merge(explanation,on='mutpos',how='left',validate='one_to_one').fillna({'attribution':0})
        attr=attr[['mutpos','attribution','freq']]
        attr=attr.sort_values('attribution',ascending=False,kind='stable')
        attr.to_csv(out/'engine_attribution.csv',index=False)
        metrics=dict(protocol=mode,training_performed=True,attribution_recomputed=True,
            positive_class=positive,train_samples=split[split.split=='train'].Sample_ID.nunique(),
            train_rows_after_duplication=len(runner.train_dataset.dataset),val_samples=len(runner.val_dataset.dataset),test_samples=len(z['y_true']),
            vocabulary_size=len(runner.test_dataset.dataset.map),
            test_auc=float(roc_auc_score(np.eye(2)[z['y_true']].ravel(),z['y_output'].ravel())),
            patient_hcc_auc=float(roc_auc_score(z['y_true']==0,z['y_output'][:,0])),
            test_accuracy=float(np.mean(z['y_output'].argmax(1)==z['y_true'])),
            warnings=['Historical semantics retained: cross-patient attention, full-cohort vocabulary, positive-class duplication, CE on probabilities, state_dict reference checkpoint, repeated test evaluation, duplicate-index attribution accumulation and absolute attribution.',
                      'Test data is used for panel discovery; it is not an independent final panel validation set.',
                      'New training follows old scripts but is not a replay of undocumented historical RNG/model states. No fixed 74-mutation result is promised.'])
        (out/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        manifest=dict(settings=settings,config=vars(cfg),class_encoding={positive:0,next(c for c in classes if c!=positive):1},
            sources={name:sha(ROOT/name) for name in ['train.py','dataset_qPCR.py','load_as_dict.py','models.py','models_for_explain.py','explain.py']},
            model_sha256=sha(out/'model.pth'),torch_version=torch.__version__,numpy_version=np.__version__,
            captum_version=captum.__version__,sklearn_version=sklearn.__version__,python_version=sys.version,
            legacy_dash_vaf_rows=int(frame.VAF.eq('-').sum()),
            batch_size=64,explanation_batch_size=4,explanation_dataset_ids=explain_ids.tolist(),
            unknown_token_legacy_attribution_alias=ranked[vocab-2],
            frequency='all raw mutation occurrences, original mutation_gene_freq',
            score='not reconstructed: downstream scripts use attribution and frequency, not the archived display score')
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    from pipeline import read_table,validate_database
    folder=Path(sys.argv[1]).resolve()
    settings=json.loads((folder/'settings.json').read_text(encoding='utf-8'))
    frame=validate_database(pd.read_pickle(folder/'engine_input.pkl'))
    run(frame,settings,folder)

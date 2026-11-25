from pathlib import Path
import importlib.util


def test_explicit_final_labels_and_fourarm_projection():
    p=Path(__file__).resolve().parents[1]/'analysis/final_paper_tables.py'
    assert p.is_file(), 'Missing stable-label final table export'
    spec=importlib.util.spec_from_file_location('final_tables_test',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for dataset in ['c10','c100','im16']:
        for metric in ['RMSE','MAE','Pearson','Spearman','R2']:
            for a,b,va,vb in [('A','A_T',4.,3.),('A','A_LC',4.,2.),('A_LC','A_LC_T',2.,1.)]:
                rows.append(dict(dataset=dataset,population='target_ge20',contrast=a+' -> '+b,
                                 metric=metric,reference=va,candidate=vb,n=10,groups=4))
    result=module.fourarm(rows)
    assert len(result)==60
    assert {r['arm'] for r in result}=={'A','A_T','A_LC','A_LC_T'}
    assert all(r['population']=='target_ge20' for r in result)
    assert module.LEGACY_LABELS['tab:sm-checkpoints']=='S15'

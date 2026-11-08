"""Read-only consolidated Point 4 report from finished per-dataset analyses."""
import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                    help='Exported Point 4 package with manifest and completed analyses')
P=parser.parse_args().root.resolve()
manifest=json.loads((P/'manifest.json').read_text())
expected={'cifar-10':'lemur-robustness-all3-v1-cifar10',
          'cifar-100':'lemur-robustness-all3-v1-cifar100',
          'imagenette':'lemur-robustness-all3-v1-imagenette'}
bundles=manifest['bundles']
if len(bundles)!=3 or {b['dataset']:b['name'] for b in bundles}!=expected:
    raise RuntimeError('Expected the three distinct Point 4 dataset bundles')
# Validate every analysis before printing any results. Receipts establish export
# integrity, not independent proof that the historical experiment ran.
for bundle in bundles:
    a=(P/bundle['name']/'results/analysis').resolve()
    if not a.is_relative_to(P):
        raise RuntimeError('Analysis path escapes exported package')
    receipt=json.loads((a/'complete.json').read_text())
    hashes=receipt.get('files_sha256',{})
    required={'study_report.json','measurement_summary.csv','rank_agreement.json'}
    if receipt.get('status')!='complete' or not required.issubset(hashes):
        raise RuntimeError('Incomplete analysis receipt: '+bundle['dataset'])
    for name,digest in hashes.items():
        path=(a/name).resolve()
        if not path.is_relative_to(a) or not path.is_file():
            raise RuntimeError('Missing or unsafe analysis input: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Changed analysis input: '+name)
for bundle in manifest['bundles']:
    dataset=bundle['dataset']; a=P/bundle['name']/'results/analysis'
    print('\n=== '+dataset+' ===')
    if not (a/'complete.json').is_file():
        print('NOT FINISHED: outputs preserved; no rerun requested')
        continue
    report=json.loads((a/'study_report.json').read_text())
    print('Candidates:',report['completed_architectures'],'/',report['selected_architectures'],
          '| measurement failures:',report['measurement_failures'],
          '| scaling checks:',report['scaling_checks'],
          '| disagreements:',report['scaling_disagreements'])
    with (a/'measurement_summary.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    pairs=defaultdict(dict)
    for r in rows:
        key=(r['candidate'],int(r['epoch']),int(r['batch_size']),r['mode'],r['augmentation'])
        if r['loss_reduction'] in pairs[key]: raise RuntimeError('Ambiguous coverage '+str(key))
        pairs[key][r['loss_reduction']]=r
    groups=defaultdict(list)
    for key,pair in pairs.items():
        if set(pair)!={'mean','sum'}: continue
        m,s=pair['mean'],pair['sum']
        if any(r['candidate_status']!='complete' or int(r['n_successful_measurements'])!=6 or int(r['coverage_variants_at_checkpoint'])!=1 for r in (m,s)): continue
        if m['coverage_id']!=s['coverage_id']: continue
        try: values=[float(m['raw_mean']),float(m['rescaled_mean']),float(s['raw_mean']),100*float(m['raw_cv'])]
        except (ValueError,TypeError): continue
        if all(math.isfinite(x) for x in values): groups[key[1:]].append(values)
    print('epoch batch mode aug n raw_mean_loss rescaled_BxT sum_loss median_CV_percent')
    for key,values in sorted(groups.items()):
        print(*key,len(values),*(f'{median(column):.6g}' for column in zip(*values)))
    print('Scores are medians of candidate means; CV is median within-candidate repeat CV.')
    print('Rescaling leaves CV unchanged; candidate subsets can differ between rows.')
    rank=json.loads((a/'rank_agreement.json').read_text())
    print('RANK AGREEMENT: epoch batch mode aug matched_n spearman (mean loss)')
    for r in rank:
        if r['loss_reduction']=='mean':
            print(r['epoch'],r['batch_size'],r['mode'],r['augmentation'],r['n_matched_architectures'],r['spearman'])

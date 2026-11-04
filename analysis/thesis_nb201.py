"""Recompute NB201 thesis tables from frozen prediction exports only.

No predictor fitting. The primary endpoint's crossed statistical mixed model is
refit as specified by its protocol. Imports do not alter sys.path/sys.modules.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import tempfile


def _primary(repo):
    spec = importlib.util.spec_from_file_location('_thesis_nb201_primary', Path(repo) / 'nb201_v1/primary_endpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_mae(repo):
    # evaluate.py imports legacy `panel` with a global sys.path mutation. Execute
    # its exact self-contained function AST instead, without importing training.
    import numpy as np
    path = Path(repo) / 'nb201_v1/evaluate.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'paired_mae')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['paired_mae']


def _differences(actual, expected, path='', rtol=1e-9, atol=1e-11):
    """Return explicit mismatches, never turn numerical failures into passes."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path]
        return [p for k, v in expected.items() for p in
                (_differences(actual[k], v, f'{path}.{k}', rtol, atol) if k in actual else [f'{path}.{k}: missing'])]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path]
        return [p for i, v in enumerate(expected) for p in _differences(actual[i], v, f'{path}[{i}]', rtol, atol)]
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return [] if isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol) else [path]
    return [] if actual == expected else [path]


def _hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _check_sources(before):
    if before != _hashes([Path(p) for p in before]):
        raise RuntimeError('NB201 source inputs changed during reproduction')


def _new_root(work, prefix):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=work))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def build_classical_tables(repo: Path, work: Path):
    """Return eight all-record metric rows and four Bonferroni8 rows."""
    repo = Path(repo)
    primary, paired = _primary(repo), _paired_mae(repo)
    dest = _new_root(work, 'nb201-classical-')
    metrics, rows = [], []
    for dataset in ('cifar10', 'cifar100'):
        for model in ('ridge', 'gbm'):
            source = repo / 'evidence/nb201-classical-matched-v1' / dataset / model
            before = _hashes([source / name for name in ('panel.csv', 'test_registry.json', 'results.json')])
            y, pred, registry = primary.load_panel(source / 'panel.csv', source / 'test_registry.json')
            # Tiny bootstrap is unused; descriptive metrics are exact six-seed means.
            descriptive, _ = primary.analyze(y, pred, registry, draws=100, seed=0, mixed=False)
            saved = json.loads((source / 'results.json').read_text())
            report = {}
            for population, mask in (('all', y == y), ('target_ge20', y >= 20)):
                result, _ = paired(y[mask], pred[0, mask].T, pred[1, mask].T,
                                   [r['family'] for r, keep in zip(registry, mask) if keep], draws=10000)
                desc = descriptive['descriptive']['all' if population == 'all' else 'noncollapsed']
                result['mean_metrics'] = {c: desc[c]['mean'] for c in ('no-layer', 'taylor')}
                expected = {k: saved[population][k] for k in result}
                failures = _differences(result, expected)
                result['verification_status'] = 'mismatch' if failures else 'verified'
                result['verification_failures'] = failures
                report[population] = result
            path = dest / f'{dataset}-{model}.json'
            _check_sources(before)
            _write(path, dict(results=report, input_sha256=before))
            result = report['all']
            common = dict(dataset=dataset, model=model, population='all', n=len(y),
                          report_path=str(path), verification_status=result['verification_status'])
            for condition in ('no-layer', 'taylor'):
                metrics.append(dict(common, condition=condition, **result['mean_metrics'][condition]))
            row = dict(common, families=result['families'], improvement=result['improvement'])
            for key in ('record_ci95', 'family_ci95', 'record_bonferroni8', 'family_bonferroni8'):
                row[key + '_low'], row[key + '_high'] = result[key]
            rows.append(row)
    return metrics, rows


def primary_rows(dataset, report, saved, report_path):
    """Flat table projections, with separately labelled numerical verification."""
    sections = ('counts', 'primary', 'descriptive', 'collapse_auc', 'seed_improvements', 'robust_location')
    failures = [p for key in sections for p in _differences(report[key], saved[key], key)]
    robustness = report['robustness']
    for key in saved['robustness']:
        if key != 'crossed_mixed_model':
            failures.extend(_differences(robustness.get(key), saved['robustness'][key], key))
    mixed = robustness['crossed_mixed_model']
    old_mixed = saved['robustness']['crossed_mixed_model']
    # Platform/BLAS dependent numerical fit; never overwrite recomputed outputs.
    mixed_keys = ('status', 'converged', 'improvement', 'p', 'ci95')
    mixed_failures = _differences({k: mixed.get(k) for k in mixed_keys},
                                  {k: old_mixed.get(k) for k in mixed_keys},
                                  'crossed_mixed_model', rtol=1e-4, atol=1e-6)
    mixed_status = ('unverified fit' if mixed.get('status') != 'ok' else
                    'mismatch' if mixed_failures else 'verified')
    verification = dict(core_status='mismatch' if failures else 'verified',
                        core_failures=failures, mixed_status=mixed_status,
                        mixed_failures=mixed_failures, core_rtol=1e-9, core_atol=1e-11,
                        mixed_rtol=1e-4, mixed_atol=1e-6)
    common = dict(dataset=dataset, model='qwen', report_path=str(report_path),
                  verification_status=verification['core_status'], mixed_verification_status=mixed_status)
    metrics = [dict(common, condition=c, population='all', n=report['counts']['all'],
                    **report['descriptive']['all'][c]['mean']) for c in ('no-layer', 'taylor')]
    row = dict(common, population='noncollapsed', **report['counts'])
    for unit in ('record', 'seed'):
        test = report['primary'][unit]
        for key in ('n', 'improvement', 't', 'p', 'holm_p', 'status'):
            row[f'{unit}_{key}'] = test.get(key)
        ci = test.get('ci95') or [None, None]
        row[f'{unit}_ci95_low'], row[f'{unit}_ci95_high'] = ci
    for condition in ('no-layer', 'taylor'):
        row[condition + '_MAE'] = report['descriptive']['noncollapsed'][condition]['mean']['MAE']
        auc = report['collapse_auc'][condition]
        row[condition + '_collapse_auc'] = auc['auc']
        ci = auc.get('ci95') or [None, None]
        row[condition + '_collapse_auc_low'], row[condition + '_collapse_auc_high'] = ci
    row['multiplicity'] = 'Holm2 within dataset: record and seed tests'
    appendix = []
    if dataset in ('cifar10', 'cifar100'):
        sensitivity = dict(common, population='noncollapsed', n=report['counts']['noncollapsed'],
                           improvement=report['primary']['record']['improvement'])
        for name in ('record_bootstrap', 'two_way_bootstrap', 'family_bootstrap', 'wilcoxon', 'cell_signflip', 'crossed_mixed_model'):
            if name not in robustness:
                continue
            test = robustness[name]
            prefix = 'mixed' if name == 'crossed_mixed_model' else name
            sensitivity[prefix + '_p'] = test.get('p')
            ci = test.get('ci95') or [None, None]
            sensitivity[prefix + '_ci95_low'], sensitivity[prefix + '_ci95_high'] = ci
            if 'status' in test:
                sensitivity[prefix + '_status'] = test['status']
        sensitivity['mixed_converged'] = mixed.get('converged')
        sensitivity['mixed_optimizer'] = mixed.get('optimizer')
        sensitivity['multiplicity'] = 'Supporting unadjusted sensitivities'
        appendix.append(sensitivity)
    return metrics, [row], appendix, verification


def build_tables(repo: Path, work: Path) -> dict:
    """Build 6.14/6.15/6.16/A.5, saving new reports beneath work.

    Source reports are comparison targets only: every reported statistic is
    calculated from saved prediction panels. Existing reports are never replaced.
    """
    repo, work = Path(repo).resolve(), Path(work).resolve()
    evidence = repo / 'evidence'
    if work == evidence or evidence in work.parents:
        raise ValueError('Work must not be inside canonical evidence')
    primary = _primary(repo)
    root = _new_root(work, 'nb201-primary-')
    tables = {'6.14': [], '6.15': [], '6.16': [], 'A.5': []}
    manifest = {}
    for dataset in ('cifar10', 'cifar100', 'imagenet16'):
        source = (evidence / 'nb201-imagenet16-primary-v1' if dataset == 'imagenet16'
                  else evidence / 'nb201-qwen-primary-v1' / dataset)
        panel, registry = source / 'inputs/panel.csv', source / 'inputs/registry.json'
        reference = source / 'analysis/results.json'
        before = _hashes([panel, registry, reference])
        out = root / dataset
        report = primary.run_panel(panel, registry, out, draws=10000, seed=0, mixed=True)
        saved = json.loads(reference.read_text())
        metrics, tests, appendix, verification = primary_rows(dataset, report, saved, out / 'results.json')
        _check_sources(before)
        _write(out / 'verification.json', dict(verification, source_sha256=before))
        manifest[dataset] = dict(verification, report_path=str(out / 'results.json'))
        tables['6.14'].extend(metrics)
        tables['6.15'].extend(tests)
        tables['A.5'].extend(appendix)
    metrics, paired = build_classical_tables(repo, work)
    tables['6.14'].extend(metrics)
    tables['6.16'].extend(paired)
    _write(root / 'verification.json', manifest)
    _write(root / 'tables.json', tables)
    return tables

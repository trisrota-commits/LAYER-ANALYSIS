"""Runner adapter for unmodified frozen-prediction standalone calculations."""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

PACKET = 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip'
SCRIPTS = ('reproduce_main_paired_tables.py', 'reproduce_lodo_tables.py', 'reproduce_lofo_tables.py')
INPUTS = (PACKET, *('evidence/' + name for name in SCRIPTS))
MAIN_ARCHIVE = 'article-tables-only-20261014-111938.tar.gz'
HOLDOUT_ARCHIVE = 'holdout-reproduction-inputs-728444c5.zip'
REPORTS = ('paired_main_tables.log', 'lodo_tables_reproduced.log', 'lofo_tables_reproduced.log')
STAGED = (MAIN_ARCHIVE, HOLDOUT_ARCHIVE, 'experiment_logs/paired_holdout_tables.log',
          'experiment_logs/lodo_descriptive_table_evidence.log',
          *('experiment_logs/' + name for name in REPORTS))

NUMBER = r'([-+0-9.eE]+)'
METRIC = r'(RMSE|MAE|Pearson|Spearman|R2)'
MAIN_TABLES = {'tab:taylor-paired-bootstrap-rmse': 'S21',
               'tab:readmission-paired-five-metrics': 'S23',
               'tab:zcp-paired-five-metrics': 'S19',
               'tab:per-layer-versus-averaged-taylor': 'S40'}


def parse_main(text):
    tables = {name: [] for name in MAIN_TABLES.values()}
    table = None
    candidate = reference = ''
    for line in text.splitlines():
        if line.startswith('tab:'):
            table = MAIN_TABLES[line.split()[0]]
        elif line.startswith('candidate='):
            candidate, reference = line[10:].split('; reference=')
        elif table and ': delta=' in line:
            match = re.fullmatch(r'\s+' + METRIC + ': delta=' + NUMBER + r' CI95=\[' + NUMBER + r', ' + NUMBER + r'\] finite_draws=(\d+)/(\d+)', line)
            if not match:
                raise ValueError('Unparsed paired row: ' + line)
            metric, delta, low, high, count, total = match.groups()
            tables[table].append(dict(candidate=candidate, reference=reference, metric=metric,
                                      improvement=float(delta), low=float(low), high=float(high),
                                      valid_draws=int(count), draws=int(total)))
    return tables


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_reports(main, lodo, lofo):
    """Preserve report precision; never reround endpoints or infer finite counts."""
    tables = parse_main(main)
    desc = []
    condition = None
    active = False
    for line in main.splitlines():
        if line.startswith('DESCRIPTIVE SIX-SEED'):
            active = True
        elif line.startswith('tab:'):
            active = False
        elif active and line and not line.startswith(' '):
            condition = line
        elif active and ': ' in line:
            m = re.fullmatch(r'\s+' + METRIC + ': ' + NUMBER + r' \[' + NUMBER + r', ' + NUMBER + r'\] SD=' + NUMBER, line)
            require(m is not None, 'Unparsed main descriptive: ' + line)
            metric, mean, low, high, sd = m.groups()
            desc.append(dict(condition=condition, metric=metric, mean=float(mean), low=float(low), high=float(high), sd=float(sd)))
    tables['main_descriptive'] = desc
    means = {(r['condition'], r['metric']): r['mean'] for r in desc}
    for row in tables['S40']:
        row['candidate_mean'] = means[row['candidate'], row['metric']]
        row['reference_mean'] = means[row['reference'], row['metric']]
    tables['S22'] = [r for r in desc if r['condition'] in ('T OFF', 'T+S', 'T+G', 'T+E', 'T+S+G', 'T+S+E')]
    tables['S36'], tables['S37'] = [], []
    folds = {}
    for line in lodo.splitlines():
        if line.startswith('FOLD_PASS '):
            m = re.match(r'FOLD_PASS (\S+) records=(\d+) architecture_groups=(\d+)', line)
            require(m is not None, 'Malformed LODO fold')
            fold, n, groups = m.groups()
            folds[fold] = dict(records=int(n), architecture_groups=int(groups))
        elif line.startswith('DESCRIPTIVE,'):
            _, fold, arm, metric, mean, low, high = line.split(',')
            tables['S36'].append(dict(dataset=fold, condition=arm, metric=metric, mean=float(mean), low=float(low), high=float(high), **folds[fold]))
        elif line.startswith('BOOTSTRAP,'):
            _, fold, n, groups, metric, delta, low, high, valid = line.split(',')
            tables['S37'].append(dict(dataset=fold, candidate='taylor', reference='no-layer', records=int(n), architecture_groups=int(groups), metric=metric,
                                     improvement=float(delta), low=float(low), high=float(high), valid_draws=int(valid), draws=10000))
    tables['lofo_all_descriptive'], tables['S39'] = [], []
    folds = {}
    for line in lofo.splitlines():
        if line.startswith('FOLD '):
            m = re.match(r'FOLD (\S+) records=(\d+) groups=(\d+)', line)
            require(m is not None, 'Malformed LOFO fold')
            fold, n, groups = m.groups()
            folds[fold] = dict(records=int(n), architecture_groups=int(groups))
        elif line.startswith('DESCRIPTIVE '):
            _, fold, arm, rest = line.split(' ', 3)
            matches = list(re.finditer(METRIC + '=' + NUMBER + r'\[' + NUMBER + ',' + NUMBER + r'\]', rest))
            require(len(matches) == 5, 'Incomplete LOFO descriptive row')
            for m in matches:
                metric, mean, low, high = m.groups()
                tables['lofo_all_descriptive'].append(dict(family=fold, condition=arm, metric=metric, mean=float(mean), low=float(low), high=float(high), **folds[fold]))
        elif line.startswith('BOOTSTRAP '):
            m = re.match(r'BOOTSTRAP (\S+) ' + METRIC + ' delta=' + NUMBER + r' CI=\[' + NUMBER + ',' + NUMBER + r'\] valid=(\d+)/(\d+) reference=', line)
            require(m is not None, 'Unparsed LOFO bootstrap: ' + line)
            fold, metric, delta, low, high, valid, total = m.groups()
            tables['S39'].append(dict(family=fold, candidate='TaylorOFF', reference='no-layerON', metric=metric,
                                     improvement=float(delta), low=float(low), high=float(high), valid_draws=int(valid), draws=int(total), **folds[fold]))
    tables['S38'] = [r for r in tables['lofo_all_descriptive'] if r['condition'] in ('no-layerON', 'TaylorOFF')]
    expected = {'S19': 10, 'S21': 35, 'S23': 20, 'S36': 70, 'S37': 35,
                'S38': 50, 'S39': 25, 'S40': 5, 'S22': 30, 'main_descriptive': 65, 'lofo_all_descriptive': 75}
    for name, count in expected.items():
        require(len(tables[name]) == count, f'Incomplete {name}: {len(tables[name])} != {count}')
        keys = [tuple(r.get(k) for k in ('dataset', 'family', 'condition', 'candidate', 'reference', 'metric')) for r in tables[name]]
        require(len(set(keys)) == count, 'Duplicate rows in ' + name)
    require('MISMATCH' not in main + lodo + lofo, 'Standalone reference mismatch')
    require('"mismatches": []' in lodo, 'Missing LODO successful summary')
    require('matches=25 mismatches=0' in lofo, 'Missing LOFO successful comparison')
    return tables


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stage_packet(archive, destination):
    """Authenticate every nested member, stage only allowlisted regular files.

    No tar or inner ZIP is extracted: the original programs read them in memory.
    Reject Windows aliases as well as POSIX traversal before creating anything.
    """
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        normalized = set()
        for info in z.infolist():
            name = info.orig_filename
            path = PurePosixPath(name)
            require(not path.is_absolute() and bool(path.parts) and '..' not in path.parts
                    and ':' not in name and '\\' not in name
                    and all(p not in ('.', '') and p == p.rstrip(' .') for p in name.rstrip('/').split('/')),
                    'Unsafe archive member: ' + name)
            require((info.external_attr >> 16) & 0o170000 not in (0o120000, 0o060000, 0o020000, 0o010000),
                    'Unsafe archive link/device: ' + name)
            key = str(path).casefold()
            require(key not in normalized, 'Unsafe duplicate archive member: ' + name)
            normalized.add(key)
        manifest_name = 'reproduction_packet_manifest.json'
        manifest = json.loads(z.read(manifest_name))
        entries = manifest['files']
        # Published README and manifest are authenticated by the outer SHA256;
        # the README intentionally has no self-entry in the packet manifest.
        require(set(names) == set(entries) | {manifest_name, 'README.txt'}, 'Packet manifest membership mismatch')
        for name, entry in entries.items():
            with z.open(name) as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            require(digest == entry['sha256'] and z.getinfo(name).file_size == entry['bytes'],
                    'Packet member SHA256/size mismatch: ' + name)
        require(set(STAGED) <= set(entries), 'Missing required packet members')
        destination.mkdir(parents=True, exist_ok=False)
        for name in (*STAGED, manifest_name):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(z.read(name))
    return manifest


def compare_tables(actual, expected):
    """Compare every parsed field with authenticated historical reports."""
    require(actual.keys() == expected.keys(), 'Reference table coverage mismatch')
    for table, rows in actual.items():
        require(len(rows) == len(expected[table]), 'Reference count mismatch: ' + table)
        for index, (row, old) in enumerate(zip(rows, expected[table])):
            require(row.keys() == old.keys(), 'Reference fields mismatch: ' + table)
            for key, value in row.items():
                equal = math.isclose(value, old[key], rel_tol=0, abs_tol=1e-10) if isinstance(value, float) else value == old[key]
                require(equal, f'Reference mismatch: {table} row={index} field={key}: {value} != {old[key]}')


def run(repo, out):
    """Recompute all standalone tables into a NEW directory and return results.

    Uses the calling Python environment, which must provide the repository's
    pinned NumPy/SciPy requirements. Executes only the three hash-pinned scripts;
    their RNG, grouping, ranking and finite-draw algorithms remain unmodified.
    No fitting, model loading, training, inference, or stale-cohort substitution.
    """
    repo, out = Path(repo).resolve(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Existing output preserved: ' + str(out))
    pinned = json.loads((repo / 'INPUT_MANIFEST.json').read_text(encoding='utf-8'))['files']
    hashes = {name: pinned[name] for name in INPUTS}
    for name, digest in hashes.items():
        require(sha(repo / name) == digest, 'Input SHA256 mismatch: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    stage = out / 'frozen_inputs'
    manifest = stage_packet(repo / PACKET, stage)
    for script in SCRIPTS:
        require(hashes['evidence/' + script] == manifest['files'][script]['sha256'], 'Script differs from published packet: ' + script)
    paired_ref = stage / 'experiment_logs/paired_holdout_tables.log'
    arguments = [
        ['--archive', stage / MAIN_ARCHIVE],
        ['--archive', stage / HOLDOUT_ARCHIVE, '--paired-reference', paired_ref,
         '--descriptive-reference', stage / 'experiment_logs/lodo_descriptive_table_evidence.log'],
        ['--archive', stage / HOLDOUT_ARCHIVE, '--reference', paired_ref],
    ]
    commands = []
    for script, report, args in zip(SCRIPTS, REPORTS, arguments):
        # Isolated mode ignores inherited PYTHONOPTIMIZE/PYTHONPATH, preserving
        # the standalone assertions and preventing import shadowing from cwd.
        command = [sys.executable, '-I', '-B', str(repo / 'evidence' / script),
                   *map(str, args), '--output', str(out / report)]
        commands.append(command)
        with (out / (script + '.console.log')).open('x', encoding='utf-8') as handle:
            completed = subprocess.run(command, cwd=out, stdout=handle, stderr=subprocess.STDOUT, text=True)
        require(completed.returncode == 0, 'Standalone failed; inspect ' + script + '.console.log')
    tables = parse_reports(*((out / name).read_text(encoding='utf-8-sig') for name in REPORTS))
    original = parse_reports(*((stage / 'experiment_logs' / name).read_text(encoding='utf-8-sig') for name in REPORTS))
    compare_tables(tables, original)
    for name, digest in hashes.items():
        require(sha(repo / name) == digest, 'Input SHA256 changed during run: ' + name)
    for name in STAGED:
        require(sha(stage / name) == manifest['files'][name]['sha256'], 'Staged input changed: ' + name)
    result = dict(tables=tables, input_sha256=hashes,
                  adapter_sha256=sha(Path(__file__)), commands=commands,
                  reference_comparison='PASS', reference_float_tolerance=1e-10,
                  notes=['Exact standalone algorithms; fitted models held fixed; no seed resampling.',
                         'S19=ZCP, S21=feature pairs/triples, S23=readmission, S40=individual versus averaged Taylor.',
                         'All original report precision retained; LODO bootstrap source emits six decimal places.',
                         'S40 includes both descriptive means, not only differences.',
                         'LOFO original seeds 42-44 lack some receipts; source reports document this limitation.',
                         'main_descriptive preserves all 65 available standalone cells; S17 no-layer is outside this source.',
                         'All per-seed values, provenance and methodological details remain in unmodified reports.'])
    for table, rows in tables.items():
        with (out / (table + '.csv')).open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out / 'results.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    receipt = dict(status='complete', table_rows={t: len(rows) for t, rows in tables.items()},
                   reference_comparison='PASS', files_sha256={p.relative_to(out).as_posix(): sha(p)
                   for p in sorted(out.rglob('*')) if p.is_file()})
    with (out / 'complete.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('COMPLETE paired/holdout tables:', receipt['table_rows'], str(out), flush=True)
    return result

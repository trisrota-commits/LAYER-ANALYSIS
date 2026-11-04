"""Run existing paired/grouped analysis programs and expose their fresh tables."""
import re
from analysis.thesis_common import require, run_script

NUMBER = r'([-+0-9.eE]+)'


def parse_main(text):
    result = {'A.2': [], 'A.3': [], '6.8': []}
    names = {'tab:taylor-paired-bootstrap-rmse': 'A.2', 'tab:zcp-paired-five-metrics': 'A.3',
             'tab:per-layer-versus-averaged-taylor': '6.8'}
    table = None
    candidate = reference = ''
    for line in text.splitlines():
        if line.startswith('tab:'):
            table = names.get(line.split(' ')[0])
        elif line.startswith('candidate='):
            candidate, reference = line.removeprefix('candidate=').split('; reference=')
        elif table:
            match = re.match(r'\s+(RMSE|MAE|Pearson|Spearman|R2): delta=' + NUMBER + r' CI95=\[' + NUMBER + r', ' + NUMBER + r'\] finite_draws=(\d+)/(\d+)', line)
            if match:
                metric, delta, low, high, count, total = match.groups()
                result[table].append(dict(candidate=candidate, reference=reference, metric=metric,
                                          improvement=float(delta), low=float(low), high=float(high),
                                          valid_draws=int(count), draws=int(total)))
    require([len(result[t]) for t in ['A.2', 'A.3', '6.8']] == [35, 10, 5], 'Incomplete main paired report')
    return result


def parse_lodo(text):
    desc, paired = [], []
    for line in text.splitlines():
        if line.startswith('DESCRIPTIVE,'):
            _, fold, arm, metric, mean, low, high = line.split(',')
            desc.append(dict(dataset=fold, condition=arm, metric=metric, mean=float(mean), low=float(low), high=float(high)))
        elif line.startswith('BOOTSTRAP,'):
            _, fold, n, groups, metric, delta, low, high, valid = line.split(',')
            paired.append(dict(dataset=fold, records=int(n), architecture_groups=int(groups), metric=metric,
                               improvement=float(delta), low=float(low), high=float(high), valid_draws=int(valid)))
    require(len(desc) == 70 and len(paired) == 35, 'Incomplete LODO output')
    require('"mismatches": []' in text, 'LODO reproduction differs from archived reference')
    return desc, paired


def parse_lofo(text):
    result = []
    for line in text.splitlines():
        if line.startswith('DESCRIPTIVE '):
            _, fold, arm, rest = line.split(' ', 3)
            if arm not in ['no-layerON', 'TaylorOFF']:
                continue
            for m in re.finditer(r'(RMSE|MAE|Pearson|Spearman|R2)=' + NUMBER + r'\[' + NUMBER + ',' + NUMBER + r'\]', rest):
                metric, mean, low, high = m.groups()
                result.append(dict(family=fold, condition=arm, metric=metric, mean=float(mean), low=float(low), high=float(high)))
    require(len(result) == 50, 'Incomplete LOFO output')
    return result


def build_tables(repo, packet, work):
    work.mkdir(parents=True, exist_ok=False)
    main = work / 'main_paired.log'
    run_script(repo / 'evidence/reproduce_main_paired_tables.py',
               ['--archive', packet / 'article-tables-only-20261014-111938.tar.gz', '--output', main], work / 'main_console.log')
    tables = parse_main(main.read_text(encoding='utf-8'))
    lodo = work / 'lodo.log'
    run_script(repo / 'evidence/reproduce_lodo_tables.py',
               ['--archive', packet / 'holdout-reproduction-inputs-728444c5.zip', '--output', lodo,
                '--paired-reference', packet / 'experiment_logs/paired_holdout_tables.log',
                '--descriptive-reference', packet / 'experiment_logs/lodo_descriptive_table_evidence.log'], work / 'lodo_console.log')
    tables['6.10'], tables['A.4'] = parse_lodo(lodo.read_text(encoding='utf-8'))
    lofo = work / 'lofo.log'
    run_script(repo / 'evidence/reproduce_lofo_tables.py',
               ['--archive', packet / 'holdout-reproduction-inputs-728444c5.zip', '--output', lofo,
                '--reference', packet / 'experiment_logs/paired_holdout_tables.log'], work / 'lofo_console.log')
    tables['6.11'] = parse_lofo(lofo.read_text(encoding='utf-8'))
    return tables

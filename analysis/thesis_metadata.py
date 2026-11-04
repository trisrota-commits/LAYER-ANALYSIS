"""Corpus counts and explicitly documented design tables; no database access."""
from collections import Counter
import json
import zipfile


def build_tables(repo):
    with zipfile.ZipFile(repo / 'data.zip') as archive:
        raw = [json.loads(line) for line in archive.read('3acclayer/llm_finetuning_data_1.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
    if len(raw) != 1151 or len({r['id'] for r in raw}) != 1151:
        raise ValueError('Frozen snapshot count/identity changed')
    counts = Counter(r['dataset'] for r in raw if r['task'] == 'img-classification')
    if len(counts) != 7 or sum(counts.values()) != 1139:
        raise ValueError('Classification population changed')
    corpus = [dict(dataset=name, records=n) for name, n in sorted(counts.items())]
    corpus.append(dict(dataset='Total', records=sum(counts.values())))
    targets = [
        dict(dataset='CIFAR-10-valid', early_accuracies='Benchmark seed-averaged validation curve, epochs 1-3', target='Maximum of seed-averaged validation curve'),
        dict(dataset='CIFAR-100', early_accuracies='Local validation accuracies from Taylor acquisition run', target='Mean epoch-200 validation accuracy, seeds 777 and 888'),
        dict(dataset='ImageNet16-120', early_accuracies='Local validation accuracies from Taylor acquisition run', target='Mean epoch-200 validation accuracy, seeds 777 and 888'),
    ]
    definitions = [
        ('WeightWatcher exponent', 'ww_alpha', 'Weights'),
        ('CKA redundancy', 'cka_redund', 'Activations'),
        ('Rank ratio', 'rank_ratio', 'Weights'),
        ('Taylor importance', 'taylor_imp', 'Activations and their gradients'),
        ('Weight-gradient norm', 'grad_norm', 'Weight gradients'),
        ('Near-zero activation fraction', 'dead_frac', 'Activations'),
        ('Sensitivity', 'sensitivity', 'Loss after zeroing a layer'),
    ]
    source = (repo / 'LayerAnalysis.py').read_text(encoding='utf-8')
    if any(field not in source for _, field, _ in definitions):
        raise ValueError('Layer field declaration absent from supplied source')
    inputs = [
        ('No layer', 'None', 'OFF'),
        ('Taylor epochs 1-3', 'One layer-averaged value at each of epochs 1, 2, 3', 'OFF'),
        ('Taylor epoch 1', 'One layer-averaged value at epoch 1 only', 'OFF'),
        ('Taylor with ZCPs', 'One layer-averaged value at each of epochs 1, 2, 3', 'ON'),
        ('Individual-layer Taylor', 'Individual-layer values at epochs 1, 2, 3', 'OFF'),
    ]
    return {'4.1': corpus, '4.2': targets,
            '4.3': [dict(statistic=n, field=f, computed_from=c) for n, f, c in definitions],
            '4.4': [dict(condition=c, taylor_information=t, zcp=z) for c, t, z in inputs]}

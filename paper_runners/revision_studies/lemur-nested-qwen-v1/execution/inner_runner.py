from dataclasses import fields
import gc
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
EPOCHS = 7

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    (mode, contract_path, key) = sys.argv[1:]
    assert mode in ('train', 'test')
    contract = json.loads(Path(contract_path).read_text())
    assert contract['epochs'] == EPOCHS
    run = contract['runs'][key]
    root = Path(run['data'])
    out = Path(run['output'])
    base = Path(contract['base'])
    selected = Path(run['selected'])
    model_path = out / 'tuned_model'
    for (path, digest) in contract['source_sha256'].items():
        assert sha(Path(path)) == digest, f'Source changed: {path}'
    for (name, digest) in run['input_sha256'].items():
        assert sha(root / name) == digest, f'Input changed: {root / name}'
    for (name, digest) in run['selection_sha256'].items():
        assert sha(selected / name) == digest, f'Selection metadata changed: {name}'
    assert not (base / 'adapter_config.json').exists(), 'Base path is an adapter'
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    for name in ('config.json', 'tokenizer.json', *set(index['weight_map'].values())):
        p = base / name
        assert p.is_file() and p.stat().st_size > 0, p
    from ab.gpt.act.acc_predictor import config as run_config
    run_config.USE_ZERO_COST_PROXIES = run['zcp_enabled']
    run_config.USE_THIRD_EARLY_EPOCH = True
    run_config.USE_LAYER_STATS = run['feature'] == 'taylor'
    run_config.ACTIVE_LAYER_FEATURES = 'taylor_imp' if run['feature'] == 'taylor' else None
    if mode == 'test':
        done = json.loads((out / 'refit_complete.json').read_text())
        assert done['epochs'] == EPOCHS and done['training_records'] == run['fit_n']
        assert done['run'] == key
        for (name, digest) in done['model_sha256'].items():
            assert sha(model_path / name) == digest, f'Refit model changed: {name}'
        with (out / 'test_started.json').open('x') as f:
            json.dump({'run': key, 'test': str(root / 'test_llm_dataset.jsonl')}, f)
        from ab.gpt.act.acc_predictor import eval as evaluator
        metrics = evaluator.test_model(model_path=model_path, data_path=root / 'test_llm_dataset.jsonl', output_path=out / 'test_predictions.csv', metrics_path=out / 'test_metrics.log')
        (out / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
        assert metrics['n_samples'] == run['eval_n'], metrics
        for name in ('test_predictions.csv', 'test_metrics.log'):
            assert (out / name).is_file() and (out / name).stat().st_size > 0, name
        result = {'run': key, 'epochs': EPOCHS, 'n_samples': metrics['n_samples'], 'n_valid': metrics['n_valid'], 'parse_failures': metrics['parse_failures']}
        (out / 'test_complete.json').write_text(json.dumps(result, indent=2))
        print('FINAL_REFIT_TEST_COMPLETE', json.dumps(result), flush=True)
        return
    assert not out.exists(), f'Refusing to overwrite: {out}'
    from ab.gpt.act.acc_predictor import train as t
    from ab.gpt.act.acc_predictor import config
    from ab.gpt.act.acc_predictor import eval as evaluator
    defaults = {k: v.default for (k, v) in inspect.signature(t.train_model).parameters.items()}
    assert defaults['num_epochs'] == 20
    assert defaults['early_stopping_patience'] == 3
    assert t.MAX_STEPS == 0
    assert t.torch.cuda.is_available(), 'CUDA unavailable'
    inspect.signature(evaluator.test_model).bind(model_path=model_path, data_path=root / 'test_llm_dataset.jsonl', output_path=out / 'test_predictions.csv', metrics_path=out / 'test_metrics.log')
    args = t.torch.load(selected / 'training_args.bin', map_location='cpu', weights_only=False)
    expected = {'learning_rate': defaults['learning_rate'], 'per_device_train_batch_size': defaults['batch_size'], 'gradient_accumulation_steps': defaults['gradient_accumulation_steps'], 'weight_decay': t.WEIGHT_DECAY, 'max_grad_norm': t.MAX_GRAD_NORM, 'seed': run['settings_seed'], 'num_train_epochs': 20}
    for (name, value) in expected.items():
        assert getattr(args, name) == value, (name, getattr(args, name), value)
    assert args.max_steps <= 0 and args.load_best_model_at_end
    assert getattr(args.optim, 'value', args.optim) == 'adamw_8bit'
    assert getattr(args.lr_scheduler_type, 'value', args.lr_scheduler_type) == 'linear'
    assert args.metric_for_best_model == 'eval_loss'
    seq = defaults['max_seq_len']
    saved_lengths = {name: getattr(args, name) for name in ('max_seq_length', 'max_length') if getattr(args, name, None) is not None}
    assert saved_lengths, 'No saved trainer sequence-length settings'
    print('MODEL_CONTEXT:', seq, 'SAVED_TRAINER_LENGTHS:', saved_lengths, flush=True)
    adapter = json.loads((selected / 'adapter_config.json').read_text())
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    assert adapter['r'] == defaults['lora_r']
    assert adapter['lora_alpha'] == defaults['lora_alpha']
    assert adapter['lora_dropout'] == t.LORA_DROPOUT
    assert set(adapter['target_modules']) == set(targets)
    assert adapter['bias'] == 'none' and (not adapter.get('use_rslora', False))
    assert adapter.get('base_model_name_or_path') in (str(base), config.MODEL_NAME)
    records = []
    ids = {}
    for (split, count) in (('train', run['train_n']), ('val', run['val_n']), ('test', run['eval_n'])):
        rows = [json.loads(x) for x in (root / f'{split}_llm_dataset.jsonl').read_text().splitlines() if x.strip()]
        assert len(rows) == count
        assert all((r.get('architecture_id') for r in rows))
        ids[split] = {str(r['architecture_id']) for r in rows}
        if split != 'test':
            records.extend(rows)
    assert not ids['train'] & ids['val']
    assert not (ids['train'] | ids['val']) & ids['test']
    assert len(records) == run['fit_n']
    batch = args.per_device_train_batch_size
    accumulation = args.gradient_accumulation_steps
    selection_steps = math.ceil(978 / (batch * accumulation)) * 20
    assert args.warmup_steps == max(1, int(selection_steps * defaults['warmup_ratio']))
    args.seed = run['seed']
    if getattr(args, 'data_seed', None) is not None:
        args.data_seed = run['seed']
    args.output_dir = str(model_path)
    args.logging_dir = str(out / 'trainer_logs')
    args.num_train_epochs = EPOCHS
    args.warmup_steps = max(1, int(math.ceil(run['fit_n'] / (batch * accumulation)) * EPOCHS * defaults['warmup_ratio']))
    args.eval_strategy = type(args.eval_strategy)('no')
    args.do_eval = False
    args.load_best_model_at_end = False
    args.metric_for_best_model = None
    args.greater_is_better = None
    if hasattr(args, 'evaluation_strategy'):
        args.evaluation_strategy = 'no'
    assert args.max_steps <= 0
    args.learning_rate = run['learning_rate']
    plain_fields = [f.name for f in fields(t.TrainingArguments) if f.init]
    assert 'max_length' not in plain_fields and 'max_seq_length' not in plain_fields
    args = t.TrainingArguments(**{name: getattr(args, name) for name in plain_fields if hasattr(args, name)})
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run_metadata.json').write_text(json.dumps({'run': key, 'epochs': EPOCHS, 'training_records': run['fit_n'], 'base': str(base), 'data': str(root), 'selected_evidence': str(selected), 'normalization': 'Raw per-record inputs; no fitted feature normalization', 'training_args': args.to_dict(), 'max_seq_len': seq, 'warmup_ratio_policy': defaults['warmup_ratio'], 'input_sha256': run['input_sha256']}, indent=2, default=str))
    print('REFIT_PREFLIGHT_PASS', key, 'variable-sized inner fit; epochs=7', flush=True)
    print(json.dumps(args.to_dict(), default=str), flush=True)
    t.SEED = run['seed']
    config.MODEL_FALLBACKS = ()
    (model, tokenizer) = t._load_model(str(base), seq)
    model = t.FastLanguageModel.get_peft_model(model, r=adapter['r'], target_modules=targets, lora_alpha=adapter['lora_alpha'], lora_dropout=adapter['lora_dropout'], bias='none', use_gradient_checkpointing='unsloth', random_state=run['seed'], use_rslora=False, loftq_config=None)
    t.FastLanguageModel.for_training(model)
    data = t.Dataset.from_list([{'messages': r['messages']} for r in records])
    data = data.map(t._format_dataset, fn_kwargs={'tokenizer': tokenizer}, batched=True, remove_columns=data.column_names)
    trainer = t.SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, dataset_text_field='text', max_seq_length=seq, packing=False, args=args, callbacks=[])
    for (name, expected) in saved_lengths.items():
        actual = getattr(trainer.args, name, None)
        assert actual == expected, ('Effective trainer length changed', name, actual, expected)
    assert trainer.args.seed == run['seed']
    assert getattr(trainer.args, 'data_seed', None) in (None, run['seed'])
    assert trainer.args.num_train_epochs == EPOCHS
    assert not trainer.args.load_best_model_at_end
    assert getattr(trainer.args.eval_strategy, 'value', trainer.args.eval_strategy) == 'no'
    (out / 'effective_trainer_args.json').write_text(json.dumps(trainer.args.to_dict(), indent=2, default=str))
    print('EFFECTIVE_TRAINER_MATCH:', saved_lengths, 'epochs=7; evaluation=no', flush=True)
    import importlib.metadata
    from ab.gpt.act.acc_predictor.common import _apply_chat_template_for_inference
    lengths = [len(tokenizer(row['text'], truncation=False, add_special_tokens=True)['input_ids']) for row in data]
    cap = min((int(v) for v in saved_lengths.values()))
    assert cap == 1024, ('Unexpected effective training cap', cap)
    assert max(lengths) <= cap, ('Training truncation', max(lengths), cap)
    assert trainer.args.learning_rate == run['learning_rate']
    assert not getattr(trainer.args, 'assistant_only_loss', False)
    assert getattr(trainer.args, 'completion_only_loss', None) in (None, False)
    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    validation_rows = [json.loads(line) for line in (root / 'test_llm_dataset.jsonl').read_text().splitlines() if line.strip()]
    inference_lengths = [len(text_tokenizer.encode(_apply_chat_template_for_inference(text_tokenizer, [m for m in row['messages'] if m['role'] in ('system', 'user')]), add_special_tokens=False)) for row in validation_rows]
    assert max(inference_lengths) + config.TEST_MAX_NEW_TOKENS <= seq
    audit = {'run': key, 'phase': 'inner', 'training_records': len(lengths), 'validation_records': len(validation_rows), 'effective_training_cap': cap, 'max_training_tokens': max(lengths), 'max_validation_input_tokens': max(inference_lengths), 'learning_rate': trainer.args.learning_rate, 'seed': trainer.args.seed, 'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'trl', 'unsloth', 'unsloth-zoo', 'datasets', 'peft', 'accelerate', 'bitsandbytes', 'numpy', 'pandas', 'scipy')}}
    (out / 'nested_runtime.json').write_text(json.dumps(audit, indent=2))
    print('NESTED_INNER_RUNTIME_PASS', json.dumps(audit), flush=True)
    trainer.train()
    assert trainer.state.epoch is not None and abs(trainer.state.epoch - EPOCHS) < 0.001
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    trainer.save_state()
    names = ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json')
    for name in names:
        assert (model_path / name).is_file() and (model_path / name).stat().st_size > 0, name
    (out / 'refit_complete.json').write_text(json.dumps({'run': key, 'epochs': EPOCHS, 'training_records': run['fit_n'], 'model_sha256': {name: sha(model_path / name) for name in names}}, indent=2))
    print('REFIT_TRAINING_COMPLETE', key, flush=True)
    del trainer, model, tokenizer, data
    gc.collect()
    t.torch.cuda.empty_cache()

def verify_nested_bundle():
    here = Path(__file__).resolve().parent
    bundle = here.parent
    receipt = json.loads((bundle / 'preparation_complete.json').read_text())
    assert sha(bundle / 'manifest.json') == receipt['manifest_sha256']
    for relative, expected in json.loads(
        (bundle / 'manifest.json').read_text()
    ).items():
        assert sha(bundle / relative) == expected, ('Preparation drift', relative)
    for relative, expected in json.loads(
        (here / 'execution_manifest.json').read_text()
    ).items():
        assert sha(here / relative) == expected, ('Execution drift', relative)


def score_inner(contract_path, key):
    import csv
    contract = json.loads(Path(contract_path).read_text())
    run = contract['runs'][key]
    assert run['phase'] == 'inner'
    out = Path(run['output'])
    registry = json.loads(
        (Path(run['data']) / 'validation_registry.json').read_text()
    )
    with (out / 'test_predictions.csv').open(newline='') as f:
        predictions = list(csv.DictReader(f))
    assert len(predictions) == len(registry) == run['eval_n']

    squared = []
    absolute = []
    invalid = 0
    exported = []

    for position, (prediction, item) in enumerate(zip(predictions, registry)):
        assert int(prediction['idx']) == position
        assert prediction['dataset'] == item['dataset']
        assert float(prediction['gt_best_accuracy']) == item['target']
        try:
            value = float(prediction['pred_best_accuracy'])
        except (TypeError, ValueError):
            value = float('nan')
        valid = math.isfinite(value) and 0 <= value <= 100

        exported.append({
            'record_id': item['record_id'],
            'architecture_id': item['architecture_id'],
            'dataset': item['dataset'],
            'task': item['task'],
            'target': item['target'],
            'prediction': value if valid else None,
            'valid': valid,
        })

        if item['task'] == 'img-classification':
            error = abs(value - item['target']) if valid else 100.0
            absolute.append(error)
            squared.append(error * error)
            invalid += int(not valid)

    assert squared, 'No classification validation records'
    with (out / 'inner_predictions.json').open('x') as f:
        json.dump(exported, f, indent=2, allow_nan=False)

    score = {
        'status': 'complete',
        'phase': 'inner',
        'run': key,
        'fold': run['fold'],
        'trial_id': run['trial_id'],
        'classification_n': len(squared),
        'invalid_classification_predictions': invalid,
        'selection_RMSE': math.sqrt(math.fsum(squared) / len(squared)),
        'selection_MAE': math.fsum(absolute) / len(absolute),
        'prediction_sha256': sha(out / 'test_predictions.csv'),
        'identity_prediction_sha256': sha(out / 'inner_predictions.json'),
        'contract_sha256': sha(Path(contract_path)),
        'outer_evaluated': False,
    }
    with (out / 'inner_score.json').open('x') as f:
        json.dump(score, f, indent=2, allow_nan=False)
    print('NESTED_INNER_COMPLETE', json.dumps(score), flush=True)

if __name__ == '__main__':
    verify_nested_bundle()
    main()
    if sys.argv[1] == 'test':
        score_inner(sys.argv[2], sys.argv[3])

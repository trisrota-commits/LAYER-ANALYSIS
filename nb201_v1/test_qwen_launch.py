import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import ast
import csv
import hashlib
import math
import re
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

def module():
    path = ROOT / 'nb201_v1/qwen_launch.py'
    assert path.exists(), 'Qwen launcher is not implemented'
    spec = importlib.util.spec_from_file_location('qwen_launch', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

class QwenTests(unittest.TestCase):
    def test_final_parser_and_real_csv_export(self):
        m = module()
        source = (ROOT / 'evidence/nb201-qwen-source-v1/eval.py').read_text()
        tree = ast.parse(source)
        parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_test_parse_assistant_json')
        namespace = dict(json=json, re=re)
        exec(compile(ast.Module(body=[parser], type_ignores=[]), 'actual_parser', 'exec'), namespace)
        evaluator = types.SimpleNamespace(_test_parse_assistant_json=namespace['_test_parse_assistant_json'])
        helpers = dict(json=json, math=math, sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
        exec(m.WORKER_HELPERS, helpers)
        helpers['configure_evaluator'](evaluator, {'dataset':'cifar100'})
        self.assertEqual(evaluator._test_parse_assistant_json('{"final_accuracy":60.5,"target_epoch":200}'), {'best_accuracy':60.5,'best_epoch':200})
        self.assertIsNone(evaluator._test_parse_assistant_json('not JSON'))
        with tempfile.TemporaryDirectory(prefix='hermes-verify-nb201-export-') as tmp:
            out = Path(tmp)/'result'; out.mkdir()
            root = Path(tmp)/'prompts'; root.mkdir()
            registry = [dict(record_id=str(i),dataset='cifar100',target=60.) for i in range(275)]
            (root/'test_registry.json').write_text(json.dumps(registry))
            with (out/'test_predictions.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['idx','dataset','gt_best_accuracy','pred_best_accuracy'])
                writer.writeheader()
                writer.writerows(dict(idx=i,dataset='cifar100',gt_best_accuracy=60.,pred_best_accuracy=59. if i else 'nan') for i in range(275))
            helpers['export_predictions'](out,root,'cifar100-taylor-s42',dict(seed=42,target_kind='final_validation_accuracy'))
            receipt=json.loads((out/'prediction_contract.json').read_text())
            self.assertEqual(receipt['finite_accuracy_predictions'],274)
            with (out/'per_record_predictions.csv').open() as f:
                saved=list(csv.DictReader(f))
            self.assertEqual(len(saved),275)
            self.assertTrue(math.isnan(float(saved[0]['prediction'])))
            with self.assertRaises(FileExistsError):
                helpers['export_predictions'](out,root,'cifar100-taylor-s42',dict(seed=42,target_kind='final_validation_accuracy'))

    def test_complete_bundle_matrix_and_preservation(self):
        m = module()
        self.assertTrue(hasattr(m, 'build'), 'Bundle builder missing')
        with tempfile.TemporaryDirectory(prefix='hermes-verify-nb201-') as tmp:
            dest = Path(tmp) / m.NAME
            m.build(dest)
            contract = json.loads((dest / 'contract.json').read_text())
            self.assertEqual(len(contract['runs']), 24)
            self.assertEqual(len(list((dest / 'jobs').glob('*.json'))), 24)
            self.assertEqual({r['seed'] for r in contract['runs'].values()}, set(range(42,48)))
            for key, run in contract['runs'].items():
                job = json.loads((dest / 'jobs' / (key + '.json')).read_text())
                container = job['spec']['template']['spec']['containers'][0]
                self.assertEqual(container['workingDir'], '/a/mm')
                self.assertIn(run['output'], container['args'][0])
                self.assertNotIn('zcp-off-toggle-refit7-v1', container['args'][0])
                self.assertEqual(run['settings_seed'], 42)
            for ds in ('cifar10','cifar100'):
                for split,n in [('train',772),('val',53),('test',275)]:
                    a = [json.loads(l) for l in (dest/'prompts'/ds/'no-layer'/f'{split}_llm_dataset.jsonl').read_text().splitlines()]
                    b = [json.loads(l) for l in (dest/'prompts'/ds/'taylor'/f'{split}_llm_dataset.jsonl').read_text().splitlines()]
                    self.assertEqual(len(a), n)
                    self.assertEqual([r['record_id'] for r in a], [r['record_id'] for r in b])
            with self.assertRaises(AssertionError):
                m.build(dest)
            ns = {'__file__': str(dest/'control.py'), '__name__':'fixture_control'}
            exec(m.CONTROL, ns)
            ns['verify'] = lambda: contract
            calls = []
            with patch.object(ns['sys'], 'argv', ['control.py','pilot']), patch.object(ns['subprocess'], 'check_output', return_value=''), patch.object(ns['subprocess'], 'run', side_effect=lambda cmd,check: calls.append(cmd)):
                ns['main']()
            self.assertEqual(len(calls),2)
            self.assertIn('--dry-run=server',calls[0])
            self.assertNotIn('--dry-run=server',calls[1])
            with patch.object(ns['sys'], 'argv', ['control.py','remaining']):
                with self.assertRaises(FileNotFoundError):
                    ns['main']()
            with patch.object(ns['sys'], 'argv', ['control.py','pilot']), patch.object(ns['subprocess'], 'check_output', return_value='job.batch/already-exists'):
                with self.assertRaises(AssertionError):
                    ns['main']()

    def test_worker_transformation(self):
        m = module()
        self.assertTrue(hasattr(m, 'worker_source'), 'Worker transformation missing')
        source = m.worker_source()
        compile(source, 'nb201_worker.py', 'exec')
        self.assertNotIn('training_records\'] == 1036', source)
        self.assertIn("('train', 772), ('val', 53), ('test', 275)", source)
        self.assertIn('selection_steps = math.ceil(978', source)
        self.assertIn("run_config.USE_LAYER_STATS = run['condition'] == 'taylor'", source)
        self.assertIn('NB201_RUNTIME_PREFLIGHT_PASS', source)
        self.assertIn('export_predictions(out, root, key, run)', source)
        self.assertIn('final_accuracy', source)

    def test_prompt_pairing_and_final_target(self):
        m = module()
        for ds in ('cifar10', 'cifar100'):
            rows = [json.loads(x) for x in (ROOT / 'evidence/nb201-collected-v1' / ds / 'snapshot.jsonl').read_text().splitlines()]
            for r in rows:
                plain = m.example(r, ds, False)
                taylor = m.example(r, ds, True)
                self.assertEqual(plain['messages'][0], taylor['messages'][0])
                self.assertEqual(plain['messages'][2], taylor['messages'][2])
                text = taylor['messages'][1]['content']
                before, after = text.split('LAYER_STATS_MEAN_BY_EPOCH\n')
                _, tail = after.split('\n\n', 1)
                self.assertEqual(before + tail, plain['messages'][1]['content'])
                for e in (1,2,3):
                    self.assertIn(f'epoch_{e}_accuracy:', text)
                if ds == 'cifar100':
                    answer = json.loads(plain['messages'][2]['content'])
                    self.assertEqual(answer, {'final_accuracy': round(r['target_final_validation_accuracy'], 2), 'target_epoch': 200})
                    self.assertNotIn('best_accuracy', json.dumps(plain['messages']))
                    self.assertNotIn('peak', json.dumps(plain['messages']))

if __name__ == '__main__':
    unittest.main()

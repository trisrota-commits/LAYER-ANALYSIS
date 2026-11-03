import importlib.util
from pathlib import Path
import tempfile
import unittest
import ast
import hashlib
import json
import shutil
import csv
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


def load():
    path = ROOT/'nb201_v1/imagenet16.py'
    assert path.exists(), 'ImageNet16 experiment launcher missing'
    spec = importlib.util.spec_from_file_location('imagenet16_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImageNetExperimentTests(unittest.TestCase):
    def test_full_collection_is_complete_and_exclusive(self):
        m=load()
        with tempfile.TemporaryDirectory(prefix='hermes-verify-imagenet-collect-') as tmp:
            root=Path(tmp);b=root/m.NAME;b.mkdir()
            m.put(b/'common.py',m.common_source((ROOT/'nb201_v1/cifar100.py').read_text()))
            common=m.module(b/'common.py','fixture_full_collect')
            cohort=m.js(ROOT/'nb201_v1/cohort.json');plan=m.partition(cohort)
            m.put(b/'manifest.json',plan)
            m.put(b/'contract.json',dict(sha256={},target_kind='final_validation_accuracy',benchmark_target_seeds=[777,888]))
            for k,ids in enumerate(plan['shards'][:-1]):
                m.put(b/'results/shards'/f's{k}'/'report.json',dict(attempted=len(ids),complete=len(ids),failures=[]))
            members={i:{} for i in plan['architectures']}
            with patch.object(common,'input_maps',return_value=(members,{})),patch.object(common,'validate_candidate',side_effect=lambda out,i,members,targets:dict(architecture_id=i,dataset='ImageNet16-120',fixture=True)):
                with self.assertRaises(FileNotFoundError):common.collect(root)
                self.assertFalse((b/'snapshot.jsonl').exists())
                ids=plan['shards'][-1];m.put(b/'results/shards/s4/report.json',dict(attempted=len(ids),complete=len(ids),failures=[]))
                common.collect(root)
                rows=[json.loads(l) for l in (b/'snapshot.jsonl').read_text().splitlines()]
                self.assertEqual([r['architecture_id'] for r in rows],plan['architectures'])
                self.assertEqual(m.js(b/'collection_complete.json')['n'],1100)
                with self.assertRaises(RuntimeError):common.collect(root)

    def test_image_checks_and_preparation_guards(self):
        m=load()
        self.assertTrue(hasattr(m,'prepare'),'ImageNet preparation missing')
        with tempfile.TemporaryDirectory(prefix='hermes-verify-imagenet-prep-') as tmp:
            root=Path(tmp);data=root/'images';data.mkdir()
            checks={name:hashlib.md5(b'fixture').hexdigest() for name in [f'train_data_batch_{i}' for i in range(1,11)]+['val_data']}
            for name in checks:(data/name).write_bytes(b'fixture')
            m.verify_images(data,checks)
            (data/'val_data').write_bytes(b'bad')
            with self.assertRaises(RuntimeError):m.verify_images(data,checks)
            (data/'val_data').write_bytes(b'fixture')
            parent=root/m.PARENT;parent.mkdir()
            shutil.copy2(ROOT/'nas201_taylor_pilot_v1.py',parent/'runner.py')
            (parent/'LayerAnalysis.py').write_text('# fixture only\n')
            ds=parent/'autodl/xautodl/datasets/DownsampledImageNet.py';ds.parent.mkdir(parents=True)
            ds.write_text('class ImageNet16:\n    train_list = '+repr(list(checks.items())[:-1])+'\n    valid_list = '+repr(list(checks.items())[-1:])+'\n')
            (parent/'contract.json').write_text(json.dumps(dict(official_commit=m.COMMIT,sha256={p.relative_to(parent).as_posix():m.sha(p) for p in parent.rglob('*') if p.is_file()})))
            tree=ast.parse((ROOT/'nas201_taylor_pilot_v1.py').read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
            jobnode=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='job' for t in n.targets))
            template=eval(compile(ast.Expression(jobnode.value),'original-job','eval'),dict(root=root,node='w2-mi',NAME=m.PARENT,bundle=parent))
            (parent/'job.json').write_text(json.dumps(template))
            cp=root/'nb201_v1/cohort.json';cp.parent.mkdir();shutil.copy2(ROOT/'nb201_v1/cohort.json',cp)
            metadata=root/'metadata';metadata.mkdir()
            cohort=m.js(cp);cohort['dataset']=m.DATASET
            membership={i:s for s,ids in cohort['splits'].items() for i in ids}
            for r in cohort['architectures']:r['id']=hashlib.md5(f"nb201:{m.DATASET}:{r['arch_index']}".encode()).hexdigest()
            (metadata/'cohort.json').write_text(json.dumps(cohort))
            with (metadata/'targets.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['record_id','architecture_id','dataset','split','target_final_validation_accuracy','target_epoch','benchmark_seed_777_accuracy','benchmark_seed_888_accuracy']);w.writeheader()
                w.writerows(dict(record_id=r['id'],architecture_id=r['arch_index'],dataset=m.DATASET,split=membership[r['arch_index']],target_final_validation_accuracy=30.,target_epoch=200,benchmark_seed_777_accuracy=29.,benchmark_seed_888_accuracy=31.) for r in cohort['architectures'])
            (metadata/'image-test-split.txt').write_text(json.dumps({'xvalid':['int',list(range(3000))],'xtest':['int',list(range(3000,6000))]}))
            (metadata/'metadata.json').write_text(json.dumps(dict(dataset=m.DATASET,n=1100,source_archive_sha256=m.ARCHIVE_SHA,benchmark_target_seeds=[777,888],target_kind='final_validation_accuracy',target_epoch=200,source_cifar10_cohort_sha256=m.sha(cp),files_sha256={p.name:m.sha(p) for p in metadata.iterdir()})))
            with patch.object(m,'EXTRACTOR_SHA',m.sha(parent/'LayerAnalysis.py')):
                m.prepare(root,data,metadata)
            b=root/m.NAME
            for name,h in m.js(b/'metadata.json')['files_sha256'].items():
                self.assertEqual(m.sha(b/name),h)
            self.assertEqual(len(list((b/'jobs').glob('*.json'))),5)
            for path in (b/'jobs').glob('*.json'):
                job=m.js(path);pod=job['spec']['template']['spec'];c=pod['containers'][0]
                self.assertNotIn('nodeSelector',pod)
                self.assertEqual(c['workingDir'],'/a/output')
                self.assertEqual(c['image'],m.IMAGE)
                volumes={v['name']:v for v in pod['volumes']}
                self.assertEqual(volumes['data']['hostPath']['path'],str(data))
                self.assertEqual(volumes['output']['hostPath']['path'],str(b/'results'))
            with self.assertRaises(RuntimeError):m.prepare(root,data,metadata)
            common=m.module(b/'common.py','fixture_prepared_common')
            self.assertTrue(hasattr(m,'submit'),'Submission controller missing')
            with self.assertRaises(FileNotFoundError):m.submit(root,'full')
            calls=[]
            empty=type('Result',(),{'returncode':0,'stdout':'','stderr':''})()
            with patch.object(m.subprocess,'run',side_effect=lambda cmd,**kw:(calls.append(cmd) or empty)):
                m.submit(root,'pilot')
            self.assertEqual(sum('--dry-run=server' in c for c in calls),1)
            self.assertEqual(sum('create' in c and '--dry-run=server' not in c for c in calls),1)
            with self.assertRaises(RuntimeError):m.submit(root,'pilot')

    def test_adapted_engine_and_exact_record_validation(self):
        m=load()
        self.assertTrue(hasattr(m,'engine_source'),'ImageNet engine adaptation missing')
        original=(ROOT/'nas201_taylor_pilot_v1.py').read_text()
        engine=m.engine_source(original);compile(engine,'engine.py','exec')
        def functions(source):
            return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        before,after=functions(original),functions(engine)
        for name in ('state_hash','measure_taylor','train_epoch'):
            self.assertEqual(before[name],after[name])
        self.assertIn('num_classes=120',engine)
        self.assertIn('RandomCrop(16,padding=2)',engine)
        self.assertIn('len(train)==151700 and len(validation)==6000',engine)
        self.assertIn('total==3000',engine)
        self.assertNotIn('datasets.CIFAR100(',engine)
        common=m.common_source((ROOT/'nb201_v1/cifar100.py').read_text())
        ns={'__name__':'test_common'};exec(common,ns)
        r=dict(record_id='id',architecture_id=3,architecture_string='arch',dataset='ImageNet16-120',candidate_seed=777,
            split='train',target_kind='final_validation_accuracy',target_epoch=200,benchmark_target_seeds=[777,888],target_final_validation_accuracy=30.)
        for e in (1,2,3):r[f'accuracy_epoch_{e}']=10.;r[f'taylor_epoch_{e}']=.01
        members={3:dict(id='id',arch_str='arch')};targets={3:dict(split='train',target_final_validation_accuracy='30.0')}
        ns['validate_record'](r,3,members,targets)
        with self.assertRaises(RuntimeError):
            ns['validate_record'](dict(r,dataset='cifar100'),3,members,targets)

    def test_frozen_cohort_targets_and_pilot_partition(self):
        m=load()
        with tempfile.TemporaryDirectory(prefix='hermes-verify-imagenet-targets-') as tmp:
            dest=Path(tmp)/'metadata'
            m.build_targets(ROOT/'nb201_v1/cohort.json',Path('C:/Users/trisr/Downloads/NATS-tss-v1_0-3ffb9-simple.tar'),dest)
            cohort=m.js(dest/'cohort.json')
            self.assertEqual(cohort['dataset'],'ImageNet16-120')
            self.assertEqual(cohort['target_seed_policy'],'Fixed benchmark seeds777/888 for every architecture; seed999 is not used')
            self.assertEqual(cohort['counts'],{'train':772,'val':53,'test':275})
            self.assertEqual(cohort['splits'],m.js(ROOT/'nb201_v1/cohort.json')['splits'])
            plan=m.partition(cohort)
            self.assertEqual(plan['shards'][0],[3,32,41])
            flat=sum(plan['shards'],[])
            self.assertEqual(len(flat),1100)
            self.assertEqual(len(set(flat)),1100)
            self.assertEqual(set(flat),set(plan['architectures']))
            with self.assertRaises(RuntimeError):
                m.build_targets(ROOT/'nb201_v1/cohort.json',Path('C:/Users/trisr/Downloads/NATS-tss-v1_0-3ffb9-simple.tar'),dest)


if __name__=='__main__':
    unittest.main(verbosity=2)

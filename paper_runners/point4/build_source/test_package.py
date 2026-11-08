import ast
import json
from pathlib import Path
import unittest

P = Path(__file__).parent / 'point4-all3-no-checkpoints-v1'

class PackageTests(unittest.TestCase):
    def test_complete_package(self):
        self.assertTrue(P.is_dir(), 'New package has not been built')
        selection = json.loads((P/'selection.json').read_text())
        self.assertEqual({d:len(v) for d,v in selection.items()}, {'cifar-10':15,'cifar-100':166,'imagenette':159})
        self.assertEqual(sum(bool(r.get('reuse')) for v in selection.values() for r in v),20)
        original = Path(__file__).parent
        ids = set(json.loads((original/'folds.json').read_text())[0]['inner_train_ids'])
        for dataset, selected in selection.items():
            self.assertEqual(len({r['group'] for r in selected}),len(selected))
            self.assertTrue(all(r['record_id'] in ids for r in selected))
        jobs=list((P/'jobs').glob('*.json'))
        self.assertEqual(len(jobs),6)
        for jobfile in jobs:
            job=json.loads(jobfile.read_text()); spec=job['spec']['template']['spec']
            self.assertEqual(spec['nodeSelector'],{'kubernetes.io/hostname':'w2-mi'})
            self.assertEqual(job['spec']['backoffLimit'],0)
        for worker in P.glob('lemur-*/worker.py'):
            tree=ast.parse(worker.read_text()); compile(tree,str(worker),'exec')
            candidate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='candidate')
            self.assertNotIn('torch.save',ast.unparse(candidate))
            self.assertIn('probe_checkpoint',ast.unparse(candidate))
            text=worker.read_text()
            self.assertNotIn('3 * checkpoint_bytes',text)
            self.assertIn("'persistent_checkpoint': False",text)

    def test_candidate_execution_without_checkpoint_files(self):
        import importlib.util
        import sys
        import tempfile
        import types
        from unittest.mock import patch
        worker=P/'lemur-robustness-all3-v1-cifar10/worker.py'
        spec=importlib.util.spec_from_file_location('test_worker',worker)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        trace=[]
        class Model:
            def train(self): pass
            def learn(self,loader):
                trace.append('learn')
                list(loader)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            mod.build=lambda *args: ({'epoch_timeout_seconds':10},{'id':'test'},dest,Model(),[1],None,None,None)
            mod.state_hash=lambda model:'test-state'
            mod.verify=lambda root:None
            def probe(*args):
                trace.append('probe')
                (dest/f'measurements-e{args[-1]}.jsonl').write_text('')
            mod.probe_checkpoint=probe
            with patch.dict(sys.modules,{'torch':types.ModuleType('torch')}):
                mod.candidate(dest,dest,dest,0)
            self.assertEqual(trace,['learn','probe']*3)
            self.assertEqual(list(dest.glob('*.pt')),[])
            self.assertEqual(json.loads((dest/'complete.json').read_text())['epochs'],[1,2,3])
            for epoch in (1,2,3):
                info=json.loads((dest/f'epoch-{epoch}.json').read_text())
                self.assertFalse(info['persistent_checkpoint'])
                self.assertEqual(info['state_sha256'],'test-state')

    def test_reuse_integrity_rejects_changed_measurements(self):
        import importlib.util
        import tempfile
        spec=importlib.util.spec_from_file_location('prep',P/'prepare.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old'; new=root/'new'; src=old/'results/candidate-0'
            src.mkdir(parents=True); new.mkdir()
            item={'index':0,'record_id':'test','group':'g','model_file':'m.py','transform_file':'t.py','record_file':'r.json','reuse':{'index':0}}
            for directory in (old,new):
                (directory/'m.py').write_text('pass')
                (directory/'t.py').write_text('pass')
                (directory/'r.json').write_text('{"id":"test"}')
            mod.put(old/'contract.json',{'selected':[item]})
            mod.put(src/'runtime.json',{})
            for e in (1,2,3):
                mod.put(src/f'epoch-{e}.json',{})
                mod.put(src/f'scaling-checks-e{e}.json',[])
                mod.put(src/f'state-check-e{e}.json',{'original_state_preserved':True,'original_modes_preserved':True,'original_gradients_preserved':True,'rng_restored':True,'measurements':1,'failed_measurements':0,'checkpoint_state_sha256':'state'})
                (src/f'measurements-e{e}.jsonl').write_text(json.dumps({'epoch':e,'status':'ok','checkpoint_state_sha256':'state'})+'\n')
            mod.put(src/'complete.json',{'status':'complete','record_id':'test','files_sha256':{p.name:mod.sha(p) for p in src.iterdir()}})
            source,names,done=mod.reuse_plan(old,new,item)
            self.assertEqual(source,src)
            self.assertTrue(all(not n.endswith('.pt') for n in names))
            (src/'measurements-e1.jsonl').write_text('CHANGED')
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                mod.reuse_plan(old,new,item)

if __name__=='__main__': unittest.main()

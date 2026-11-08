"""Recovered worker tests: stubbed training, not GPU experiment execution."""
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1] / "paper_runners/point4"

class Point4WorkerTests(unittest.TestCase):

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

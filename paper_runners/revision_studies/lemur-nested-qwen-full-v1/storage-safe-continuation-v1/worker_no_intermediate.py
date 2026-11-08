from pathlib import Path
import hashlib
import sys
sys.dont_write_bytecode = True
p = Path(__file__).resolve().parent.parent / 'worker.py'
expected = 'fe6309053b6b88f46f744515c8238031705751159a093659da789fd6bc745dad'
if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Original worker changed')
source = p.read_text()
anchor = "    args.learning_rate = run['learning_rate']"
replacement = "    args.learning_rate = run['learning_rate']\n    args.save_strategy = type(args.save_strategy)('no')\n    args.save_total_limit = None"
check_anchor = '    trainer.train()'
check_replacement = "    assert getattr(trainer.args.save_strategy, 'value', trainer.args.save_strategy) == 'no'\n    trainer.train()"
if source.count(anchor) != 1 or source.count(check_anchor) != 1:
    raise RuntimeError('Unexpected worker source')
source = source.replace(anchor, replacement, 1).replace(check_anchor, check_replacement, 1)
print('SAVE_POLICY: no intermediate checkpoints; final adapter retained', flush=True)
exec(compile(source, str(p), 'exec'), {'__name__': '__main__', '__file__': str(p)})

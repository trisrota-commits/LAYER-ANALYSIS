"""Portable adapter for the byte-preserved six-seed outer analysis and renderer.

Only input/output paths are relocated. The checksum-pinned original scientific
calculations remain unchanged; no training or Kubernetes operation is performed.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(repo):
    folder=Path(repo)/'paper_runners/nested_multiseed'
    provenance=json.loads((folder/'POSTPROCESSING_PROVENANCE.json').read_text())
    expected={r['filename']:r['sha256'] for r in provenance['files']}
    paths={name:folder/name for name in ['analyze_nested_outer_six_seeds.py','render_nested_outer_report.py']}
    for name,path in paths.items():
        if sha(path)!=expected[name]:
            raise ValueError('Original nested source changed: '+name)
    return paths


def load_analyzer(repo,archive,out):
    path=sources(repo)['analyze_nested_outer_six_seeds.py']
    spec=importlib.util.spec_from_file_location('archived_nested_analysis',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARCHIVE=Path(archive).resolve()
    module.OUTPUT=Path(out).resolve()
    return module


def run(repo,archive,out):
    repo,archive,out=Path(repo).resolve(),Path(archive).resolve(),Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Existing output preserved: '+str(out))
    paths=sources(repo)
    original_hashes={name:sha(path) for name,path in paths.items()}
    analyzer=load_analyzer(repo,archive,out)
    # Fail before creating any output directories if the supplied archive differs.
    if sha(archive)!=analyzer.EXPECTED:
        raise ValueError('Nested input archive checksum mismatch')
    renderer=paths['render_nested_outer_report.py']
    tree=ast.parse(renderer.read_text(),filename=str(renderer))
    assignments=[node for node in tree.body if isinstance(node,ast.Assign)
                 and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                 and node.targets[0].id=='ROOT']
    if len(assignments)!=1:
        raise ValueError('Unexpected renderer root assignment')
    assignments[0].value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(out))],keywords=[])
    ast.fix_missing_locations(tree)
    code=compile(tree,str(renderer),'exec')
    out.parent.mkdir(parents=True,exist_ok=True)
    analyzer.main()
    exec(code,{'__name__':'__main__','__file__':str(renderer)})
    for name,path in paths.items():
        if sha(path)!=original_hashes[name]:raise ValueError('Source changed during execution')
    if sha(archive)!=analyzer.EXPECTED:raise ValueError('Archive changed during execution')
    receipt_path=out/'complete.json'
    receipt=json.loads(receipt_path.read_text())
    for name,digest in receipt['files_sha256'].items():
        if Path(name).name!=name or sha(out/name)!=digest:
            raise ValueError('Output receipt mismatch: '+name)
    provenance=dict(adapter_sha256=sha(Path(__file__)),source_sha256=original_hashes,
                    input_archive_sha256=sha(archive),path_overrides_only=True,
                    renderer_override='ROOT assignment only; AST replacement',model_training=False)
    with (out/'adapter_execution.json').open('x') as f:json.dump(provenance,f,indent=2)
    receipt['files_sha256']['adapter_execution.json']=sha(out/'adapter_execution.json')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('NESTED_OUTER_COMPLETE: five metrics, paired Bonferroni5 intervals and PDF; inputs unchanged.',flush=True)

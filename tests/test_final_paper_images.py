from pathlib import Path
import importlib.util
import json
import hashlib
import pytest


def test_final_image_export_names_and_preservation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'analysis/final_paper_images.py'
    assert path.is_file(), 'Missing final manuscript image packaging'
    spec=importlib.util.spec_from_file_location('final_images_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from analysis.revision_paper_figures import FIGURES
    source=tmp_path/'source';source.mkdir()
    entries={};hashes={}
    for key,stem in FIGURES.items():
        files={}
        for ext in ['pdf','png']:
            name=f'{stem}.{ext}';p=source/name;p.write_bytes((key+ext).encode())
            digest=hashlib.sha256(p.read_bytes()).hexdigest();hashes[name]=digest
            files[ext]=dict(path=name,sha256=digest)
        entries[key]=dict(stem=stem,files=files,plot_data=[])
    index=source/'FIGURE_INDEX.json';index.write_text(json.dumps(dict(status='complete',figures=entries)))
    hashes[index.name]=hashlib.sha256(index.read_bytes()).hexdigest()
    (source/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256=hashes)))
    out=tmp_path/'out';result=module.export(source,out)
    assert result['unique_assets']==16 and result['figure_placements']==17
    assert (out/'images/fig_dcor.pdf').read_bytes()==b'S1pdf'
    assert result['figures']['main_2']['stem']=='fig_nb201_improve'
    assert result['figures']['S5']['stem']=='fig_strata'
    assert not (out/'images/fig_taylor_tally.pdf').exists()
    with pytest.raises(FileExistsError):module.export(source,out)
    (source/'fig_cost.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError):module.export(source,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

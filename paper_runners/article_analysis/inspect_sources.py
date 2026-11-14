"""Read-only audit provenance inspection (not part of production execution)."""
import json, zipfile
from pathlib import Path
if __name__=='__main__':
    repo=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(repo/'evidence/thesis-new-logs-20261112-084434.zip') as z:
        names=[n for n in z.namelist() if n.endswith('registry-audit-corrected-v1/verified/audit_results.json')]
        for name in names:
            obj=json.loads(z.read(name)); print(name); print(json.dumps(obj.get('source_script_sha256'),indent=2))

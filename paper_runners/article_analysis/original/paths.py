"""Resolve the data this project's analysis reads, extracting the archives once.

Everything the scripts need ships with the project:
  logs/overleaf-table-evidence-reproducible-20261019-v2.zip
      -> experiment_logs/ and overleaf-experiment-logs-20261015-100424.zip
      -> experiment_logs/<experiment>/<run>/results/*/test_predictions.csv
  data/data.zip
      -> 3acclayer/llm_finetuning_data_1.jsonl  (the 1151-record snapshot)

Archives are unpacked into .cache/ at the project root on first use.
"""
import os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")

def _unzip(archive, dest):
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    return dest

def experiment_logs():
    """Directory holding one subdirectory per experiment, each with results/."""
    outer = _unzip(os.path.join(ROOT, "logs",
                   "overleaf-table-evidence-reproducible-20261019-v2.zip"),
                   os.path.join(CACHE, "evidence"))
    inner_zip = os.path.join(outer, "overleaf-experiment-logs-20261015-100424.zip")
    inner = _unzip(inner_zip, os.path.join(CACHE, "runs"))
    return os.path.join(inner, "experiment_logs")

def snapshot():
    """The 1151-record predictor snapshot with per-epoch layer statistics."""
    d = _unzip(os.path.join(ROOT, "data", "data.zip"), os.path.join(CACHE, "data"))
    return os.path.join(d, "3acclayer", "llm_finetuning_data_1.jsonl")

def artifact(name):
    """A generated result file, written next to the scripts."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

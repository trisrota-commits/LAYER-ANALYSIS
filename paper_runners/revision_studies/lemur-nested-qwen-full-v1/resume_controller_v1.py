
from pathlib import Path
import importlib.util
import time
import traceback
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "original_controller", root / "control.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)

original_get_jobs = controller.get_jobs
original_completed = controller.completed
latest_jobs = {}
adoption = controller.js(root / "adoption.json")

def observed_get_jobs():
    jobs = original_get_jobs()
    latest_jobs.clear()
    latest_jobs.update(jobs)
    return jobs

def completed_with_visibility_grace(key, run):
    # Existing verification remains authoritative. Integrity errors propagate.
    if original_completed(key, run):
        return True

    name = controller.NAME + "-" + key
    if key == "inner-f0-c00" and adoption is not None:
        name = adoption["job"]

    job = latest_jobs.get(name)
    if not job or not job.get("status", {}).get("succeeded"):
        return False

    print(
        "RECEIPT_VISIBILITY_WAIT", name,
        "Job succeeded; waiting up to 300 seconds for verified receipt.",
        flush=True,
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        time.sleep(5)
        if original_completed(key, run):
            print("RECEIPT_VERIFIED_AFTER_WAIT", name, flush=True)
            return True

    # The original controller will stop rather than accept an unverified result.
    print("RECEIPT_STILL_MISSING", name, flush=True)
    return False

controller.get_jobs = observed_get_jobs
controller.completed = completed_with_visibility_grace

if __name__ == "__main__":
    try:
        print(
            "CONTROLLER_RESUMED: existing outputs retained; "
            "bounded receipt-visibility wait enabled.",
            flush=True,
        )
        controller.main()
    except Exception:
        print(
            "CONTROLLER_STOPPED — outputs preserved; inspect error.",
            flush=True,
        )
        traceback.print_exc()
        raise

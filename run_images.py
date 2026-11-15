"""Reproduce current paper images separately from the numerical table runner."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List current paper figure IDs without rendering')
    parser.add_argument('--inputs', type=Path, help='Completed, hash-verified paper-tables output directory')
    parser.add_argument('--out', type=Path, help='New image output directory; existing outputs are preserved')
    args = parser.parse_args(argv)
    if args.list:
        if args.inputs is not None or args.out is not None:
            parser.error('--list cannot be combined with --inputs or --out')
        from analysis.revision_paper_figures import FIGURES
        print(json.dumps(FIGURES, indent=2))
        return
    if args.inputs is None or not args.inputs.is_dir():
        parser.error('--inputs must be an existing completed paper-tables directory')
    if args.out is None:
        parser.error('--out is required')
    if args.out.exists():
        parser.error('Existing output preserved: ' + str(args.out))
    from analysis.revision_paper_figures import run
    run(ROOT, args.inputs.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()

"""Reproduce all final-paper images from completed tables; no table recalculation."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List current paper figure IDs without rendering')
    parser.add_argument('--inputs', type=Path, help='Completed table output (default: recomputed/tables)')
    parser.add_argument('--out', type=Path, help='New output directory (default: recomputed/images)')
    args = parser.parse_args(argv)
    if args.list:
        if args.inputs is not None or args.out is not None:
            parser.error('--list cannot be combined with --inputs or --out')
        from analysis.final_paper_images import PLACEMENTS
        print(json.dumps({placement:stem for placement, _, stem in PLACEMENTS}, indent=2))
        return
    args.inputs = args.inputs or Path('recomputed/tables')
    args.out = args.out or Path('recomputed/images')
    if not args.inputs.is_dir():
        parser.error('--inputs must be an existing completed paper-tables directory')
    if args.out.exists() or args.out.is_symlink():
        parser.error('Existing output preserved: ' + str(args.out))
    from analysis.public_runners import images
    images(ROOT, args.inputs.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()

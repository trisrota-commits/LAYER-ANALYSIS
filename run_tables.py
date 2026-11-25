"""Reproduce all final-paper numerical tables, including BCa and Zou analyses."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('recomputed/tables'),
                        help='New output directory (default: recomputed/tables)')
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error('Existing output preserved: '+str(args.out))
    from analysis.public_runners import tables
    tables(ROOT, args.out.resolve())


if __name__ == '__main__':
    main()

"""Correct only Figure 6.1's CIFAR-10 label, preserving the vector plot.

Requires PyMuPDF and NumPy. By default, kpsewhich locates the original CMR9
Type 1 font; --font can supply cmr9.pfb explicitly without a TeX lookup.
All outputs go into a new directory; the source is never modified.
"""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pymupdf


LABEL = 'CIFAR-10-valid'
SCALE = 3


def load_font(font_path=None):
    """Load the original label font, or an explicitly supplied font file."""
    if font_path is None:
        found = subprocess.check_output(['kpsewhich', 'cmr9.pfb'], text=True).strip()
        if not found:
            raise FileNotFoundError('cmr9.pfb not found; supply --font PATH')
        font_path = found
    return pymupdf.Font(fontbuffer=Path(font_path).read_bytes())


def verify_pixels(original_page, edited_page, region):
    """Require exact 3x RGB pixel equality outside the label rectangle."""
    if original_page.rect != edited_page.rect:
        raise ValueError('Page dimensions changed')
    arrays = []
    for page in (original_page, edited_page):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False)
        arrays.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n))
    changed = np.any(arrays[0] != arrays[1], axis=2)
    mask = np.zeros(changed.shape, dtype=bool)
    mask[max(0, int(region.y0*SCALE)):max(0, int(region.y1*SCALE)+1),
         max(0, int(region.x0*SCALE)):max(0, int(region.x1*SCALE)+1)] = True
    if np.any(changed & ~mask):
        raise ValueError('Unexpected changes outside label')


def relabel_summary(source, out, font_path=None):
    """Write verified taylor_summary.pdf and .png into a NEW directory.

    Return the PDF path. Reject ambiguous labels and pre-existing destinations.
    Verification runs before creating outputs, and again on the serialized PDF.
    """
    source, out = Path(source), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Output directory already exists: {out}')
    original_bytes = source.read_bytes()
    with pymupdf.open(stream=original_bytes, filetype='pdf') as original, \
            pymupdf.open(stream=original_bytes, filetype='pdf') as doc:
        if len(doc) != 1:
            raise ValueError('Expected a single-page Figure 6.1 PDF')
        page = doc[0]
        spans = [span for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
        matches = [span for span in spans if span['text'] == 'CIFAR-10']
        if len(matches) != 1:
            raise ValueError('Expected exactly one CIFAR-10 text span')
        span = matches[0]
        font = load_font(font_path)
        size = span['size']
        right = span['bbox'][2]
        x = right - font.text_length(LABEL, fontsize=size)
        if x <= 0:
            raise ValueError('New label would extend beyond the original page')
        page.add_redact_annot(pymupdf.Rect(span['bbox']), fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        writer = pymupdf.TextWriter(page.rect)
        writer.append((x, span['origin'][1]), LABEL, font=font, fontsize=size)
        writer.write_text(page)
        region = pymupdf.Rect(min(x, span['bbox'][0])-2, span['bbox'][1]-3,
                              right+2, span['bbox'][3]+3)
        verify_pixels(original[0], page, region)
        pdf_bytes = doc.tobytes(garbage=4, deflate=True)
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as saved:
            lines = saved[0].get_text().splitlines()
            if lines.count(LABEL) != 1 or 'CIFAR-10' in lines:
                raise ValueError('Saved PDF does not contain the corrected label')
            verify_pixels(original[0], saved[0], region)
            png_bytes = saved[0].get_pixmap(
                matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False).tobytes('png')
        # Exclusive creation prevents overwriting even if another process races us.
        out.mkdir(parents=True, exist_ok=False)
        dest = out / 'taylor_summary.pdf'
        with dest.open('xb') as handle:
            handle.write(pdf_bytes)
        with (out / 'taylor_summary.png').open('xb') as handle:
            handle.write(png_bytes)
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Original Figure 6.1 PDF')
    parser.add_argument('--out', required=True, type=Path, help='New output directory (must not exist)')
    parser.add_argument('--font', type=Path, help='CMR9 font file (default: kpsewhich cmr9.pfb)')
    args = parser.parse_args(argv)
    dest = relabel_summary(args.source, args.out, args.font)
    print('PASS: label corrected; page size unchanged; all pixels outside label unchanged.')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

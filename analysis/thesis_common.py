"""Small shared I/O helpers for exported-data reproduction; no framework imports."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name,
            'Unsafe archive member: ' + name)
    return p


def unpack_zip(archive, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        require(len({m.filename for m in members}) == len(members), 'Duplicate archive members')
        for member in members:
            safe_name(member.filename)
            require((member.external_attr >> 16) & 0o170000 != 0o120000, 'Archive symlink refused')
        destination.mkdir(parents=True, exist_ok=False)
        z.extractall(destination)
    return destination


def tar_files(path):
    with tarfile.open(path) as archive:
        result = {}
        for member in archive:
            safe_name(member.name)
            require(not member.issym() and not member.islnk(), 'Archive links refused')
            if member.isfile():
                name = member.name.removeprefix('./')
                require(name not in result, 'Duplicate member: ' + name)
                result[name] = archive.extractfile(member).read()
    return result


def csv_rows(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    if isinstance(data, bytes):
        data = data.decode('utf-8-sig')
    return list(csv.DictReader(io.StringIO(data, newline='')))


def module(path, name):
    # Some original analysis scripts parse argv at import time; expose only defaults.
    previous = sys.argv
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.argv = previous


def run_script(script, args, log):
    with Path(log).open('x', encoding='utf-8') as handle:
        completed = subprocess.run([sys.executable, '-B', str(script), *map(str, args)],
                                   stdout=handle, stderr=subprocess.STDOUT, text=True)
    require(completed.returncode == 0, 'Analysis failed; inspect ' + str(log))


def write_table(folder, number, rows, provenance):
    require(bool(rows), 'Empty table ' + number)
    name = 'table_' + number.replace('.', '_')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(folder) / (name + '.csv')
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def cell(value):
        if isinstance(value, float):
            return f'{value:.10g}'
        return str(value)
    md = '| ' + ' | '.join(fields) + ' |\n| ' + ' | '.join('---' for _ in fields) + ' |\n'
    md += '\n'.join('| ' + ' | '.join(cell(row.get(f, '')).replace('|', '/') for f in fields) + ' |' for row in rows)
    (Path(folder) / (name + '.md')).write_text('# Table ' + number + '\n\n' + provenance + '\n\n' + md + '\n', encoding='utf-8')
    def tex(value):
        value = cell(value)
        for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
            value = value.replace(a, b)
        return value
    content = '\\begin{tabular}{' + 'l' * len(fields) + '}\n' + ' & '.join(map(tex, fields)) + r' \\' + '\n\\hline\n'
    content += '\n'.join(' & '.join(tex(row.get(f, '')) for f in fields) + r' \\' for row in rows)
    content += '\n\\end{tabular}\n'
    (Path(folder) / (name + '.tex')).write_text(content, encoding='utf-8')
    return dict(path=path.name, rows=len(rows), provenance=provenance, sha256=sha(path))

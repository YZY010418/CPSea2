#!/usr/bin/env python3
"""Filter CPbase metadata with ordered rules and rebuild from full AFDB PDBs.

Python 3.10+, standard library only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

BASE_COLUMNS = [
    'LigandID', 'OriginalPosition', 'Length', 'absBSA', 'relBSA', 'raBSA',
    'BSA_per_length', 'rel_n_rBSA', 'rel_c_rBSA', 'GRAVY', 'HelixRatio',
    'SheetRatio', 'LigandAllUpstream', 'LigandAllDownstream', 'InterchainDisulfide',
]
IDS = ['Source_ID', 'Domain_ID', 'Cluster_ID', 'Ligand_ID']
RESID_COLUMNS = IDS + ['receptor_res_ids', 'ligand_res_ids']
ID_RE = re.compile(r'(.+)_(\d+)_cluster(\d+)_ligand(\d+)')
NAME_RE = re.compile(r'[A-Za-z][A-Za-z0-9_]*')
MISSING = {'', 'NA', 'N/A', 'NULL', 'NONE', 'NAN', '.'}


def open_text(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)


def input_files(paths):
    found = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob('*')
                         if p.is_file() and (p.name.endswith('.tsv') or p.name.endswith('.tsv.gz')))
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise ValueError(f'Input not found: {path}')
    if not found:
        raise ValueError('No TSV inputs found')
    return sorted(found)


def rows(path):
    with open_text(path) as stream:
        for number, row in enumerate(csv.reader(stream, delimiter='\t'), 1):
            if row and any(item.strip() for item in row):
                yield number, [item.strip() for item in row]


def quote(name):
    if not NAME_RE.fullmatch(name):
        raise ValueError(f'Invalid column name: {name!r}; use letters, digits and underscores')
    return f'"{name}"'


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Expected finite number, got {value!r}')
    return result


def bounds(rule, expression):
    parts, values = [], []
    for key, operator in [('min', '>='), ('max', '<='), ('gt', '>'), ('lt', '<')]:
        if key in rule:
            parts.append(f'{expression} {operator} ?')
            values.append(finite(rule[key]))
    if not parts:
        raise ValueError('A range needs at least one of min/max/gt/lt')
    return ' AND '.join(parts), values


def metadata_columns(first, custom):
    if 'LigandID' in first:
        return first, True
    if custom:
        return custom, False
    if len(first) == 15:
        return BASE_COLUMNS, False
    if len(first) == 16:
        return BASE_COLUMNS[:12] + ['SeparateChain'] + BASE_COLUMNS[12:], False
    raise ValueError('Headerless metadata needs 15/16 columns or config metadata_columns')


def load_metadata(db, paths, config):
    numeric = set(BASE_COLUMNS[2:]) | {'SeparateChain'}
    numeric.update(rule['field'] for rule in config.get('rules', []) if 'field' in rule)
    columns = None
    total = 0
    for path in paths:
        iterator = iter(rows(path))
        first = next(iterator, None)
        if first is None:
            continue
        names, header = metadata_columns(first[1], config.get('metadata_columns'))
        if len(set(n.lower() for n in names)) != len(names) or 'LigandID' not in names:
            raise ValueError(f'{path}: invalid/duplicate metadata columns')
        if {n.lower() for n in names} & {n.lower() for n in (IDS + ['receptor_res_ids', 'ligand_res_ids', 'source_path', 'output_path', 'reference_value', 'rank_value', 'member_count'])}:
            raise ValueError(f'{path}: reserved metadata column name')
        for name in names:
            quote(name)
        if columns is None:
            columns = list(names)
            definitions = [f'{quote(n)} {"REAL" if n in numeric else "TEXT"}' for n in names]
            definitions += [f'{quote(n)} {"TEXT" if n == "Source_ID" else "INTEGER"}' for n in IDS]
            db.execute('CREATE TABLE candidates (' + ','.join(definitions) + ', PRIMARY KEY (LigandID))')
            db.execute('CREATE UNIQUE INDEX identity ON candidates(Source_ID,Domain_ID,Cluster_ID,Ligand_ID)')
        if names != columns:
            raise ValueError(f'{path}: metadata schemas differ across shards')
        data = iterator if header else itertools.chain([first], iterator)
        batch = []
        for number, values in data:
            try:
                if len(values) != len(columns):
                    raise ValueError('Wrong column count')
                record = dict(zip(columns, values))
                match = ID_RE.fullmatch(record['LigandID'])
                if not match:
                    raise ValueError('LigandID must be <Source_ID>_<Domain_ID>_cluster<N>_ligand<N>')
                converted = []
                for name, value in zip(columns, values):
                    if value.upper() in MISSING:
                        converted.append(None)
                    elif name in numeric:
                        converted.append(finite(value))
                    else:
                        converted.append(value)
                identity = [match[1], *map(int, match.groups()[1:])]
                batch.append(converted + identity)
                total += 1
                if len(batch) >= 10000:
                    db.executemany('INSERT INTO candidates VALUES (' + ','.join('?' for _ in batch[0]) + ')', batch)
                    batch.clear()
            except (ValueError, sqlite3.IntegrityError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
        if batch:
            db.executemany('INSERT INTO candidates VALUES (' + ','.join('?' for _ in batch[0]) + ')', batch)
        db.commit()
        print(f'Metadata loaded: {total:,}', flush=True)
    if columns is None:
        raise ValueError('Metadata is empty')
    return columns


def apply_rules(db, columns, config):
    available = set(columns + IDS)
    audit = []
    for number, rule in enumerate(config.get('rules', []), 1):
        op = rule.get('op')
        allowed = {
            'range': {'field', 'min', 'max', 'gt', 'lt'},
            'top': {'field', 'scope', 'n', 'fraction', 'order', 'ties'},
            'relative': {'field', 'scope', 'stat', 'mode', 'min', 'max', 'gt', 'lt'},
            'aggregate': {'field', 'scope', 'stat', 'min', 'max', 'gt', 'lt'},
        }
        if op not in allowed or set(rule) - (allowed.get(op, set()) | {'op', 'name'}):
            raise ValueError(f'Rule {number}: unknown operation or option: {rule}')
        field = rule.get('field')
        if field is not None and (field not in available or field in IDS or field in ['LigandID', 'OriginalPosition']):
            raise ValueError(f'Rule {number}: invalid numeric field {field!r}')
        if field is None and not (op == 'aggregate' and rule.get('stat') == 'count'):
            raise ValueError(f'Rule {number}: field is required')
        value = quote(field) if field else '*'
        partition = ''
        if op != 'range':
            scope = rule.get('scope', 'cluster')
            if scope != 'cluster':
                raise ValueError(f'Rule {number}: scope must be cluster')
            partition = 'PARTITION BY ' + ','.join(quote(k) for k in IDS[:3])
            missing = ' OR '.join(f'{quote(k)} IS NULL' for k in IDS[:3])
            if db.execute(f'SELECT 1 FROM candidates WHERE {missing} LIMIT 1').fetchone():
                raise ValueError(f'Rule {number}: missing values for scope {scope!r}')
        before = db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        params = []
        if op == 'range':
            condition, params = bounds(rule, value)
            select = f'SELECT LigandID FROM candidates WHERE {condition}'
        elif op == 'top':
            if ('n' in rule) == ('fraction' in rule):
                raise ValueError('top requires exactly one of n/fraction')
            order = rule.get('order', 'desc').lower()
            ties = rule.get('ties', 'truncate')
            if order not in {'asc', 'desc'} or ties not in {'truncate', 'include'}:
                raise ValueError('top order must be asc/desc; ties must be truncate/include')
            function = 'ROW_NUMBER' if ties == 'truncate' else 'RANK'
            tiebreak = ', LigandID ASC' if ties == 'truncate' else ''
            ranked = (f'SELECT LigandID, {function}() OVER ({partition} ORDER BY {value} {order}{tiebreak}) AS rank_value, '
                      f'COUNT(*) OVER ({partition}) AS member_count FROM candidates WHERE {value} IS NOT NULL')
            if 'n' in rule:
                n = rule['n']
                if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                    raise ValueError('top n must be a positive integer')
                condition, params = 'rank_value <= ?', [n]
            else:
                fraction = finite(rule['fraction'])
                if not 0 < fraction <= 1:
                    raise ValueError('top fraction must be in (0, 1]')
                condition, params = 'rank_value - 1 < ? * member_count', [fraction]
            select = f'SELECT LigandID FROM ({ranked}) WHERE {condition}'
        else:
            stat = rule.get('stat', 'max')
            functions = {'max': 'MAX', 'min': 'MIN', 'mean': 'AVG', 'sum': 'SUM', 'count': 'COUNT'}
            if stat not in functions or (op == 'relative' and stat == 'count'):
                raise ValueError(f'Invalid statistic: {stat}')
            aggregate = f'{functions[stat]}({value}) OVER ({partition})'
            expression = 'reference_value'
            if op == 'relative':
                mode = rule.get('mode', 'ratio')
                if mode == 'ratio':
                    expression = f'1.0 * {value} / NULLIF(reference_value, 0)'
                elif mode == 'difference':
                    expression = f'{value} - reference_value'
                else:
                    raise ValueError('relative mode must be ratio/difference')
            condition, params = bounds(rule, expression)
            select = f'SELECT LigandID FROM (SELECT *, {aggregate} AS reference_value FROM candidates) WHERE {condition}'
        db.execute('CREATE TEMP TABLE keep AS ' + select, params)
        db.execute('CREATE UNIQUE INDEX keep_id ON keep(LigandID)')
        db.execute('DELETE FROM candidates WHERE NOT EXISTS (SELECT 1 FROM keep WHERE keep.LigandID=candidates.LigandID)')
        db.execute('DROP TABLE keep')
        db.commit()
        after = db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        audit.append({'rule': rule, 'before': before, 'after': after})
        print(f'Rule {number} ({op}): {before:,} -> {after:,}', flush=True)
    return audit


def join_resids(db, paths):
    db.execute('CREATE TABLE selected (LigandID TEXT PRIMARY KEY, receptor_res_ids TEXT, ligand_res_ids TEXT)')
    for path in paths:
        iterator = iter(rows(path))
        first = next(iterator, None)
        if first is None:
            continue
        if not set(RESID_COLUMNS) <= set(first[1]):
            raise ValueError(f'{path}: expected ResID header {RESID_COLUMNS}')
        names = first[1]
        if len(set(names)) != len(names):
            raise ValueError(f'{path}: duplicate ResID columns')
        batch = []
        for number, values in iterator:
            if len(values) != len(names):
                raise ValueError(f'{path}:{number}: wrong ResID column count')
            row = dict(zip(names, values))
            try:
                identity = [row['Source_ID'], *[int(row[k]) for k in IDS[1:]]]
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: invalid domain/cluster/ligand ID') from exc
            batch.append([row['receptor_res_ids'], row['ligand_res_ids'], *identity])
            if len(batch) >= 10000:
                insert_resids(db, batch)
                batch.clear()
        insert_resids(db, batch)
        db.commit()
    missing = db.execute('SELECT LigandID FROM candidates WHERE LigandID NOT IN (SELECT LigandID FROM selected) LIMIT 5').fetchall()
    if missing:
        raise ValueError(f'Selected metadata missing ResID entries: {missing}')


def insert_resids(db, batch):
    db.executemany('INSERT INTO selected SELECT LigandID, ?, ? FROM candidates WHERE Source_ID=? AND Domain_ID=? AND Cluster_ID=? AND Ligand_ID=?', batch)


def residue_ids(text):
    result = set()
    for token in text.split(','):
        match = re.fullmatch(r'\s*(-?\d+)(?:\s*-\s*(-?\d+))?\s*', token)
        if not match:
            raise ValueError(f'Invalid residue range: {token!r}')
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start or end - start > 100000:
            raise ValueError(f'Invalid residue range: {token!r}')
        result.update(range(start, end + 1))
    return result


def locate_sources(db, root, manifest):
    db.execute('CREATE TABLE sources (Source_ID TEXT PRIMARY KEY, path TEXT NOT NULL)')
    # Store the wanted set on disk; do not retain millions of paths in RAM.
    db.execute('CREATE TEMP TABLE wanted AS SELECT DISTINCT Source_ID FROM candidates')
    db.execute('CREATE UNIQUE INDEX wanted_id ON wanted(Source_ID)')
    def add(source, path):
        if db.execute('SELECT 1 FROM wanted WHERE Source_ID=?', (source,)).fetchone():
            if not path.is_file():
                raise ValueError(f'Source file not found: {path}')
            previous = db.execute('SELECT path FROM sources WHERE Source_ID=?', (source,)).fetchone()
            if previous and previous[0] != str(path):
                raise ValueError(f'Ambiguous source {source}: {previous[0]} and {path}; provide --source-manifest')
            db.execute('INSERT OR IGNORE INTO sources VALUES (?,?)', (source, str(path)))
    if manifest:
        with open_text(manifest) as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if not {'Source_ID', 'path'} <= set(reader.fieldnames or []):
                raise ValueError('Source manifest requires Source_ID and path columns')
            for row in reader:
                path = Path(row['path'])
                if not path.is_absolute():
                    path = manifest.parent / path
                add(row['Source_ID'], path.resolve())
    else:
        if root is None or not root.is_dir():
            raise ValueError('--source-dir must be an extracted AFDB50 PDB directory')
        for path in root.rglob('*'):
            if path.is_file() and (path.name.endswith('.pdb') or path.name.endswith('.pdb.gz')):
                stem = re.sub(r'\.pdb(?:\.gz)?$', '', path.name)
                add(stem, path.resolve())
                base = re.sub(r'-model_v\d+$', '', stem)
                if base != stem:
                    add(base, path.resolve())
    missing = db.execute('SELECT Source_ID FROM wanted WHERE Source_ID NOT IN (SELECT Source_ID FROM sources) LIMIT 5').fetchall()
    if missing:
        raise ValueError(f'Missing full AFDB sources: {missing}; use --source-manifest for different naming')
    db.commit()


def rebuild(source, receptor_text, ligand_text, identity):
    receptor, ligand = residue_ids(receptor_text), residue_ids(ligand_text)
    if not receptor or len(ligand) < 3 or receptor & ligand:
        raise ValueError('Need a nonempty receptor and >=3 disjoint ligand residues including caps')
    first, last = min(ligand), max(ligand)
    if len(ligand) != last - first + 1:
        raise ValueError('Ligand ResIDs including caps must be contiguous')
    caps = {first: ('ACE', {'CA', 'C', 'O'}), last: ('NME', {'N', 'CA'})}
    requested = receptor | ligand
    atoms = {}
    chains = set()
    with open_text(source) as stream:
        for line in stream:
            if line.startswith('ENDMDL'):
                break
            if line[:6] not in {'ATOM  ', 'HETATM'}:
                continue
            resid = int(line[22:26])
            if resid not in requested:
                continue
            if len(line.rstrip('\n')) < 54 or line[26] != ' ':
                raise ValueError('Source has truncated atom records or insertion codes')
            if line[16] not in {' ', 'A'}:
                raise ValueError('Alternate conformations are unsupported for full AFDB sources')
            chains.add(line[21])
            if len(chains) > 1:
                raise ValueError('Selected residue numbers occur in multiple source chains')
            name = line[12:16].strip()
            key = (resid, name)
            if key in atoms:
                raise ValueError(f'Duplicate source atom {key}')
            for a, b in [(30, 38), (38, 46), (46, 54)]:
                finite(line[a:b])
            atoms[key] = line.rstrip('\r\n').ljust(80)
    found = {}
    for resid, name in atoms:
        found.setdefault(resid, set()).add(name)
    for resid in requested:
        needed = caps[resid][1] if resid in caps else {'N', 'CA', 'C', 'O'}
        missing = needed - found.get(resid, set())
        if missing:
            raise ValueError(f'Residue {resid} lacks atoms {sorted(missing)}')
    result = [f'REMARK 900 CPBASE_ID: {identity}\n',
              'REMARK 900 CHAIN_IDS: RECEPTOR=R; LIGAND_WITH_CAPS=L\n',
              'REMARK 900 CAP_MODE: FIRST_AND_LAST_LIGAND_RES_IDS_ARE_ACE_AND_NME\n']
    serial = 1
    for chain, residues in [('R', receptor), ('L', ligand)]:
        for (resid, name), line in sorted(atoms.items()):
            if resid not in residues or (resid in caps and name not in caps[resid][1]):
                continue
            if serial > 99999:
                raise ValueError('PDB atom serial limit exceeded')
            record = 'HETATM' if resid in caps else line[:6]
            resname = caps[resid][0] if resid in caps else line[17:20]
            result.append(f'{record}{serial:5d}' + line[11:16] + ' ' + resname + ' ' + chain + line[22:] + '\n')
            serial += 1
        result.append('TER\n')
    result.append('END\n')
    return ''.join(result)


def export(db, columns, output, dry_run, compressed):
    names = columns + IDS + ['receptor_res_ids', 'ligand_res_ids', 'source_path', 'output_path']
    select = ','.join('c.' + quote(n) for n in columns + IDS)
    query = (f'SELECT {select}, s.receptor_res_ids, s.ligand_res_ids, p.path '
             'FROM candidates c JOIN selected s USING(LigandID) JOIN sources p USING(Source_ID) ORDER BY c.LigandID')
    count, failed = 0, 0
    with (output / 'selected.tsv').open('w') as manifest, (output / 'errors.tsv').open('w') as errors:
        writer = csv.writer(manifest, delimiter='\t')
        error_writer = csv.writer(errors, delimiter='\t')
        writer.writerow(names)
        error_writer.writerow(['LigandID', 'error'])
        for values in db.execute(query):
            record = dict(zip(names[:-1], values))
            identity = record['LigandID']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', identity):
                raise ValueError(f'Unsafe output ID {identity!r}')
            # Partition by an ID hash to avoid a single directory with millions of files.
            shard = hashlib.sha256(identity.encode()).hexdigest()[:2]
            path = output / 'structures' / shard / (identity + ('.pdb.gz' if compressed else '.pdb'))
            writer.writerow([*values, str(path)])
            count += 1
            if dry_run:
                continue
            try:
                text = rebuild(Path(record['source_path']), record['receptor_res_ids'], record['ligand_res_ids'], identity)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + '.partial')
                try:
                    opener = gzip.open if compressed else open
                    with opener(temporary, 'xt') as handle:
                        handle.write(text)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            except (ValueError, OSError) as exc:
                error_writer.writerow([identity, str(exc)])
                errors.flush()
                failed += 1
            if count % 1000 == 0:
                print(f'Rebuilt {count - failed:,}/{count:,}; failed {failed:,}', flush=True)
    return {'selected': count, 'written': 0 if dry_run else count - failed, 'failed': failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, nargs='+', required=True)
    parser.add_argument('--resid', type=Path, nargs='+', required=True)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--dry-run', action='store_true', help='select and resolve sources without writing PDBs')
    parser.add_argument('--gzip', action='store_true', help='write PDB.gz instead of PDB')
    args = parser.parse_args(argv)
    if not args.source_dir and not args.source_manifest:
        parser.error('Provide --source-dir or --source-manifest')
    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or set(config) - {'metadata_columns', 'rules'}:
            raise ValueError('Config supports metadata_columns and rules only')
        if not isinstance(config.get('rules', []), list) or any(not isinstance(r, dict) for r in config.get('rules', [])):
            raise ValueError('rules must be a list of objects')
        metadata, resids = input_files(args.metadata), input_files(args.resid)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        with sqlite3.connect(output / 'selection.sqlite') as db:
            db.execute('PRAGMA temp_store=FILE')
            columns = load_metadata(db, metadata, config)
            audit = apply_rules(db, columns, config)
            join_resids(db, resids)
            locate_sources(db, args.source_dir, args.source_manifest)
            summary = export(db, columns, output, args.dry_run, args.gzip)
        summary.update(dry_run=args.dry_run, rules=audit,
                       metadata_files=list(map(str, metadata)), resid_files=list(map(str, resids)))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return 1 if summary['failed'] else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

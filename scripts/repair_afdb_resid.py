#!/usr/bin/env python3
"""Resolve damaged AFDB IDs against metadata AND source PDBs; write new files.

The anomaly input has no header: TSV path, physical line number, then the six
original columns. Metadata input contains original 15-column metadata rows.
No match or multiple matches is an error, never a reason to guess a domain.
"""
import argparse
import hashlib
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def residues(text):
    result = set()
    for token in text.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(f"Invalid residue range: {token!r}")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first:
            raise ValueError(f"Reversed residue range: {token!r}")
        result.update(range(first, last + 1))
    return result


def resolve(anomalies, metadata, domains):
    records = defaultdict(list)
    for line in metadata.read_text().splitlines():
        fields = line.split("\t")
        match = re.fullmatch(r"(.+)_(\d+)_cluster(\d+)_ligand(\d+)", fields[0])
        if not match:
            raise ValueError(f"Invalid metadata ID: {fields[0]}")
        source, domain, cluster, ligand = match.groups()
        records[source, cluster, ligand].append((domain, fields))

    pdb_cache = {}
    patches = []
    for line in anomalies.read_text().splitlines():
        filename, number, *old = line.split("\t")
        if len(old) != 6 or old[1] != "NA" or not old[0].endswith("_"):
            raise ValueError(f"Unrecognized corruption at {filename}:{number}")
        source = old[0][:-1]
        receptor, ligand = residues(old[4]), residues(old[5])
        if receptor & ligand:
            raise ValueError(f"Receptor/ligand overlap at {filename}:{number}")
        matches = []
        for domain, fields in records[source, old[2], old[3]]:
            # Reproduce the actual writer defect; do not infer zero from NA.
            if f"{source}_{domain}"[:-2] != old[0]:
                continue
            first, last = min(ligand) + 1, max(ligand) - 1
            if ligand != set(range(first - 1, last + 2)):
                raise ValueError("Ligand range must be contiguous, including caps")
            if fields[1] != f"{source}_{domain}_{first}_{last}":
                continue
            if int(fields[2]) != len(ligand) - 2:
                continue
            pdb = domains / f"{source}_{domain}.pdb"
            if pdb not in pdb_cache:
                raw = pdb.read_bytes()
                atoms = defaultdict(set)
                for atom in raw.decode().splitlines():
                    if atom.startswith("ATOM  "):
                        atoms[int(atom[22:26])].add(atom[12:16].strip())
                pdb_cache[pdb] = atoms, hashlib.sha256(raw).hexdigest()
            atoms, digest = pdb_cache[pdb]
            if not (receptor | ligand) <= atoms.keys():
                continue
            if any(not {"N", "CA", "C", "O"} <= atoms[r] for r in ligand):
                continue
            matches.append((domain, fields[0], str(pdb), digest))
        if len(matches) != 1:
            raise ValueError(f"Expected one verified ID at {filename}:{number}, got {matches}")
        domain, metadata_id, pdb, digest = matches[0]
        patches.append(dict(file=filename, line=int(number), old="\t".join(old),
                            new="\t".join([source, domain, *old[2:]]),
                            metadata_id=metadata_id, pdb=pdb, pdb_sha256=digest))
    return patches


def verify_output(path, rows, expected_original_hash, expected_output_hash):
    """Re-read disk output and undo only listed fixes to reproduce input SHA256."""
    targets = sorted(rows)
    number = 0
    restored_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(8 * 1024 * 1024)
            if not data:
                break
            if not data.endswith(b"\n"):
                data += handle.readline()
            output_hash.update(data)
            end = number + data.count(b"\n")
            in_block = targets[bisect_right(targets, number):bisect_right(targets, end)]
            if in_block:
                lines = data.splitlines(keepends=True)
                for target in in_block:
                    patch = rows[target]
                    index = target - number - 1
                    if lines[index] != (patch["new"] + "\n").encode():
                        raise ValueError(f"Verification failed at {path}:{target}")
                    lines[index] = (patch["old"] + "\n").encode()
                data = b"".join(lines)
            restored_hash.update(data)
            number = end
    if (restored_hash.hexdigest(), output_hash.hexdigest()) != (expected_original_hash, expected_output_hash):
        raise ValueError(f"Checksum verification failed: {path}")


def apply(patches, output):
    """Copy affected shards, checking old rows and hashing both byte streams."""
    grouped = defaultdict(dict)
    for patch in patches:
        rows = grouped[patch["file"]]
        if patch["line"] in rows:
            raise ValueError("Duplicate patch line")
        rows[patch["line"]] = patch
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for filename, rows in grouped.items():
        source = Path(filename)
        destination = output / source.name
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if destination.exists():
            raise FileExistsError(destination)
        before = source.stat()
        old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
        changed = 0
        number = 0
        with source.open("rb") as src, temporary.open("xb") as dst:
            while True:
                data = src.read(8 * 1024 * 1024)
                if not data:
                    break
                # Finish the last line so a damaged row cannot cross blocks.
                if not data.endswith(b"\n"):
                    data += src.readline()
                old_hash.update(data)
                pieces = []
                position = 0
                while (match := data.find(b"\tNA\t", position)) >= 0:
                    start = data.rfind(b"\n", 0, match) + 1
                    end = data.find(b"\n", match) + 1
                    line_number = number + data.count(b"\n", 0, start) + 1
                    patch = rows.get(line_number)
                    if patch is None or data[start:end] != (patch["old"] + "\n").encode():
                        raise ValueError(f"Unexpected damaged row: {source}:{line_number}")
                    pieces.extend((data[position:start], (patch["new"] + "\n").encode()))
                    position = end
                    changed += 1
                number += data.count(b"\n")
                pieces.append(data[position:])
                fixed = b"".join(pieces)
                dst.write(fixed)
                new_hash.update(fixed)
        after = source.stat()
        if changed != len(rows) or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed or patches missing: {source}")
        verify_output(temporary, rows, old_hash.hexdigest(), new_hash.hexdigest())
        temporary.rename(destination)
        summary = dict(source=str(source), output=str(destination), rows=number - 1,
                       changed=changed, source_sha256=old_hash.hexdigest(),
                       output_sha256=new_hash.hexdigest(), verified=True)
        summaries.append(summary)
        (output / (source.name + ".audit.json")).write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve_args = sub.add_parser("resolve")
    resolve_args.add_argument("anomalies", type=Path)
    resolve_args.add_argument("metadata", type=Path)
    resolve_args.add_argument("domains", type=Path)
    resolve_args.add_argument("manifest", type=Path)
    apply_args = sub.add_parser("apply")
    apply_args.add_argument("manifest", type=Path)
    apply_args.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "resolve":
        patches = resolve(args.anomalies, args.metadata, args.domains)
        with args.manifest.open("x") as handle:
            json.dump(patches, handle, indent=2)
            handle.write("\n")
        print(f"Resolved {len(patches)} rows with unique metadata/PDB evidence")
    else:
        apply(json.loads(args.manifest.read_text()), args.output)


if __name__ == "__main__":
    main()

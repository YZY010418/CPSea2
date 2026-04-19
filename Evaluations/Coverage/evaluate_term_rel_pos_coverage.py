#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import multiprocessing as mp
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


CAP_RESNAMES = {"ACE", "NME"}
VECTOR_DIM = 15


@dataclass
class AtomRecord:
    name: str
    coord: np.ndarray

    def get_coord(self):
        return self.coord


@dataclass
class ResidueBlock:
    res_name: str
    chain_id: str
    auth_seq_id: str
    label_seq_id: str
    ins_code: str
    atoms: List[AtomRecord] = field(default_factory=list)

    def __iter__(self):
        return iter(self.atoms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate one 15D terminal relative position descriptor per ligand block "
            "from ACE-firstAA and lastAA-NME amides."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing .cif.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV path. Each row is one ligand block and each column is one descriptor dimension.",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def open_cif_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def collect_input_files(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".cif") or path.name.endswith(".cif.gz"))
    )


def split_data_blocks(lines: Sequence[str]) -> List[Tuple[str, List[str]]]:
    blocks: List[Tuple[str, List[str]]] = []
    current_name = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("data_"):
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = stripped.strip()[5:]
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        blocks.append((current_name, current_lines))

    return blocks


def find_atom_site_loop(block_lines: Sequence[str]) -> Optional[Tuple[List[str], int]]:
    idx = 0
    while idx < len(block_lines):
        if block_lines[idx].strip() != "loop_":
            idx += 1
            continue

        idx += 1
        columns: List[str] = []
        while idx < len(block_lines):
            stripped = block_lines[idx].strip()
            if stripped.startswith("_"):
                columns.append(stripped)
                idx += 1
                continue
            break

        if columns and any(column.startswith("_atom_site.") for column in columns):
            return columns, idx

        while idx < len(block_lines):
            stripped = block_lines[idx].strip()
            if stripped == "#" or stripped == "loop_" or stripped.startswith("data_"):
                break
            idx += 1

    return None


def build_column_lookup(columns: Sequence[str]) -> dict:
    return {column: idx for idx, column in enumerate(columns)}


def pick_first_present(column_to_idx: dict, keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        if key in column_to_idx:
            return column_to_idx[key]
    return None


def parse_atom_rows(block_lines: Sequence[str]) -> List[dict]:
    atom_site_loop = find_atom_site_loop(block_lines)
    if atom_site_loop is None:
        return []

    columns, start_idx = atom_site_loop
    column_to_idx = build_column_lookup(columns)

    group_idx = pick_first_present(column_to_idx, ["_atom_site.group_PDB"])
    atom_name_idx = pick_first_present(column_to_idx, ["_atom_site.auth_atom_id", "_atom_site.label_atom_id"])
    res_name_idx = pick_first_present(column_to_idx, ["_atom_site.auth_comp_id", "_atom_site.label_comp_id"])
    chain_idx = pick_first_present(column_to_idx, ["_atom_site.auth_asym_id", "_atom_site.label_asym_id"])
    auth_seq_idx = pick_first_present(column_to_idx, ["_atom_site.auth_seq_id"])
    label_seq_idx = pick_first_present(column_to_idx, ["_atom_site.label_seq_id"])
    ins_idx = pick_first_present(column_to_idx, ["_atom_site.pdbx_PDB_ins_code"])
    x_idx = pick_first_present(column_to_idx, ["_atom_site.Cartn_x"])
    y_idx = pick_first_present(column_to_idx, ["_atom_site.Cartn_y"])
    z_idx = pick_first_present(column_to_idx, ["_atom_site.Cartn_z"])

    required = [atom_name_idx, res_name_idx, chain_idx, x_idx, y_idx, z_idx]
    if any(idx is None for idx in required):
        raise ValueError("Missing required _atom_site columns in CIF block")

    rows: List[dict] = []
    for idx in range(start_idx, len(block_lines)):
        stripped = block_lines[idx].strip()
        if not stripped:
            continue
        if stripped == "#" or stripped == "loop_" or stripped.startswith("_") or stripped.startswith("data_"):
            break

        fields = shlex.split(block_lines[idx], posix=True)
        if len(fields) != len(columns):
            raise ValueError(
                f"Malformed _atom_site row in block: expected {len(columns)} fields, got {len(fields)}"
            )

        if group_idx is not None and fields[group_idx] != "ATOM":
            continue

        rows.append(
            {
                "atom_name": fields[atom_name_idx].strip(),
                "res_name": fields[res_name_idx].strip().upper(),
                "chain_id": fields[chain_idx].strip(),
                "auth_seq_id": "" if auth_seq_idx is None else fields[auth_seq_idx].strip(),
                "label_seq_id": "" if label_seq_idx is None else fields[label_seq_idx].strip(),
                "ins_code": "" if ins_idx is None or fields[ins_idx] in (".", "?", "") else fields[ins_idx].strip(),
                "coord": np.array(
                    [float(fields[x_idx]), float(fields[y_idx]), float(fields[z_idx])],
                    dtype=np.float32,
                ),
            }
        )

    return rows


def build_residue_blocks(atom_rows: Sequence[dict]) -> List[ResidueBlock]:
    residues: List[ResidueBlock] = []
    current_key = None
    current_residue: Optional[ResidueBlock] = None

    for row in atom_rows:
        residue_key = (
            row["chain_id"],
            row["auth_seq_id"],
            row["label_seq_id"],
            row["ins_code"],
            row["res_name"],
        )

        if residue_key != current_key:
            if current_residue is not None:
                residues.append(current_residue)
            current_residue = ResidueBlock(
                res_name=row["res_name"],
                chain_id=row["chain_id"],
                auth_seq_id=row["auth_seq_id"],
                label_seq_id=row["label_seq_id"],
                ins_code=row["ins_code"],
            )
            current_key = residue_key

        current_residue.atoms.append(AtomRecord(name=row["atom_name"], coord=row["coord"]))

    if current_residue is not None:
        residues.append(current_residue)

    return residues


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    return v / norm


def terminal_descriptor(a, b, c, d, e):
    peptide_axis = normalize(d - b)
    carbonyl_axis = normalize(c - b)
    plane_normal = normalize(np.cross(peptide_axis, carbonyl_axis))

    if np.linalg.norm(plane_normal) <= 1e-8:
        flank_axis = normalize(e - a)
        plane_normal = normalize(np.cross(peptide_axis, flank_axis))

    center = 0.5 * (b + d)
    return peptide_axis, plane_normal, center.astype(np.float32)


def get_atom_coord(block, atom_name):
    if block is None:
        return None
    for atom in block:
        if atom.name == atom_name:
            return np.array(atom.get_coord(), dtype=np.float32)
    return None


def compute_ligand_term_rel_pos(ace_block, first_block, last_block, nme_block):
    """Compute one 15D descriptor per ligand from ACE-firstAA and lastAA-NME amides."""
    try:
        c_ac = get_atom_coord(ace_block, "C")
        c_n = get_atom_coord(first_block, "C")
        o_n = get_atom_coord(first_block, "O")
        n_n = get_atom_coord(first_block, "N")
        ca_n = get_atom_coord(first_block, "CA")

        ca_c = get_atom_coord(last_block, "CA")
        c_c = get_atom_coord(last_block, "C")
        o_c = get_atom_coord(last_block, "O")
        n_c = get_atom_coord(nme_block, "N")
        ca_cn = get_atom_coord(nme_block, "CA")

        if any(x is None for x in [c_ac, c_n, o_n, n_n, ca_n, ca_c, c_c, o_c, n_c, ca_cn]):
            return np.zeros(VECTOR_DIM, dtype=np.float32)

        d_n, norm_n, center_n = terminal_descriptor(c_ac, c_n, o_n, n_n, ca_n)
        d_c, norm_c, center_c = terminal_descriptor(ca_c, c_c, o_c, n_c, ca_cn)

        d = np.linalg.norm(center_n - center_c)
        r = (center_n - center_c) / (d + 1e-8) if d > 1e-6 else np.zeros(3, dtype=np.float32)

        dots = np.array(
            [
                np.dot(r, d_n),
                np.dot(r, norm_n),
                np.dot(r, d_c),
                np.dot(r, norm_c),
                np.dot(d_n, d_c),
                np.dot(d_n, norm_c),
                np.dot(norm_n, d_c),
                np.dot(norm_n, norm_c),
            ],
            dtype=np.float32,
        )

        triples = np.array(
            [
                np.dot(r, np.cross(d_c, norm_c)),
                np.dot(d_n, np.cross(d_c, norm_c)),
                np.dot(norm_n, np.cross(d_c, norm_c)),
                np.dot(r, np.cross(d_n, norm_n)),
                np.dot(d_c, np.cross(d_n, norm_n)),
                np.dot(norm_c, np.cross(d_n, norm_n)),
            ],
            dtype=np.float32,
        )

        return np.concatenate([np.array([d], dtype=np.float32), dots, triples])
    except Exception:
        return np.zeros(VECTOR_DIM, dtype=np.float32)


def extract_ligand_vector_from_block(block_lines: Sequence[str]) -> np.ndarray:
    atom_rows = parse_atom_rows(block_lines)
    if not atom_rows:
        return np.zeros(VECTOR_DIM, dtype=np.float32)

    residues = build_residue_blocks(atom_rows)
    if len(residues) < 4:
        return np.zeros(VECTOR_DIM, dtype=np.float32)

    ace_block = residues[0] if residues[0].res_name == "ACE" else None
    nme_block = residues[-1] if residues[-1].res_name == "NME" else None
    if ace_block is None or nme_block is None:
        return np.zeros(VECTOR_DIM, dtype=np.float32)

    core_residues = [residue for residue in residues[1:-1] if residue.res_name not in CAP_RESNAMES]
    if len(core_residues) == 0:
        return np.zeros(VECTOR_DIM, dtype=np.float32)

    first_block = core_residues[0]
    last_block = core_residues[-1]
    return compute_ligand_term_rel_pos(ace_block, first_block, last_block, nme_block)


def iter_ligand_vectors(path: Path):
    with open_cif_text(path) as handle:
        lines = handle.read().splitlines()

    for block_name, block_lines in split_data_blocks(lines):
        if "ligand" not in block_name.lower():
            continue
        yield extract_ligand_vector_from_block(block_lines)


def process_one_file(path: Path) -> List[np.ndarray]:
    return list(iter_ligand_vectors(path))


def write_output_csv(output_path: Path, vectors: Sequence[np.ndarray]) -> None:
    header = [f"term_rel_pos_{i}" for i in range(VECTOR_DIM)]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for vector in vectors:
            writer.writerow([f"{float(x):.6f}" for x in vector])


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)

    files = collect_input_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .cif/.cif.gz files found under: {input_dir}")

    vectors: List[np.ndarray] = []
    cpus = max(1, args.cpus)

    if cpus == 1:
        for path in tqdm(files, total=len(files), desc="Processing CIFs", unit="file"):
            vectors.extend(process_one_file(path))
    else:
        with mp.Pool(processes=cpus) as pool:
            for file_vectors in tqdm(
                pool.imap_unordered(process_one_file, files, chunksize=1),
                total=len(files),
                desc="Processing CIFs",
                unit="file",
            ):
                vectors.extend(file_vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output_csv(output_path, vectors)

    print(f"Scanned files: {len(files)}")
    print(f"Ligand vectors written: {len(vectors)}")
    print(f"CPUs used: {cpus}")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()

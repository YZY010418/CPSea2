#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import math
import multiprocessing as mp
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Use non-interactive backend for clusters / servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


CAP_RESNAMES = {"ACE", "NME"}
AXIS_EPS = 1e-6
DEG = 180.0 / math.pi


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


@dataclass
class PoseRecord:
    source_file: str
    block_name: str
    rho: float
    psi: float
    cos_psi: float
    axis_x: float
    axis_y: float
    axis_z: float
    theta_r: float
    phi_r: float
    tx: float
    ty: float
    tz: float
    valid_axis: int


@dataclass
class BlockFailure:
    source_file: str
    block_name: str
    reason: str


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute terminal relative pose descriptors from ACE-firstAA and "
            "lastAA-NME amides for each ligand/legend block in CIF/CIF.GZ files, "
            "then export summary CSV and coverage plots."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input directory containing .cif / .cif.gz files.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for CSV, summary, and figures.",
    )
    parser.add_argument(
        "-c", "--cpus", type=int, default=1,
        help="Number of worker processes. Default: 1.",
    )
    parser.add_argument(
        "--bins-rho", type=int, default=80,
        help="Bins for rho histogram / rho-psi heatmap. Default: 80.",
    )
    parser.add_argument(
        "--bins-psi", type=int, default=72,
        help="Bins for psi-related histogram / heatmap. Default: 72.",
    )
    parser.add_argument(
        "--bins-sphere-lon", type=int, default=72,
        help="Longitude bins for rotation-axis occupancy map. Default: 72.",
    )
    parser.add_argument(
        "--bins-sphere-lat", type=int, default=36,
        help="Latitude bins for rotation-axis occupancy map. Default: 36.",
    )
    parser.add_argument(
        "--block-keywords", nargs="*", default=["ligand", "legend"],
        help=(
            "Only data blocks whose names contain at least one of these keywords "
            "(case-insensitive) will be processed. Default: ligand legend"
        ),
    )
    return parser.parse_args()


# -----------------------------
# CIF parsing
# -----------------------------
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


def build_column_lookup(columns: Sequence[str]) -> Dict[str, int]:
    return {column: idx for idx, column in enumerate(columns)}


def pick_first_present(column_to_idx: Dict[str, int], keys: Sequence[str]) -> Optional[int]:
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
                    dtype=np.float64,
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


# -----------------------------
# Geometry helpers
# -----------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / norm


def safe_acos(x: float) -> float:
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def get_atom_coord(block: ResidueBlock, atom_name: str) -> Optional[np.ndarray]:
    for atom in block:
        if atom.name == atom_name:
            return np.array(atom.get_coord(), dtype=np.float64)
    return None


def terminal_frame(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray):
    """
    Build a right-handed local frame from one terminal amide motif.

    Inputs reuse the same convention as the user's original script:
      a: flank atom for fallback
      b: carbonyl C
      c: carbonyl O
      d: amide N across the terminal bond
      e: flank atom for fallback
    """
    x_axis = normalize(d - b)
    carbonyl_axis = normalize(c - b)
    z_axis = normalize(np.cross(x_axis, carbonyl_axis))

    if np.linalg.norm(z_axis) <= 1e-8:
        flank_axis = normalize(e - a)
        z_axis = normalize(np.cross(x_axis, flank_axis))

    if np.linalg.norm(x_axis) <= 1e-8 or np.linalg.norm(z_axis) <= 1e-8:
        raise ValueError("Degenerate local frame")

    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    if np.linalg.norm(y_axis) <= 1e-8 or np.linalg.norm(z_axis) <= 1e-8:
        raise ValueError("Degenerate orthonormal local frame")

    center = 0.5 * (b + d)
    frame = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are basis vectors in global coords
    return frame, center.astype(np.float64)


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Return (psi, cos_psi, axis) where psi in [0, pi], axis is unit vector.
    Axis is undefined when psi ~ 0; caller should gate on psi.
    """
    trace = float(np.trace(R))
    cos_psi = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    psi = safe_acos(cos_psi)

    if psi < 1e-8:
        return psi, cos_psi, np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    # Generic case.
    sin_psi = math.sin(psi)
    if abs(sin_psi) > 1e-6:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        ) / (2.0 * sin_psi)
        axis = normalize(axis)
        if np.linalg.norm(axis) > 0:
            return psi, cos_psi, axis

    # Near-pi fallback.
    diag = np.diag(R)
    axis = np.empty(3, dtype=np.float64)
    axis[0] = math.sqrt(max(0.0, (diag[0] + 1.0) / 2.0))
    axis[1] = math.sqrt(max(0.0, (diag[1] + 1.0) / 2.0))
    axis[2] = math.sqrt(max(0.0, (diag[2] + 1.0) / 2.0))

    if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-8:
        axis[1] = math.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = math.copysign(axis[2], R[0, 2] + R[2, 0])
    elif axis[1] >= axis[2] and axis[1] > 1e-8:
        axis[0] = math.copysign(axis[0], R[0, 1] + R[1, 0])
        axis[2] = math.copysign(axis[2], R[1, 2] + R[2, 1])
    elif axis[2] > 1e-8:
        axis[0] = math.copysign(axis[0], R[0, 2] + R[2, 0])
        axis[1] = math.copysign(axis[1], R[1, 2] + R[2, 1])
    else:
        axis = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return psi, cos_psi, axis

    axis = normalize(axis)
    return psi, cos_psi, axis


def axis_to_theta_phi(axis: np.ndarray) -> Tuple[float, float]:
    axis = normalize(axis)
    if np.linalg.norm(axis) <= 1e-8:
        return float("nan"), float("nan")
    theta = safe_acos(axis[2])               # polar angle from +z in [0, pi]
    phi = float(np.arctan2(axis[1], axis[0]))  # longitude in [-pi, pi]
    return theta, phi


def compute_terminal_pose(ace_block: ResidueBlock, first_block: ResidueBlock,
                          last_block: ResidueBlock, nme_block: ResidueBlock) -> Tuple[PoseRecord, Optional[str]]:
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

        needed = [c_ac, c_n, o_n, n_n, ca_n, ca_c, c_c, o_c, n_c, ca_cn]
        if any(x is None for x in needed):
            return None, "missing_required_atoms"

        frame_n, center_n = terminal_frame(c_ac, c_n, o_n, n_n, ca_n)
        frame_c, center_c = terminal_frame(ca_c, c_c, o_c, n_c, ca_cn)

        t_global = center_c - center_n
        t_local = frame_n.T @ t_global
        rho = float(np.linalg.norm(t_local))

        R_rel = frame_n.T @ frame_c
        psi, cos_psi, axis_local = rotation_matrix_to_axis_angle(R_rel)

        valid_axis = int(np.isfinite(axis_local).all() and psi > AXIS_EPS)
        if valid_axis:
            theta_r, phi_r = axis_to_theta_phi(axis_local)
            axis_x, axis_y, axis_z = [float(v) for v in axis_local]
        else:
            theta_r = float("nan")
            phi_r = float("nan")
            axis_x = float("nan")
            axis_y = float("nan")
            axis_z = float("nan")

        rec = PoseRecord(
            source_file="",
            block_name="",
            rho=rho,
            psi=float(psi),
            cos_psi=float(cos_psi),
            axis_x=axis_x,
            axis_y=axis_y,
            axis_z=axis_z,
            theta_r=theta_r,
            phi_r=phi_r,
            tx=float(t_local[0]),
            ty=float(t_local[1]),
            tz=float(t_local[2]),
            valid_axis=valid_axis,
        )
        return rec, None
    except Exception as exc:
        return None, f"geometry_error:{type(exc).__name__}"


# -----------------------------
# Block/file processing
# -----------------------------
def block_is_target(block_name: str, keywords: Sequence[str]) -> bool:
    lower = block_name.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def extract_pose_from_block(source_file: str, block_name: str, block_lines: Sequence[str]) -> Tuple[Optional[PoseRecord], Optional[BlockFailure]]:
    try:
        atom_rows = parse_atom_rows(block_lines)
    except Exception as exc:
        return None, BlockFailure(source_file, block_name, f"parse_atom_rows:{type(exc).__name__}")

    if not atom_rows:
        return None, BlockFailure(source_file, block_name, "no_atom_rows")

    residues = build_residue_blocks(atom_rows)
    if len(residues) < 4:
        return None, BlockFailure(source_file, block_name, "too_few_residues")

    ace_block = residues[0] if residues[0].res_name == "ACE" else None
    nme_block = residues[-1] if residues[-1].res_name == "NME" else None
    if ace_block is None or nme_block is None:
        return None, BlockFailure(source_file, block_name, "missing_ACE_or_NME")

    core_residues = [res for res in residues[1:-1] if res.res_name not in CAP_RESNAMES]
    if not core_residues:
        return None, BlockFailure(source_file, block_name, "no_core_residues")

    first_block = core_residues[0]
    last_block = core_residues[-1]
    pose, err = compute_terminal_pose(ace_block, first_block, last_block, nme_block)
    if pose is None:
        return None, BlockFailure(source_file, block_name, err or "compute_failed")

    pose.source_file = source_file
    pose.block_name = block_name
    return pose, None


def process_one_file(args_tuple: Tuple[Path, Tuple[str, ...]]) -> Tuple[List[PoseRecord], List[BlockFailure]]:
    path, keywords = args_tuple
    poses: List[PoseRecord] = []
    failures: List[BlockFailure] = []

    try:
        with open_cif_text(path) as handle:
            lines = handle.read().splitlines()
    except Exception as exc:
        failures.append(BlockFailure(str(path), "<file>", f"read_error:{type(exc).__name__}"))
        return poses, failures

    try:
        blocks = split_data_blocks(lines)
    except Exception as exc:
        failures.append(BlockFailure(str(path), "<file>", f"split_blocks:{type(exc).__name__}"))
        return poses, failures

    for block_name, block_lines in blocks:
        if not block_is_target(block_name, keywords):
            continue
        pose, failure = extract_pose_from_block(str(path), block_name, block_lines)
        if pose is not None:
            poses.append(pose)
        elif failure is not None:
            failures.append(failure)

    return poses, failures


# -----------------------------
# Output helpers
# -----------------------------
def write_pose_csv(path: Path, poses: Sequence[PoseRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source_file", "block_name", "rho", "psi_rad", "psi_deg", "cos_psi",
            "axis_x", "axis_y", "axis_z", "theta_r_rad", "theta_r_deg",
            "phi_r_rad", "phi_r_deg", "tx", "ty", "tz", "valid_axis"
        ])
        for p in poses:
            writer.writerow([
                p.source_file,
                p.block_name,
                f"{p.rho:.8f}",
                f"{p.psi:.8f}",
                f"{p.psi * DEG:.8f}",
                f"{p.cos_psi:.8f}",
                "" if not np.isfinite(p.axis_x) else f"{p.axis_x:.8f}",
                "" if not np.isfinite(p.axis_y) else f"{p.axis_y:.8f}",
                "" if not np.isfinite(p.axis_z) else f"{p.axis_z:.8f}",
                "" if not np.isfinite(p.theta_r) else f"{p.theta_r:.8f}",
                "" if not np.isfinite(p.theta_r) else f"{p.theta_r * DEG:.8f}",
                "" if not np.isfinite(p.phi_r) else f"{p.phi_r:.8f}",
                "" if not np.isfinite(p.phi_r) else f"{p.phi_r * DEG:.8f}",
                f"{p.tx:.8f}",
                f"{p.ty:.8f}",
                f"{p.tz:.8f}",
                p.valid_axis,
            ])


def write_failure_csv(path: Path, failures: Sequence[BlockFailure]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_file", "block_name", "reason"])
        for f in failures:
            writer.writerow([f.source_file, f.block_name, f.reason])


def write_summary(path: Path, files: Sequence[Path], poses: Sequence[PoseRecord], failures: Sequence[BlockFailure]) -> None:
    rho = np.array([p.rho for p in poses], dtype=np.float64)
    psi = np.array([p.psi for p in poses], dtype=np.float64)
    cos_psi = np.array([p.cos_psi for p in poses], dtype=np.float64)
    valid_axis = np.array([p.valid_axis for p in poses], dtype=np.int32)

    reason_counts: Dict[str, int] = {}
    for f in failures:
        reason_counts[f.reason] = reason_counts.get(f.reason, 0) + 1

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Scanned files: {len(files)}\n")
        fh.write(f"Valid pose records: {len(poses)}\n")
        fh.write(f"Failed / skipped target blocks: {len(failures)}\n")
        fh.write(f"Records with valid rotation axis (psi > {AXIS_EPS:g}): {int(valid_axis.sum())}\n")
        fh.write(f"Records with undefined axis (typically psi ~ 0): {len(poses) - int(valid_axis.sum())}\n")

        if len(poses) > 0:
            fh.write("\n[rho]\n")
            fh.write(f"mean   : {rho.mean():.6f}\n")
            fh.write(f"std    : {rho.std(ddof=0):.6f}\n")
            fh.write(f"min    : {rho.min():.6f}\n")
            fh.write(f"median : {np.median(rho):.6f}\n")
            fh.write(f"max    : {rho.max():.6f}\n")

            fh.write("\n[psi_deg]\n")
            fh.write(f"mean   : {(psi * DEG).mean():.6f}\n")
            fh.write(f"std    : {(psi * DEG).std(ddof=0):.6f}\n")
            fh.write(f"min    : {(psi * DEG).min():.6f}\n")
            fh.write(f"median : {np.median(psi * DEG):.6f}\n")
            fh.write(f"max    : {(psi * DEG).max():.6f}\n")

            fh.write("\n[cos_psi]\n")
            fh.write(f"mean   : {cos_psi.mean():.6f}\n")
            fh.write(f"std    : {cos_psi.std(ddof=0):.6f}\n")
            fh.write(f"min    : {cos_psi.min():.6f}\n")
            fh.write(f"median : {np.median(cos_psi):.6f}\n")
            fh.write(f"max    : {cos_psi.max():.6f}\n")

        if reason_counts:
            fh.write("\n[failures_by_reason]\n")
            for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
                fh.write(f"{reason}: {count}\n")


# -----------------------------
# Plotting
# -----------------------------
def _setup_figure(figsize=(7, 5)):
    fig = plt.figure(figsize=figsize, dpi=180)
    return fig


def plot_rho_hist(path: Path, rho: np.ndarray, bins: int = 80) -> None:
    fig = _setup_figure((7, 5))
    ax = fig.add_subplot(111)
    ax.hist(rho, bins=bins)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of terminal translation distance $\rho$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cos_psi_hist(path: Path, cos_psi: np.ndarray, bins: int = 72) -> None:
    fig = _setup_figure((7, 5))
    ax = fig.add_subplot(111)
    ax.hist(cos_psi, bins=bins, range=(-1.0, 1.0))
    ax.set_xlabel(r"$\cos(\psi)$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $\cos(\psi)$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rho_psi_heatmap(path: Path, rho: np.ndarray, psi: np.ndarray,
                         bins_rho: int = 80, bins_psi: int = 72) -> None:
    fig = _setup_figure((7, 5.5))
    ax = fig.add_subplot(111)
    h = ax.hist2d(rho, psi * DEG, bins=[bins_rho, bins_psi])
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("Count")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title(r"Joint distribution of $\rho$ and $\psi$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_axis_occupancy_mollweide(path: Path, theta: np.ndarray, phi: np.ndarray,
                                  bins_lon: int = 72, bins_lat: int = 36) -> None:
    lat = 0.5 * math.pi - theta  # [-pi/2, pi/2]
    lon = phi                    # [-pi, pi]

    lon_edges = np.linspace(-math.pi, math.pi, bins_lon + 1)
    lat_edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, bins_lat + 1)
    hist, lat_edges_out, lon_edges_out = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])

    Lon, Lat = np.meshgrid(lon_edges_out, lat_edges_out)

    fig = plt.figure(figsize=(9, 5.5), dpi=180)
    ax = fig.add_subplot(111, projection="mollweide")
    mesh = ax.pcolormesh(Lon, Lat, hist, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.08)
    cb.set_label("Count")
    ax.grid(True, alpha=0.5)
    ax.set_title("Rotation-axis occupancy map (Mollweide projection)")
    ax.set_xlabel("Longitude $\\phi_r$")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .cif / .cif.gz files found under: {input_dir}")

    keywords = tuple(args.block_keywords)
    tasks = [(path, keywords) for path in files]

    poses: List[PoseRecord] = []
    failures: List[BlockFailure] = []
    cpus = max(1, args.cpus)

    if cpus == 1:
        for item in tqdm(tasks, total=len(tasks), desc="Processing CIFs", unit="file"):
            file_poses, file_failures = process_one_file(item)
            poses.extend(file_poses)
            failures.extend(file_failures)
    else:
        with mp.Pool(processes=cpus) as pool:
            for file_poses, file_failures in tqdm(
                pool.imap_unordered(process_one_file, tasks, chunksize=1),
                total=len(tasks),
                desc="Processing CIFs",
                unit="file",
            ):
                poses.extend(file_poses)
                failures.extend(file_failures)

    pose_csv = output_dir / "terminal_relative_pose.csv"
    fail_csv = output_dir / "terminal_relative_pose_failures.csv"
    summary_txt = output_dir / "summary.txt"

    write_pose_csv(pose_csv, poses)
    write_failure_csv(fail_csv, failures)
    write_summary(summary_txt, files, poses, failures)

    if not poses:
        print(f"Scanned files: {len(files)}")
        print("No valid target blocks were converted into pose records.")
        print(f"See summary: {summary_txt}")
        return

    rho = np.array([p.rho for p in poses], dtype=np.float64)
    psi = np.array([p.psi for p in poses], dtype=np.float64)
    cos_psi = np.array([p.cos_psi for p in poses], dtype=np.float64)
    valid_axis_mask = np.array([bool(p.valid_axis) for p in poses], dtype=bool)

    plot_rho_hist(output_dir / "rho_hist.png", rho, bins=args.bins_rho)
    plot_cos_psi_hist(output_dir / "cos_psi_hist.png", cos_psi, bins=args.bins_psi)
    plot_rho_psi_heatmap(output_dir / "rho_vs_psi_heatmap.png", rho, psi,
                         bins_rho=args.bins_rho, bins_psi=args.bins_psi)

    if valid_axis_mask.any():
        theta = np.array([p.theta_r for p in poses], dtype=np.float64)[valid_axis_mask]
        phi = np.array([p.phi_r for p in poses], dtype=np.float64)[valid_axis_mask]
        plot_axis_occupancy_mollweide(
            output_dir / "rotation_axis_occupancy_mollweide.png",
            theta,
            phi,
            bins_lon=args.bins_sphere_lon,
            bins_lat=args.bins_sphere_lat,
        )

    print(f"Scanned files: {len(files)}")
    print(f"Valid pose records: {len(poses)}")
    print(f"Failures / skipped target blocks: {len(failures)}")
    print(f"Output directory: {output_dir}")
    print(f"Pose CSV: {pose_csv}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()

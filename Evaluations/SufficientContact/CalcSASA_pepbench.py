#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import gzip
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import rust_sasa_python as sasa


VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}

OUTPUT_COLUMNS = [
    "name",
    "peptide_length",
    "absBSA",
    "relBSA",
    "raBSA",
    "BSA_per_length",
]


def safe_divide(numerator, denominator, default=np.nan):
    denominator = float(denominator)
    if denominator <= 0.0:
        return default
    return float(numerator) / denominator


def strip_known_suffix(filename):
    name = os.path.basename(filename)
    lower = name.lower()
    if lower.endswith(".pdb.gz"):
        return name[:-7]
    if lower.endswith(".pdb"):
        return name[:-4]
    return name


def guess_element_from_atom_name(atom_name):
    name = str(atom_name).strip().upper()
    letters = "".join([c for c in name if c.isalpha()])
    if not letters:
        return "C"
    if letters.startswith("SE"):
        return "SE"
    return letters[0]


def get_elements(atom_array):
    if hasattr(atom_array, "element"):
        raw_elements = np.asarray(atom_array.element).astype(str)
    else:
        raw_elements = np.array([""] * len(atom_array), dtype=str)

    atom_names = np.asarray(atom_array.atom_name).astype(str)
    elements = []

    for elem, atom_name in zip(raw_elements, atom_names):
        elem = elem.strip().upper()
        if elem == "" or elem == "?":
            elem = guess_element_from_atom_name(atom_name)
        elements.append(elem)

    return np.array(elements, dtype=str)


def classify_atom(atom_array):
    elements = get_elements(atom_array)
    atom_names = np.char.upper(np.char.strip(np.asarray(atom_array.atom_name).astype(str)))
    residue_names = np.char.upper(np.char.strip(np.asarray(atom_array.res_name).astype(str)))

    backbone_mask = np.isin(atom_names, ["N", "CA", "C", "O"])
    polar_mask_1 = np.isin(elements, ["N", "O", "S"])
    polar_mask_2 = np.isin(
        atom_names,
        ["H", "HG1", "1HE2", "2HE2", "HE", "HH", "1HH1", "2HH1", "1HH2", "2HH2", "1HZ", "2HZ", "3HZ"],
    )
    polar_mask_3 = np.isin(residue_names, ["CYS", "SER"]) & np.isin(atom_names, ["HG"])
    polar_mask_4 = np.isin(residue_names, ["ASN"]) & np.isin(atom_names, ["1HD2", "2HD2"])
    polar_mask_5 = np.isin(residue_names, ["HIS"]) & np.isin(atom_names, ["HD1", "HE2"])
    polar_mask = polar_mask_1 | polar_mask_2 | polar_mask_3 | polar_mask_4 | polar_mask_5
    apolar_mask = ~polar_mask

    assert polar_mask.shape == atom_array.shape, f"polar_mask shape: {polar_mask.shape} is not equal to atomarray shape: {atom_array.shape}"
    assert apolar_mask.shape == atom_array.shape, f"apolar_mask shape: {apolar_mask.shape} is not equal to atomarray shape: {atom_array.shape}"

    return {
        "backbone": backbone_mask,
        "polar": polar_mask,
        "apolar": apolar_mask,
    }


def load_atoms(pdb_path):
    if pdb_path.lower().endswith(".gz"):
        with gzip.open(pdb_path, "rt") as f:
            pdb_file = pdb.PDBFile.read(f)
    else:
        pdb_file = pdb.PDBFile.read(pdb_path)

    atoms = pdb_file.get_structure(model=1)

    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]

    if len(atoms) == 0:
        raise ValueError(f"Empty structure: {pdb_path}")

    return atoms


def make_positive_residue_labels(atom_array):
    chain_ids = np.asarray(atom_array.chain_id).astype(str)
    res_ids = np.asarray(atom_array.res_id).astype(int)
    res_names = np.asarray(atom_array.res_name).astype(str)

    if hasattr(atom_array, "ins_code"):
        ins_codes = np.asarray(atom_array.ins_code).astype(str)
    else:
        ins_codes = np.array([""] * len(atom_array), dtype=str)

    if hasattr(atom_array, "hetero"):
        hetero = np.asarray(atom_array.hetero).astype(bool)
    else:
        hetero = np.zeros(len(atom_array), dtype=bool)

    keys = [
        (chain_ids[i], int(res_ids[i]), ins_codes[i], res_names[i], bool(hetero[i]))
        for i in range(len(atom_array))
    ]

    label_map = {}
    labels = np.empty(len(atom_array), dtype=np.int64)
    next_label = 1

    for i, key in enumerate(keys):
        if key not in label_map:
            label_map[key] = next_label
            next_label += 1
        labels[i] = label_map[key]

    return labels


def count_residues(atom_array):
    return int(len(np.unique(make_positive_residue_labels(atom_array))))


def compute_chain_sasa(chain_mask, all_atoms, probe_radius=1.4, n_points=200):
    chain_mask = np.asarray(chain_mask, dtype=bool)
    if len(chain_mask) != len(all_atoms):
        raise ValueError("Length mismatch: chain_mask vs all_atoms")
    if not np.any(chain_mask):
        raise ValueError("Empty chain selection")

    coords = np.asarray(all_atoms.coord, dtype=float)
    residue_labels = make_positive_residue_labels(all_atoms)
    elements = np.char.upper(np.char.strip(get_elements(all_atoms).astype(str)))
    radii = np.array([VDW_RADII.get(elem, 1.70) for elem in elements], dtype=float)

    atoms_input = list(
        zip(
            map(tuple, coords.tolist()),
            radii.tolist(),
            residue_labels.astype(int).tolist(),
        )
    )

    atom_sasa_all = np.asarray(
        sasa.calculate_sasa_internal(
            atoms_input,
            probe_radius=probe_radius,
            n_points=n_points,
        ),
        dtype=float,
    )

    atom_sasa_chain = atom_sasa_all[chain_mask]
    residue_labels_chain = residue_labels[chain_mask]

    chain_atoms = all_atoms[chain_mask]
    classes = classify_atom(chain_atoms)
    is_apolar = np.asarray(classes["apolar"], dtype=bool)
    is_backbone = np.asarray(classes["backbone"], dtype=bool)

    total_atom = atom_sasa_chain
    apolar_atom = atom_sasa_chain * is_apolar.astype(float)
    sidechain_atom = atom_sasa_chain * (~is_backbone).astype(float)

    _, inverse = np.unique(residue_labels_chain, return_inverse=True)
    total_list = np.bincount(inverse, weights=total_atom)
    apolar_list = np.bincount(inverse, weights=apolar_atom)
    sidechain_list = np.bincount(inverse, weights=sidechain_atom)

    return total_list, apolar_list, sidechain_list


def process_one(pdb_path, ligand_chain="L", receptor_chain="R", probe_radius=1.4, n_points=200):
    name = strip_known_suffix(pdb_path)
    result = {col: "" for col in OUTPUT_COLUMNS}
    result["name"] = name

    try:
        atoms = load_atoms(pdb_path)
        chain_ids = np.asarray(atoms.chain_id).astype(str)

        ligand_mask_all = (chain_ids == ligand_chain)
        receptor_mask_all = (chain_ids == receptor_chain)

        if not np.any(ligand_mask_all):
            raise ValueError(f"Missing ligand chain '{ligand_chain}'")
        if not np.any(receptor_mask_all):
            raise ValueError(f"Missing receptor chain '{receptor_chain}'")

        # 只保留 L / R 两条链参与计算
        complex_mask = ligand_mask_all | receptor_mask_all
        complex_atoms = atoms[complex_mask]

        complex_chain_ids = np.asarray(complex_atoms.chain_id).astype(str)
        ligand_mask = (complex_chain_ids == ligand_chain)
        receptor_mask = (complex_chain_ids == receptor_chain)

        ligand_atoms = complex_atoms[ligand_mask]
        receptor_atoms = complex_atoms[receptor_mask]

        ligand_length = count_residues(ligand_atoms)

        # L 链 unbound / bound
        ligand_total_u, ligand_apolar_u, ligand_sidechain_u = compute_chain_sasa(
            np.ones(len(ligand_atoms), dtype=bool),
            ligand_atoms,
            probe_radius=probe_radius,
            n_points=n_points,
        )
        ligand_total_b, ligand_apolar_b, ligand_sidechain_b = compute_chain_sasa(
            ligand_mask,
            complex_atoms,
            probe_radius=probe_radius,
            n_points=n_points,
        )

        ligand_total_sasa_unbound = float(np.sum(ligand_total_u))
        ligand_total_sasa_bound = float(np.sum(ligand_total_b))

        abs_bsa = ligand_total_sasa_unbound - ligand_total_sasa_bound
        rel_bsa = safe_divide(abs_bsa, ligand_total_sasa_unbound, default=0.0)

        ligand_apolar_sasa_unbound = float(np.sum(ligand_apolar_u))
        ligand_apolar_sasa_bound = float(np.sum(ligand_apolar_b))
        apolar_bsa = ligand_apolar_sasa_unbound - ligand_apolar_sasa_bound
        ra_bsa = safe_divide(apolar_bsa, abs_bsa, default=0.0)

        bsa_per_length = safe_divide(abs_bsa, ligand_length, default=0.0)

        result.update({
            "peptide_length": ligand_length,
            "absBSA": abs_bsa,
            "relBSA": rel_bsa,
            "raBSA": ra_bsa,
            "BSA_per_length": bsa_per_length,
        })

    except Exception as e:
        print(f"[Error] {pdb_path}: {type(e).__name__}: {e}")

    return result


def list_input_pdbs(input_dir):
    paths = []
    for item in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, item)
        if not os.path.isfile(path):
            continue
        lower = item.lower()
        if lower.endswith(".pdb") or lower.endswith(".pdb.gz"):
            paths.append(path)
    return paths


def main(input_dir, output_csv, n_cpu, ligand_chain="L", receptor_chain="R", probe_radius=1.4, n_points=200):
    pdb_paths = list_input_pdbs(input_dir)

    output_parent = os.path.dirname(os.path.abspath(output_csv))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    worker = partial(
        process_one,
        ligand_chain=ligand_chain,
        receptor_chain=receptor_chain,
        probe_radius=probe_radius,
        n_points=n_points,
    )

    with open(output_csv, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(OUTPUT_COLUMNS)
        out_handle.flush()

        with mp.Pool(processes=n_cpu) as pool:
            for result in tqdm(pool.imap_unordered(worker, pdb_paths), total=len(pdb_paths)):
                writer.writerow([result.get(col, "") for col in OUTPUT_COLUMNS])
                out_handle.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute BSA/SASA features between ligand chain L and receptor chain R from a folder of .pdb/.pdb.gz files"
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing many .pdb or .pdb.gz files",
    )
    parser.add_argument(
        "-o", "--output_csv",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "-n", "--n_cpu",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--ligand_chain",
        type=str,
        default="L",
        help="Ligand chain ID (default: L)",
    )
    parser.add_argument(
        "--receptor_chain",
        type=str,
        default="R",
        help="Receptor chain ID (default: R)",
    )
    parser.add_argument(
        "--probe_radius",
        type=float,
        default=1.4,
        help="Probe radius for SASA calculation",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=200,
        help="Number of sphere points for rust_sasa",
    )

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        n_cpu=args.n_cpu,
        ligand_chain=args.ligand_chain,
        receptor_chain=args.receptor_chain,
        probe_radius=args.probe_radius,
        n_points=args.n_points,
    )


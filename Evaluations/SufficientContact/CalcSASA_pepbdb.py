#!/usr/bin/env python3

import os
import csv
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm
import biotite.structure as struc
import biotite.structure.io as strucio
import rust_sasa_python as sasa


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
}

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}
APOLAR_ELEMENTS = {"C"}

OUTPUT_COLUMNS = [
    "name",
    "absBSA",
    "relSASA",
    "var",
    "max_run",
    "apolar_BSA",
    "rel_apolar_BSA",
    "polar_BSA",
    "rel_polar_BSA",
    "sidechain_BSA",
    "rel_sidechain_BSA",
    "peptide_length",
    "receptor_length",
    "peptide_SASA_unbound",
    "receptor_SASA_unbound",
]


def safe_divide(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0.0:
        return np.nan
    return float(numerator) / denominator


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

    is_apolar = np.isin(elements, list(APOLAR_ELEMENTS))
    is_backbone = np.isin(atom_names, list(BACKBONE_ATOM_NAMES))

    return {
        "apolar": is_apolar,
        "backbone": is_backbone,
    }


def load_atoms(pdb_path):
    atoms = strucio.load_structure(pdb_path)
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
    return int(np.unique(make_positive_residue_labels(atom_array)).size)


def compute_chain_sasa(chain_atoms, all_atoms, atom_start=0, probe_radius=1.4, n_points=200):
    coords = np.asarray(all_atoms.coord, dtype=float)
    residue_labels = make_positive_residue_labels(all_atoms)
    elements = get_elements(all_atoms)

    radii = np.array([VDW_RADII.get(elem, 1.70) for elem in elements], dtype=float)

    atoms_input = list(
        zip(
            map(tuple, coords.tolist()),
            radii.tolist(),
            residue_labels.tolist(),
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

    chain_len = len(chain_atoms)
    atom_end = atom_start + chain_len

    if chain_len > len(all_atoms):
        raise ValueError("chain_atoms is longer than all_atoms")
    if atom_start < 0 or atom_end > len(all_atoms):
        raise ValueError("Invalid atom range for chain_atoms in all_atoms")

    mask = np.zeros(len(all_atoms), dtype=bool)
    mask[atom_start:atom_end] = True

    sasa_chain = atom_sasa_all[mask]
    residue_labels_chain = residue_labels[mask]

    classes = classify_atom(chain_atoms)
    is_apolar = np.asarray(classes["apolar"], dtype=bool)
    is_backbone = np.asarray(classes["backbone"], dtype=bool)

    if len(sasa_chain) != len(is_apolar) or len(sasa_chain) != len(is_backbone):
        raise ValueError("Length mismatch between chain SASA and atom classification")

    total_atom = sasa_chain
    apolar_atom = sasa_chain * is_apolar.astype(float)
    sidechain_atom = sasa_chain * (~is_backbone).astype(float)

    _, inverse = np.unique(residue_labels_chain, return_inverse=True)
    total_list = np.bincount(inverse, weights=total_atom)
    apolar_list = np.bincount(inverse, weights=apolar_atom)
    sidechain_list = np.bincount(inverse, weights=sidechain_atom)

    return total_list, apolar_list, sidechain_list


def process_one(subdir_path, probe_radius=1.4, n_points=200):
    name = os.path.basename(os.path.normpath(subdir_path))
    peptide_path = os.path.join(subdir_path, "peptide.pdb")
    receptor_path = os.path.join(subdir_path, "receptor.pdb")

    result = {col: "" for col in OUTPUT_COLUMNS}
    result["name"] = name

    try:
        if not os.path.isfile(peptide_path):
            raise FileNotFoundError(f"Missing file: {peptide_path}")
        if not os.path.isfile(receptor_path):
            raise FileNotFoundError(f"Missing file: {receptor_path}")

        peptide_atoms = load_atoms(peptide_path)
        receptor_atoms = load_atoms(receptor_path)

        peptide_length = count_residues(peptide_atoms)
        receptor_length = count_residues(receptor_atoms)

        # Fixed definition:
        # peptide.pdb  -> ligand / peptide
        # receptor.pdb -> receptor
        complex_atoms = peptide_atoms + receptor_atoms

        # Peptide SASA
        total_u, apolar_u, sidechain_u = compute_chain_sasa(
            peptide_atoms,
            peptide_atoms,
            atom_start=0,
            probe_radius=probe_radius,
            n_points=n_points,
        )
        total_b, apolar_b, sidechain_b = compute_chain_sasa(
            peptide_atoms,
            complex_atoms,
            atom_start=0,
            probe_radius=probe_radius,
            n_points=n_points,
        )

        # Receptor SASA
        total_r_u, apolar_r_u, sidechain_r_u = compute_chain_sasa(
            receptor_atoms,
            receptor_atoms,
            atom_start=0,
            probe_radius=probe_radius,
            n_points=n_points,
        )
        total_r_b, apolar_r_b, sidechain_r_b = compute_chain_sasa(
            receptor_atoms,
            complex_atoms,
            atom_start=len(peptide_atoms),
            probe_radius=probe_radius,
            n_points=n_points,
        )

        peptide_residue_bsa = total_u - total_b
        peptide_residue_bsa = np.clip(peptide_residue_bsa, 0.0, None)
        peptide_residue_rbsa = peptide_residue_bsa / np.clip(total_u, 1e-8, None)
        x = np.clip(peptide_residue_rbsa, 0.0, 1.0)

        window = 3
        kernel = np.ones(window) / window
        pad = window // 2
        x_pad = np.pad(x, (pad, pad), mode='edge')
        x_smooth = np.convolve(x_pad, kernel, mode='valid')

        rel_bsa_var = np.var(x_smooth)

        thr = 0.2
        mask = (x < thr)
        max_run = 0
        cur = 0
        for v in mask:
            if v:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0

        peptide_total_sasa_unbound = float(np.sum(total_u))
        peptide_total_sasa_bound = float(np.sum(total_b))
        abs_bsa = peptide_total_sasa_unbound - peptide_total_sasa_bound

        receptor_total_sasa_unbound = float(np.sum(total_r_u))

        # relSASA is now always relative to peptide unbound SASA
        rel_sasa = safe_divide(abs_bsa, peptide_total_sasa_unbound)

        peptide_apolar_sasa_unbound = float(np.sum(apolar_u))
        peptide_apolar_sasa_bound = float(np.sum(apolar_b))
        apolar_bsa = peptide_apolar_sasa_unbound - peptide_apolar_sasa_bound
        rel_apolar_bsa = safe_divide(apolar_bsa, abs_bsa)

        peptide_sidechain_sasa_unbound = float(np.sum(sidechain_u))
        peptide_sidechain_sasa_bound = float(np.sum(sidechain_b))
        sidechain_bsa = peptide_sidechain_sasa_unbound - peptide_sidechain_sasa_bound
        rel_sidechain_bsa = safe_divide(sidechain_bsa, abs_bsa)

        peptide_polar_sasa_unbound = peptide_total_sasa_unbound - peptide_apolar_sasa_unbound
        peptide_polar_sasa_bound = peptide_total_sasa_bound - peptide_apolar_sasa_bound
        polar_bsa = peptide_polar_sasa_unbound - peptide_polar_sasa_bound
        rel_polar_bsa = safe_divide(polar_bsa, abs_bsa)

        result.update({
            "absBSA": abs_bsa,
            "relSASA": rel_sasa,
            "var": rel_bsa_var,
            "max_run": max_run,
            "apolar_BSA": apolar_bsa,
            "rel_apolar_BSA": rel_apolar_bsa,
            "polar_BSA": polar_bsa,
            "rel_polar_BSA": rel_polar_bsa,
            "sidechain_BSA": sidechain_bsa,
            "rel_sidechain_BSA": rel_sidechain_bsa,
            "peptide_length": peptide_length,
            "receptor_length": receptor_length,
            "peptide_SASA_unbound": peptide_total_sasa_unbound,
            "receptor_SASA_unbound": receptor_total_sasa_unbound,
        })

    except Exception as e:
        print(f"[{name}] {type(e).__name__}: {e}")

    return result


def callback(return_data, writer, out_handle, tbar):
    if return_data is not None:
        writer.writerow([return_data.get(col, "") for col in OUTPUT_COLUMNS])
        out_handle.flush()
    tbar.update(1)


def main(input_dir, output_csv, n_cpu, probe_radius=1.4, n_points=200):
    subdirs = []
    for item in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, item)
        if os.path.isdir(path):
            subdirs.append(path)

    output_parent = os.path.dirname(os.path.abspath(output_csv))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(output_csv, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(OUTPUT_COLUMNS)
        out_handle.flush()

        tbar = tqdm(total=len(subdirs))
        partial_callback = partial(
            callback,
            writer=writer,
            out_handle=out_handle,
            tbar=tbar,
        )

        with mp.Pool(processes=n_cpu) as pool:
            for subdir_path in subdirs:
                pool.apply_async(
                    func=process_one,
                    args=(subdir_path, probe_radius, n_points),
                    callback=partial_callback,
                )
            pool.close()
            pool.join()

        tbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute peptide SASA/BSA features for subdirectories containing peptide.pdb and receptor.pdb"
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Root directory containing many subdirectories",
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
        probe_radius=args.probe_radius,
        n_points=args.n_points,
    )
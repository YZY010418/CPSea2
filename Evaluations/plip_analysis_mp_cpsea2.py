#!/usr/bin/env python3

import csv
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import gemmi
except ImportError as e:
    raise ImportError(
        "This script requires gemmi. Please install it with: pip install gemmi"
    ) from e


FIXED_HEADERS = [
    "Input",
    "hydrophobic_interactions",
    "hydrogen_bonds",
    "water_bridges",
    "salt_bridges",
    "pi_stacks",
    "pi_cation_interactions",
    "halogen_bonds",
    "metal_complexes",
]


def count_interactions(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        interaction_counts = {}
        interactions = root.find("bindingsite/interactions")
        if interactions is not None:
            for child in interactions:
                interaction_counts[child.tag] = len(child.findall("./*"))
        return interaction_counts
    except Exception as e:
        print(f"Error processing {xml_file}: {e}", file=sys.stderr)
        return {}


def standardize_interaction_types(interaction_counts):
    aliases = {
        "hydrophobic": "hydrophobic_interactions",
        "hydrophobicinteraction": "hydrophobic_interactions",
        "hbond": "hydrogen_bonds",
        "hbondinteraction": "hydrogen_bonds",
        "hbonds": "hydrogen_bonds",
        "waterbridge": "water_bridges",
        "waterbridges": "water_bridges",
        "saltbridge": "salt_bridges",
        "saltbridges": "salt_bridges",
        "pistack": "pi_stacks",
        "pistacks": "pi_stacks",
        "pication": "pi_cation_interactions",
        "picationinteraction": "pi_cation_interactions",
        "halogenbond": "halogen_bonds",
        "halogenbonds": "halogen_bonds",
        "metalcomplex": "metal_complexes",
        "metalcomplexes": "metal_complexes",
    }

    standardized = {}
    for key, value in interaction_counts.items():
        normalized_key = key.lower().replace("_", "").replace("-", "")
        standardized[aliases.get(normalized_key, key)] = value
    return standardized


def strip_cif_gz_suffix(path):
    name = os.path.basename(path)
    for suffix in (".cif.gz", ".mmcif.gz", ".cif", ".mmcif", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def normalize_ins_code(value):
    if value in (None, "", " ", ".", "?"):
        return ""
    return str(value).strip()


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def safe_seqnum(residue):
    try:
        return int(residue.seqid.num)
    except Exception:
        return None


def safe_icode(residue):
    try:
        return normalize_ins_code(residue.seqid.icode)
    except Exception:
        return ""


def atom_altloc_rank(atom):
    altloc = atom.altloc
    if altloc == "\x00":
        altloc = ""
    priority = 0 if altloc in ("", "A") else 1
    try:
        occ = float(atom.occ)
    except Exception:
        occ = 0.0
    return (priority, -occ, altloc)


def choose_atoms_from_residue(residue):
    atom_groups = defaultdict(list)
    for atom in residue:
        if atom.is_hydrogen():
            continue
        atom_groups[atom.name].append(atom)

    chosen = []
    for atoms in atom_groups.values():
        chosen.append(sorted(atoms, key=atom_altloc_rank)[0])

    chosen.sort(key=lambda x: x.serial if getattr(x, "serial", 0) else 0)
    return chosen


def residue_candidate_keys(chain, residue):
    chain_name = getattr(chain, "name", "") or ""
    seqnum = safe_seqnum(residue)
    icode = safe_icode(residue)
    keys = set()

    if seqnum is not None:
        keys.add(("auth", chain_name, seqnum, icode))
        keys.add(("label", chain_name, seqnum))

    label_seq = safe_int(getattr(residue, "label_seq", None))
    if label_seq is not None:
        keys.add(("label", chain_name, label_seq))

    subchain = getattr(residue, "subchain", None)
    if subchain and seqnum is not None:
        keys.add(("label", str(subchain), seqnum))
    if subchain and label_seq is not None:
        keys.add(("label", str(subchain), label_seq))

    return keys


def collect_model_residues(model):
    residues = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            atoms = choose_atoms_from_residue(residue)
            if not atoms:
                continue
            residues.append(
                {
                    "chain_name": getattr(chain, "name", "") or "",
                    "seqnum": safe_seqnum(residue),
                    "icode": safe_icode(residue),
                    "label_seq": safe_int(getattr(residue, "label_seq", None)),
                    "subchain": getattr(residue, "subchain", "") or "",
                    "resname": residue.name,
                    "atoms": atoms,
                    "candidate_keys": residue_candidate_keys(chain, residue),
                }
            )
    return residues


def format_pdb_atom_line(
    record_name,
    atom_serial,
    atom_name,
    altloc,
    resname,
    chain_id,
    resseq,
    x,
    y,
    z,
    occupancy,
    bfactor,
    element,
):
    atom_name = (atom_name or "")[:4]
    resname = (resname or "UNK")[:3]
    chain_id = (chain_id or "A")[:1]
    altloc = "" if altloc in ("\x00", None) else str(altloc)[:1]
    element = (str(element) if element is not None else "").strip()[:2].rjust(2)
    atom_name_field = atom_name.rjust(4)

    return (
        f"{record_name:<6}"
        f"{atom_serial:>5} "
        f"{atom_name_field}"
        f"{altloc:1}"
        f"{resname:>3} "
        f"{chain_id:1}"
        f"{resseq:>4}"
        f"    "
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{occupancy:>6.2f}"
        f"{bfactor:>6.2f}"
        f"          "
        f"{element}"
    )


def write_residue_block(out_fh, residues, new_chain_id, start_resseq, start_atom_serial):
    next_resseq = start_resseq
    next_atom_serial = start_atom_serial
    last_resname = "UNK"

    for res in residues:
        resname = (res.get("resname") or "UNK").strip() or "UNK"
        atoms = res.get("atoms", [])
        if not atoms:
            continue

        for atom in atoms:
            try:
                occ = float(atom.occ)
            except Exception:
                occ = 1.0
            try:
                b_iso = float(atom.b_iso)
            except Exception:
                b_iso = 0.0

            out_fh.write(
                format_pdb_atom_line(
                    record_name="ATOM",
                    atom_serial=next_atom_serial,
                    atom_name=atom.name,
                    altloc=atom.altloc,
                    resname=resname,
                    chain_id=new_chain_id,
                    resseq=next_resseq,
                    x=float(atom.pos.x),
                    y=float(atom.pos.y),
                    z=float(atom.pos.z),
                    occupancy=occ,
                    bfactor=b_iso,
                    element=atom.element.name,
                )
                + "\n"
            )
            next_atom_serial += 1

        last_resname = resname
        next_resseq += 1

    if next_atom_serial > start_atom_serial:
        out_fh.write(
            f"TER   {next_atom_serial:>5}      {last_resname:>3} "
            f"{new_chain_id}{(next_resseq - 1):>4}\n"
        )
        next_atom_serial += 1

    return next_resseq, next_atom_serial


def load_coordinate_units(cif_path):
    doc = gemmi.cif.read(cif_path)
    units = []

    for block_idx, block in enumerate(doc, start=1):
        atom_site = block.find_mmcif_category("_atom_site.")
        if not atom_site:
            continue

        structure = gemmi.make_structure_from_block(block)
        if len(structure) == 0:
            continue

        for model_idx, model in enumerate(structure, start=1):
            unit_tag = block.name
            if len(structure) > 1:
                unit_tag = f"{block.name}_model_{model.num}"

            units.append(
                {
                    "block_index": block_idx,
                    "block_name": block.name,
                    "model_index": model_idx,
                    "model_num": str(model.num),
                    "model": model,
                    "unit_tag": unit_tag,
                }
            )

    if not units:
        raise ValueError(f"No coordinate-bearing data blocks found in {cif_path}")

    return units, doc


def parse_receptor_ligand_map(receptor_block):
    table = receptor_block.find("_cpsea_receptor_ligand_map.", [
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
        "label_asym_id",
        "label_seq_id",
        "ligand_ids",
    ])
    if table is None:
        raise ValueError("Missing _cpsea_receptor_ligand_map category in receptor block")

    ligand_to_residues = defaultdict(set)
    for row in table:
        auth_asym_id, auth_seq_id, ins_code, label_asym_id, label_seq_id, ligand_ids = row
        auth_seq_id = safe_int(auth_seq_id)
        label_seq_id = safe_int(label_seq_id)
        ins_code = normalize_ins_code(ins_code)

        ligand_indices = []
        for token in str(ligand_ids).split(","):
            token = token.strip()
            if not token:
                continue
            ligand_idx = safe_int(token)
            if ligand_idx is not None:
                ligand_indices.append(ligand_idx)

        residue_keys = set()
        if auth_seq_id is not None:
            residue_keys.add(("auth", str(auth_asym_id), auth_seq_id, ins_code))
        if label_seq_id is not None:
            residue_keys.add(("label", str(label_asym_id), label_seq_id))

        for ligand_idx in ligand_indices:
            ligand_to_residues[ligand_idx].update(residue_keys)

    if not ligand_to_residues:
        raise ValueError("No usable rows found in _cpsea_receptor_ligand_map")

    return ligand_to_residues


def select_receptor_residues(receptor_residues_all, ligand_index, receptor_ligand_map):
    expected_keys = receptor_ligand_map.get(ligand_index, set())
    if not expected_keys:
        raise ValueError(
            f"No receptor residues defined for ligand index {ligand_index} in "
            "_cpsea_receptor_ligand_map"
        )

    selected_residues = []
    for residue in receptor_residues_all:
        if residue["candidate_keys"] & expected_keys:
            selected_residues.append(residue)

    if not selected_residues:
        raise ValueError(
            f"Failed to match receptor residues for ligand index {ligand_index} "
            "using _cpsea_receptor_ligand_map"
        )

    return selected_residues


def build_complex_pdb(
    cif_path,
    units,
    receptor_ligand_map,
    ligand_index,
    out_pdb_path,
):
    if len(units) < 2:
        raise ValueError(
            f"{cif_path} must contain at least one receptor block and one ligand block"
        )

    receptor_unit = units[0]
    ligand_unit_position = ligand_index
    if ligand_unit_position >= len(units):
        raise ValueError(
            f"Ligand index {ligand_index} is out of range for {cif_path} "
            f"with {len(units) - 1} ligand blocks"
        )

    ligand_unit = units[ligand_unit_position]
    receptor_residues_all = collect_model_residues(receptor_unit["model"])
    ligand_residues = collect_model_residues(ligand_unit["model"])
    receptor_residues = select_receptor_residues(
        receptor_residues_all=receptor_residues_all,
        ligand_index=ligand_index,
        receptor_ligand_map=receptor_ligand_map,
    )

    if not ligand_residues:
        raise ValueError(
            f"No valid ligand residues found for ligand index {ligand_index} in {cif_path}"
        )

    out_dir = os.path.dirname(out_pdb_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_pdb_path, "w") as out_fh:
        out_fh.write(
            f"REMARK source={os.path.basename(cif_path)} receptor={receptor_unit['unit_tag']} "
            f"ligand={ligand_unit['unit_tag']} ligand_index={ligand_index}\n"
        )
        next_resseq, next_atom_serial = write_residue_block(
            out_fh=out_fh,
            residues=receptor_residues,
            new_chain_id="A",
            start_resseq=1,
            start_atom_serial=1,
        )
        write_residue_block(
            out_fh=out_fh,
            residues=ligand_residues,
            new_chain_id="L",
            start_resseq=next_resseq,
            start_atom_serial=next_atom_serial,
        )
        out_fh.write("END\n")

    ligand_seqnums = [
        res.get("seqnum") for res in ligand_residues if res.get("seqnum") is not None
    ]

    return {
        "sample_name": strip_cif_gz_suffix(cif_path),
        "complex_pdb": out_pdb_path,
        "ligand_unit_tag": ligand_unit["unit_tag"],
        "receptor_unit_tag": receptor_unit["unit_tag"],
        "ligand_res_count": len(ligand_residues),
        "receptor_res_count": len(receptor_residues),
        "ligand_seqnums": ligand_seqnums,
    }


def run_plip_and_parse(complex_pdb, out_dir):
    xml_file = os.path.join(out_dir, "complex_plip_report.xml")
    cmd = [
        "plip",
        "-f",
        complex_pdb,
        "--peptides",
        "L",
        "-x",
        "-o",
        out_dir,
        "--silent",
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if completed.returncode != 0:
        return {}, f"PLIP command failed (return code {completed.returncode})"
    if not os.path.exists(xml_file):
        return {}, "PLIP XML report not found"
    return count_interactions(xml_file), None


def create_result_row(base_name, interaction_counts):
    standardized_counts = standardize_interaction_types(interaction_counts)
    row = [base_name]
    for header in FIXED_HEADERS[1:]:
        row.append(standardized_counts.get(header, 0))
    return row


def process_single_ligand(
    cif_path,
    sample_name,
    units,
    receptor_ligand_map,
    ligand_index,
    tmp_root,
    keep_complexes,
    complex_cache_dir,
):
    ligand_unit = units[ligand_index]
    ligand_unit_tag = ligand_unit["unit_tag"]
    unique_tag = f"{sample_name}_{ligand_unit_tag}_{uuid.uuid4().hex[:10]}"
    temp_items = []

    if keep_complexes:
        if complex_cache_dir:
            complex_dir = os.path.join(complex_cache_dir, sample_name)
        else:
            complex_dir = os.path.dirname(os.path.abspath(cif_path))
        os.makedirs(complex_dir, exist_ok=True)
        complex_pdb = os.path.join(
            complex_dir,
            f"{sample_name}.{ligand_unit_tag}.complex_plip.pdb",
        )
        work_dir = os.path.join(tmp_root, unique_tag)
        os.makedirs(work_dir, exist_ok=True)
        temp_items.append(work_dir)
    else:
        work_dir = os.path.join(tmp_root, unique_tag)
        os.makedirs(work_dir, exist_ok=True)
        complex_pdb = os.path.join(work_dir, "complex_plip.pdb")
        temp_items.append(work_dir)

    try:
        build_info = build_complex_pdb(
            cif_path=cif_path,
            units=units,
            receptor_ligand_map=receptor_ligand_map,
            ligand_index=ligand_index,
            out_pdb_path=complex_pdb,
        )
        interaction_counts, err = run_plip_and_parse(complex_pdb, work_dir)

        if err is not None:
            print(
                f"Warning: {sample_name} ligand_index={ligand_index} "
                f"ligand_unit={ligand_unit_tag} failed: {err}",
                file=sys.stderr,
            )

        return {
            "csv_row": create_result_row(
                f"{sample_name}__ligand{ligand_index}", interaction_counts
            ),
            "log_item": {
                "file_name": f"{sample_name}__ligand{ligand_index}",
                "source_file": os.path.basename(cif_path),
                "ligand_index": ligand_index,
                "ligand_unit_tag": ligand_unit_tag,
                "receptor_unit_tag": build_info["receptor_unit_tag"],
                "ligand_res_count": build_info["ligand_res_count"],
                "receptor_res_count": build_info["receptor_res_count"],
                "ligand_seqnum_min": min(build_info["ligand_seqnums"])
                if build_info["ligand_seqnums"]
                else "",
                "ligand_seqnum_max": max(build_info["ligand_seqnums"])
                if build_info["ligand_seqnums"]
                else "",
                "total_interactions": sum(
                    standardize_interaction_types(interaction_counts).values()
                ),
            },
        }
    except Exception as e:
        print(
            f"Error processing {sample_name} ligand_index={ligand_index} "
            f"ligand_unit={ligand_unit_tag}: {e}",
            file=sys.stderr,
        )
        return {
            "csv_row": create_result_row(f"{sample_name}__ligand{ligand_index}", {}),
            "log_item": {
                "file_name": f"{sample_name}__ligand{ligand_index}",
                "source_file": os.path.basename(cif_path),
                "ligand_index": ligand_index,
                "ligand_unit_tag": ligand_unit_tag,
                "receptor_unit_tag": units[0]["unit_tag"],
                "ligand_res_count": 0,
                "receptor_res_count": 0,
                "ligand_seqnum_min": "",
                "ligand_seqnum_max": "",
                "total_interactions": 0,
            },
        }
    finally:
        for path in temp_items:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)


def process_cif_file(
    cif_path,
    tmp_root="./tmp3",
    keep_complexes=False,
    complex_cache_dir=None,
    run_all_mode=True,
    run_random_mode=True,
):
    sample_name = strip_cif_gz_suffix(cif_path)
    try:
        units, doc = load_coordinate_units(cif_path)
        if len(units) < 2:
            print(
                f"Warning: skip {cif_path} because it has fewer than 2 coordinate blocks",
                file=sys.stderr,
            )
            return {"all_rows": [], "random_row": None, "log_items": []}

        receptor_block = doc[0]
        receptor_ligand_map = parse_receptor_ligand_map(receptor_block)

        ligand_indices = list(range(1, len(units)))
        random_ligand_index = random.choice(ligand_indices) if ligand_indices else None
        selected_indices = []
        if run_all_mode:
            selected_indices.extend(ligand_indices)
        elif run_random_mode and random_ligand_index is not None:
            selected_indices.append(random_ligand_index)

        selected_results = []
        for ligand_index in selected_indices:
            selected_results.append(
                process_single_ligand(
                    cif_path=cif_path,
                    sample_name=sample_name,
                    units=units,
                    receptor_ligand_map=receptor_ligand_map,
                    ligand_index=ligand_index,
                    tmp_root=tmp_root,
                    keep_complexes=keep_complexes,
                    complex_cache_dir=complex_cache_dir,
                )
            )

        all_rows = [item["csv_row"] for item in selected_results] if run_all_mode else []
        random_row = None
        if run_random_mode and random_ligand_index is not None:
            if run_all_mode:
                result_by_index = {
                    item["log_item"]["ligand_index"]: item for item in selected_results
                }
                random_result = result_by_index.get(random_ligand_index)
                random_row = random_result["csv_row"] if random_result else None
            elif selected_results:
                random_row = selected_results[0]["csv_row"]

        return {
            "all_rows": all_rows,
            "random_row": random_row,
            "log_items": [item["log_item"] for item in selected_results],
        }
    except Exception as e:
        print(f"Error processing {cif_path}: {e}", file=sys.stderr)
        return {"all_rows": [], "random_row": None, "log_items": []}


def discover_cif_gz_files(input_dir):
    cif_files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith((".cif.gz", ".mmcif.gz", ".cif", ".mmcif")):
            cif_files.append(path)
    return cif_files


def write_log_entries(log_file, log_items):
    if not log_file:
        return
    for item in log_items:
        log_file.write(
            f"Processed: {item['file_name']} | "
            f"source={item['source_file']} | "
            f"ligand_index={item['ligand_index']} | "
            f"ligand_unit={item['ligand_unit_tag']} | "
            f"receptor_unit={item['receptor_unit_tag']} | "
            f"receptor_res={item['receptor_res_count']} | "
            f"ligand_res={item['ligand_res_count']} | "
            f"seq_range=({item['ligand_seqnum_min']}, {item['ligand_seqnum_max']}) | "
            f"total_interactions={item['total_interactions']}\n"
        )
    log_file.flush()


def callback(return_data, tbar, all_writer, random_writer, log_file):
    if return_data:
        for row in return_data.get("all_rows", []):
            if all_writer:
                all_writer.writerow(row)
        random_row = return_data.get("random_row")
        if random_row and random_writer:
            random_writer.writerow(random_row)
        write_log_entries(log_file, return_data.get("log_items", []))
    tbar.update(1)


def process_input_folder(
    input_dir,
    output_all_file=None,
    output_random_file=None,
    n_jobs=-1,
    log_file=None,
    tmp_root="./tmp3",
    keep_complexes=False,
    complex_cache_dir=None,
):
    run_all_mode = bool(output_all_file)
    run_random_mode = bool(output_random_file)
    if not run_all_mode and not run_random_mode:
        raise ValueError("At least one of --output-all or --output-random must be provided")

    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    cif_files = discover_cif_gz_files(input_dir)
    total_files = len(cif_files)
    print(f"Processing {total_files} CIF files with {n_jobs} processes...")

    os.makedirs(tmp_root, exist_ok=True)
    log = open(log_file, "w") if log_file else None

    all_file = open(output_all_file, "w", newline="") if run_all_mode else None
    random_file = open(output_random_file, "w", newline="") if run_random_mode else None

    try:
        all_writer = csv.writer(all_file) if all_file else None
        random_writer = csv.writer(random_file) if random_file else None
        if all_writer:
            all_writer.writerow(FIXED_HEADERS)
        if random_writer:
            random_writer.writerow(FIXED_HEADERS)

        with tqdm(total=total_files, desc="Processing CIF files") as tbar:
            partial_callback = partial(
                callback,
                tbar=tbar,
                all_writer=all_writer,
                random_writer=random_writer,
                log_file=log,
            )

            with Pool(processes=n_jobs) as pool:
                results = []
                for cif_path in cif_files:
                    result = pool.apply_async(
                        func=process_cif_file,
                        args=(
                            cif_path,
                            tmp_root,
                            keep_complexes,
                            complex_cache_dir,
                            run_all_mode,
                            run_random_mode,
                        ),
                        callback=partial_callback,
                    )
                    results.append(result)

                for result in results:
                    result.wait()
    finally:
        if all_file:
            all_file.close()
        if random_file:
            random_file.close()

    if log:
        log.close()

    if run_all_mode:
        print(f"All-ligand results saved to {output_all_file}")
    if run_random_mode:
        print(f"Random-ligand results saved to {output_random_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Read *.cif.gz files where the first coordinate block is the receptor and "
            "all following coordinate blocks are ligands. Receptor residues are selected "
            "for each ligand using _cpsea_receptor_ligand_map.ligand_ids, then PLIP is "
            "run for every ligand and for one random ligand per CIF file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input folder containing .cif.gz/.mmcif.gz/.cif/.mmcif files",
    )
    parser.add_argument(
        "--output-all",
        default=None,
        help="Output CSV for PLIP results from all ligands",
    )
    parser.add_argument(
        "--output-random",
        default=None,
        help="Output CSV for PLIP results from one random ligand per CIF file",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of worker processes",
    )
    parser.add_argument("--log", help="Optional log file path")
    parser.add_argument("--tmp-root", default="./tmp3", help="Temporary working directory")
    parser.add_argument(
        "--keep-complexes",
        action="store_true",
        help="Keep generated complex PDB files",
    )
    parser.add_argument(
        "--complex-cache-dir",
        default=None,
        help="Directory for kept complex PDB files when --keep-complexes is used",
    )

    args = parser.parse_args()
    start_time = time.time()

    process_input_folder(
        input_dir=args.input,
        output_all_file=args.output_all,
        output_random_file=args.output_random,
        n_jobs=args.jobs,
        log_file=args.log,
        tmp_root=args.tmp_root,
        keep_complexes=args.keep_complexes,
        complex_cache_dir=args.complex_cache_dir,
    )

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

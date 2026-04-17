#!/usr/bin/env python3

import os
import csv
import sys
import time
import shutil
import uuid
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict
import warnings
import xml.etree.ElementTree as ET
import random

warnings.filterwarnings("ignore")

try:
    import gemmi
except ImportError as e:
    raise ImportError(
        "This script requires gemmi. Please install it with: pip install gemmi"
    ) from e


FIXED_HEADERS = [
    'Input',
    'hydrophobic_interactions',
    'hydrogen_bonds',
    'water_bridges',
    'salt_bridges',
    'pi_stacks',
    'pi_cation_interactions',
    'halogen_bonds',
    'metal_complexes'
]


def count_interactions(xml_file):
    """解析 XML 文件，统计各种相互作用数量。"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        interaction_counts = {}
        interactions = root.find('bindingsite/interactions')
        if interactions is not None:
            for child in interactions:
                interaction_type = child.tag
                interaction_count = len(child.findall('./*'))
                interaction_counts[interaction_type] = interaction_count
        return interaction_counts
    except Exception as e:
        print(f"Error processing {xml_file}: {e}", file=sys.stderr)
        return {}


def standardize_interaction_types(interaction_counts):
    """标准化相互作用类型名称，确保与固定表头匹配。"""
    aliases = {
        'hydrophobic': 'hydrophobic_interactions',
        'hydrophobicinteraction': 'hydrophobic_interactions',
        'hbond': 'hydrogen_bonds',
        'hbondinteraction': 'hydrogen_bonds',
        'hbonds': 'hydrogen_bonds',
        'waterbridge': 'water_bridges',
        'waterbridges': 'water_bridges',
        'saltbridge': 'salt_bridges',
        'saltbridges': 'salt_bridges',
        'pistack': 'pi_stacks',
        'pistacks': 'pi_stacks',
        'pication': 'pi_cation_interactions',
        'picationinteraction': 'pi_cation_interactions',
        'halogenbond': 'halogen_bonds',
        'halogenbonds': 'halogen_bonds',
        'metalcomplex': 'metal_complexes',
        'metalcomplexes': 'metal_complexes'
    }

    standardized = {}
    for key, value in interaction_counts.items():
        normalized_key = key.lower().replace('_', '').replace('-', '')
        standardized_key = aliases.get(normalized_key, key)
        standardized[standardized_key] = value

    return standardized


def strip_cif_gz_suffix(path):
    """去掉 .cif.gz / .cif / .gz 后缀，得到样本名。"""
    name = os.path.basename(path)
    for suffix in ('.cif.gz', '.mmcif.gz', '.cif', '.mmcif', '.gz'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def safe_seqnum(residue):
    """安全获取残基编号（整数部分）。"""
    try:
        return int(residue.seqid.num)
    except Exception:
        return None


def atom_altloc_rank(atom):
    """altloc 选择优先级：空白/A 优先，其次 occupancy 高。"""
    altloc = atom.altloc
    if altloc == '\x00':
        altloc = ''
    priority = 0 if altloc in ('', 'A') else 1
    try:
        occ = float(atom.occ)
    except Exception:
        occ = 0.0
    return (priority, -occ, altloc)


def choose_atoms_from_residue(residue):
    """
    对同一残基内重复 atom name 的 altloc 做去重；
    去掉氢原子。
    """
    atom_groups = defaultdict(list)
    for atom in residue:
        if atom.is_hydrogen():
            continue
        atom_groups[atom.name].append(atom)

    chosen = []
    for _, atoms in atom_groups.items():
        best_atom = sorted(atoms, key=atom_altloc_rank)[0]
        chosen.append(best_atom)

    chosen.sort(key=lambda x: x.serial if getattr(x, 'serial', 0) else 0)
    return chosen


def collect_model_residues(model):
    """
    把一个 model 展平为 residue 列表。
    每个元素:
        {
            'orig_chain': 原始链号,
            'seqnum': 原始残基编号整数,
            'resname': 残基名,
            'atoms': [gemmi.Atom, ...]
        }
    """
    residues = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            atoms = choose_atoms_from_residue(residue)
            if not atoms:
                continue
            residues.append({
                'orig_chain': chain.name,
                'seqnum': safe_seqnum(residue),
                'resname': residue.name,
                'atoms': atoms
            })
    return residues


def build_excluded_seqnums(ligand_residues, flank=17):
    """
    根据 ligand 的残基编号，构建要从 receptor 中剔除的编号集合：
    对 ligand 每个残基 n，剔除 [n-flank, n+flank]。
    """
    excluded = set()
    for res in ligand_residues:
        seqnum = res.get('seqnum')
        if seqnum is None:
            continue
        for x in range(seqnum - flank, seqnum + flank + 1):
            excluded.add(x)
    return excluded


def format_pdb_atom_line(record_name, atom_serial, atom_name, altloc, resname,
                         chain_id, resseq, x, y, z, occupancy, bfactor, element):
    """构建标准 PDB ATOM/HETATM 行。"""
    atom_name = (atom_name or '')[:4]
    resname = (resname or 'UNK')[:3]
    chain_id = (chain_id or 'A')[:1]
    altloc = '' if altloc in ('\x00', None) else str(altloc)[:1]
    element = (str(element) if element is not None else '').strip()[:2].rjust(2)
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
    """
    将 residue 列表写入 PDB。
    所有残基统一写为单链，并重新连续编号。
    返回: next_resseq, next_atom_serial, last_resname
    """
    next_resseq = start_resseq
    next_atom_serial = start_atom_serial
    last_resname = 'UNK'

    for res in residues:
        resname = (res.get('resname') or 'UNK').strip() or 'UNK'
        atoms = res.get('atoms', [])
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

            line = format_pdb_atom_line(
                record_name='ATOM',
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
                element=atom.element.name
            )
            out_fh.write(line + '\n')
            next_atom_serial += 1

        last_resname = resname
        next_resseq += 1

    if next_atom_serial > start_atom_serial:
        out_fh.write(
            f"TER   {next_atom_serial:>5}      {last_resname:>3} {new_chain_id}{(next_resseq - 1):>4}\n"
        )
        next_atom_serial += 1

    return next_resseq, next_atom_serial, last_resname


def _load_units_via_structure(cif_path):
    """
    正常 mmCIF：一个 coordinate block 中含多个 model。
    返回按 model 顺序展开的 unit 列表。
    """
    structure = gemmi.read_structure(cif_path)
    units = []
    for model_idx, model in enumerate(structure):
        units.append({
            'parent': structure,
            'model': model,
            'parse_mode': 'structure_models',
            'block_index': 1,
            'block_name': getattr(structure, 'name', 'block_1') or 'block_1',
            'model_num': str(model.num),
            'unit_tag': f"model_{model.num}",
            'unit_ordinal': model_idx + 1,
        })
    return units


def _load_units_via_blocks(cif_path):
    """
    兼容异常 mmCIF：多个 data block 都带有 _atom_site。
    将每个带坐标的 block 视作一个“model-like unit”来源；
    如果某个 block 内本身还有多个 model，则继续展开。
    """
    doc = gemmi.cif.read(cif_path)
    units = []

    for block_idx, block in enumerate(doc, start=1):
        atom_site = block.find_mmcif_category('_atom_site.')
        if not atom_site:
            continue

        structure = gemmi.make_structure_from_block(block)
        if len(structure) == 0:
            continue

        n_models = len(structure)
        for model_idx, model in enumerate(structure, start=1):
            if n_models == 1:
                unit_tag = f"block_{block_idx}"
            else:
                unit_tag = f"block_{block_idx}_model_{model.num}"

            units.append({
                'parent': structure,
                'model': model,
                'parse_mode': 'coordinate_blocks',
                'block_index': block_idx,
                'block_name': block.name,
                'model_num': str(model.num),
                'unit_tag': unit_tag,
                'unit_ordinal': len(units) + 1,
            })

    return units


def load_coordinate_units(cif_path):
    """
    统一读取“可作为 ligand/receptor 的结构单元”。

    优先：单 block 多 model 的正常解析。
    失败时：回退到“多个 coordinate block”模式。
    """
    first_error = None

    try:
        units = _load_units_via_structure(cif_path)
        if units:
            return units
    except Exception as e:
        first_error = e

    try:
        units = _load_units_via_blocks(cif_path)
        if units:
            return units
    except Exception as e2:
        if first_error is None:
            first_error = e2
        else:
            raise ValueError(
                f"Failed both parsing modes for {cif_path}. "
                f"read_structure error: {first_error}; block-wise error: {e2}"
            ) from e2

    if first_error is not None:
        raise ValueError(
            f"No coordinate-bearing units found in {cif_path}. "
            f"Initial parsing error: {first_error}"
        )
    raise ValueError(f"No coordinate-bearing units found in {cif_path}")


def build_complex_pdb_from_units(
    cif_path,
    units,
    ligand_unit_idx,
    flank=17,
    out_pdb_path=None,
):
    """
    从预解析的 units 中重建 complex：
    - 最后一个 unit 作为 receptor
    - 指定 ligand_unit_idx 的 unit 作为 ligand
    """
    if len(units) < 2:
        raise ValueError(
            f"{cif_path} must contain at least 2 coordinate units "
            f"(one or more ligands + final receptor)."
        )

    receptor_unit = units[0]
    ligand_unit = units[ligand_unit_idx]
    receptor_model = receptor_unit['model']
    ligand_model = ligand_unit['model']

    receptor_residues_all = collect_model_residues(receptor_model)
    ligand_residues = collect_model_residues(ligand_model)

    if not ligand_residues:
        raise ValueError(
            f"No valid ligand residues found in unit {ligand_unit['unit_tag']} of {cif_path}"
        )

    excluded_seqnums = build_excluded_seqnums(ligand_residues, flank=flank)
    receptor_residues = [
        res for res in receptor_residues_all
        if (res.get('seqnum') is None) or (res.get('seqnum') not in excluded_seqnums)
    ]

    if out_pdb_path is None:
        raise ValueError("out_pdb_path must be provided.")

    out_dir = os.path.dirname(out_pdb_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_pdb_path, 'w') as out_fh:
        out_fh.write(
            f"REMARK source={os.path.basename(cif_path)} ligand_unit={ligand_unit['unit_tag']} flank={flank}\n"
        )
        out_fh.write(
            f"REMARK parse_mode={ligand_unit['parse_mode']} receptor_unit={receptor_unit['unit_tag']} "
            f"selected_receptor_res={len(receptor_residues)} ligand_res={len(ligand_residues)}\n"
        )

        next_resseq, next_atom_serial, _ = write_residue_block(
            out_fh=out_fh,
            residues=receptor_residues,
            new_chain_id='A',
            start_resseq=1,
            start_atom_serial=1
        )

        write_residue_block(
            out_fh=out_fh,
            residues=ligand_residues,
            new_chain_id='L',
            start_resseq=next_resseq,
            start_atom_serial=next_atom_serial
        )

        out_fh.write('END\n')

    ligand_seqnums = [res.get('seqnum') for res in ligand_residues if res.get('seqnum') is not None]

    return {
        'complex_pdb': out_pdb_path,
        'sample_name': strip_cif_gz_suffix(cif_path),
        'ligand_unit_tag': ligand_unit['unit_tag'],
        'receptor_unit_tag': receptor_unit['unit_tag'],
        'ligand_res_count': len(ligand_residues),
        'receptor_res_count': len(receptor_residues),
        'ligand_seqnums': ligand_seqnums,
        'parse_mode': ligand_unit['parse_mode'],
    }


def run_plip_and_parse(complex_pdb, out_dir):
    """运行 PLIP，并读取 complex_plip_report.xml 统计相互作用。"""
    xml_file = os.path.join(out_dir, 'complex_plip_report.xml')

    cmd = [
        'plip',
        '-f', complex_pdb,
        '--peptides', 'L',
        '-x',
        '-o', out_dir,
        '--silent'
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if completed.returncode != 0:
        return {}, f"PLIP command failed (return code {completed.returncode})"

    if not os.path.exists(xml_file):
        return {}, "PLIP XML report not found"

    return count_interactions(xml_file), None


def process_cif_file(cif_path, flank=17, tmp_root='./tmp3', keep_complexes=False, complex_cache_dir=None):
    """
    处理单个 .cif/.cif.gz 文件：
    - 最后一个 coordinate unit 是 receptor
    - 前面的每个 coordinate unit 都是 ligand
    - coordinate unit 既可能来自一个 block 内的多个 model，也可能来自多个 data block
    """
    results = []
    sample_name = strip_cif_gz_suffix(cif_path)
    temp_items = []

    try:
        units = load_coordinate_units(cif_path)
        n_units = len(units)

        if n_units < 2:
            print(
                f"Warning: skip {cif_path} because it has fewer than 2 coordinate units",
                file=sys.stderr
            )
            return results

        receptor_unit_tag = units[-1]['unit_tag']
        parse_mode = units[0]['parse_mode']

        for ligand_unit_idx in random.sample(range(1,n_units), 1):
            ligand_unit_tag = units[ligand_unit_idx]['unit_tag']
            unique_tag = f"{sample_name}_{ligand_unit_tag}_{uuid.uuid4().hex[:10]}"

            if keep_complexes:
                if complex_cache_dir:
                    complex_dir = os.path.join(complex_cache_dir, sample_name)
                else:
                    complex_dir = os.path.dirname(os.path.abspath(cif_path))
                os.makedirs(complex_dir, exist_ok=True)
                complex_pdb = os.path.join(
                    complex_dir,
                    f"{sample_name}.{ligand_unit_tag}.complex_plip.pdb"
                )
                work_dir = os.path.join(tmp_root, unique_tag)
                os.makedirs(work_dir, exist_ok=True)
                temp_items.append(work_dir)
            else:
                work_dir = os.path.join(tmp_root, unique_tag)
                os.makedirs(work_dir, exist_ok=True)
                complex_pdb = os.path.join(work_dir, 'complex_plip.pdb')
                temp_items.append(work_dir)

            try:
                build_info = build_complex_pdb_from_units(
                    cif_path=cif_path,
                    units=units,
                    ligand_unit_idx=ligand_unit_idx,
                    flank=flank,
                    out_pdb_path=complex_pdb
                )
                interaction_counts, err = run_plip_and_parse(complex_pdb, work_dir)

                if err is not None:
                    print(
                        f"Warning: {sample_name} ligand_unit={ligand_unit_tag} failed: {err}",
                        file=sys.stderr
                    )

                results.append({
                    'file_name': f"{sample_name}__{ligand_unit_tag}",
                    'source_file': os.path.basename(cif_path),
                    'ligand_unit_tag': ligand_unit_tag,
                    'receptor_unit_tag': receptor_unit_tag,
                    'ligand_res_count': build_info['ligand_res_count'],
                    'receptor_res_count': build_info['receptor_res_count'],
                    'ligand_seqnum_min': min(build_info['ligand_seqnums']) if build_info['ligand_seqnums'] else '',
                    'ligand_seqnum_max': max(build_info['ligand_seqnums']) if build_info['ligand_seqnums'] else '',
                    'parse_mode': build_info['parse_mode'],
                    'interactions': interaction_counts
                })

            except Exception as e:
                print(
                    f"Error processing {sample_name} ligand_unit={ligand_unit_tag}: {e}",
                    file=sys.stderr
                )
                results.append({
                    'file_name': f"{sample_name}__{ligand_unit_tag}",
                    'source_file': os.path.basename(cif_path),
                    'ligand_unit_tag': ligand_unit_tag,
                    'receptor_unit_tag': receptor_unit_tag,
                    'ligand_res_count': 0,
                    'receptor_res_count': 0,
                    'ligand_seqnum_min': '',
                    'ligand_seqnum_max': '',
                    'parse_mode': parse_mode,
                    'interactions': {}
                })

        return results

    except Exception as e:
        print(f"Error processing {cif_path}: {e}", file=sys.stderr)
        return results

    finally:
        for path in temp_items:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)


def discover_cif_gz_files(input_dir):
    """发现 input_dir 下所有 .cif.gz / .mmcif.gz / .cif / .mmcif 文件。"""
    cif_files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(('.cif.gz', '.mmcif.gz', '.cif', '.mmcif')):
            cif_files.append(path)
    return cif_files


def callback(return_data, tbar, metadata_writer, log_file):
    """回调函数：一个 cif 文件会返回多个 ligand 结果，因此这里写多行 CSV。"""
    if not return_data:
        tbar.update(1)
        return

    for item in return_data:
        file_name = item.get('file_name', '')
        interaction_counts = item.get('interactions', {})
        standardized_counts = standardize_interaction_types(interaction_counts)

        row = [file_name]
        for header in FIXED_HEADERS[1:]:
            row.append(standardized_counts.get(header, 0))
        metadata_writer.writerow(row)

        if log_file:
            total_interactions = sum(standardized_counts.values())
            log_file.write(
                f"Processed: {file_name} | "
                f"source={item.get('source_file', '')} | "
                f"parse_mode={item.get('parse_mode', '')} | "
                f"ligand_unit={item.get('ligand_unit_tag', '')} | "
                f"receptor_unit={item.get('receptor_unit_tag', '')} | "
                f"receptor_res={item.get('receptor_res_count', '')} | "
                f"ligand_res={item.get('ligand_res_count', '')} | "
                f"seq_range=({item.get('ligand_seqnum_min', '')}, {item.get('ligand_seqnum_max', '')}) | "
                f"total_interactions={total_interactions}\n"
            )
            log_file.flush()

    tbar.update(1)


def process_input_folder(input_dir, output_file='PLIP_metadata_CPSet.csv', n_jobs=-1,
                         log_file=None, flank=17, tmp_root='./tmp3',
                         keep_complexes=False, complex_cache_dir=None):
    """处理输入目录下所有 cif 文件。"""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    cif_files = discover_cif_gz_files(input_dir)
    total_files = len(cif_files)
    print(f"Processing {total_files} cif files with {n_jobs} processes...")

    os.makedirs(tmp_root, exist_ok=True)
    log = open(log_file, 'w') if log_file else None

    with open(output_file, 'w', newline='') as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(FIXED_HEADERS)

        with tqdm(total=total_files, desc='Processing CIF files') as tbar:
            partial_callback = partial(
                callback,
                tbar=tbar,
                metadata_writer=metadata_writer,
                log_file=log
            )

            with Pool(processes=n_jobs) as pool:
                results = []
                for cif_path in cif_files:
                    result = pool.apply_async(
                        func=process_cif_file,
                        args=(cif_path, flank, tmp_root, keep_complexes, complex_cache_dir),
                        callback=partial_callback
                    )
                    results.append(result)

                for result in results:
                    result.wait()

    if log:
        log.close()

    print(f"Processing completed. Results saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Read *.cif.gz files. The final coordinate unit is treated as receptor, '
            'all previous coordinate units are treated as ligands. A coordinate unit can '            'be either a model in a standard mmCIF or a full data block in a nonstandard multi-block mmCIF. '
            'For each ligand, remove receptor residues within +/- flank residues of ligand seqnums, '
            'rebuild a PLIP complex, run PLIP, and summarize interactions.'
        )
    )
    parser.add_argument('-i', '--input', required=True, help='Input folder containing many .cif.gz files')
    parser.add_argument('-o', '--output', default='PLIP_metadata_CPSet.csv', help='Output CSV file')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='Number of processes to use')
    parser.add_argument('--log', help='Log file path')
    parser.add_argument('--flank', type=int, default=17, help='Exclude receptor residues within +/- flank of ligand seqnums')
    parser.add_argument('--tmp-root', default='./tmp3', help='Temporary working directory')
    parser.add_argument(
        '--keep-complexes',
        action='store_true',
        help='Keep generated complex PDB files'
    )
    parser.add_argument(
        '--complex-cache-dir',
        default=None,
        help='Directory to store kept complex PDB files (only used with --keep-complexes)'
    )

    args = parser.parse_args()

    start_time = time.time()

    process_input_folder(
        input_dir=args.input,
        output_file=args.output,
        n_jobs=args.jobs,
        log_file=args.log,
        flank=args.flank,
        tmp_root=args.tmp_root,
        keep_complexes=args.keep_complexes,
        complex_cache_dir=args.complex_cache_dir
    )

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

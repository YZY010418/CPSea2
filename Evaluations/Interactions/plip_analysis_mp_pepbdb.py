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
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# 固定表头顺序，第一列为输入子目录名称
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
    """解析XML文件，统计各种相互作用的数量"""
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
    """标准化相互作用类型名称，确保与固定表头匹配"""
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


def iter_atom_lines(pdb_path):
    """遍历PDB中的 ATOM/HETATM 记录"""
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM  ', 'HETATM')):
                yield line.rstrip('\n')


def rewrite_component_pdb(pdb_path, out_fh, new_chain_id, resseq_start, atom_serial_start):
    """
    将一个PDB组件重写到输出文件中：
    1. 统一改成新的链号
    2. 残基重新连续编号，且从 resseq_start 开始
    3. 原子序号重新连续编号，且从 atom_serial_start 开始

    返回:
        next_resseq, next_atom_serial, last_resname
    """
    residue_map = {}
    next_resseq = resseq_start
    next_atom_serial = atom_serial_start
    last_resname = 'UNK'

    for line in iter_atom_lines(pdb_path):
        old_chain = line[21]
        old_resseq = line[22:26]
        old_icode = line[26]
        resname = line[17:20]
        residue_key = (old_chain, old_resseq, old_icode, resname)

        if residue_key not in residue_map:
            residue_map[residue_key] = next_resseq
            next_resseq += 1

        new_resseq = residue_map[residue_key]
        atom_serial = next_atom_serial
        next_atom_serial += 1
        last_resname = resname.strip() or 'UNK'

        new_line = (
            f"{line[:6]}"
            f"{atom_serial:>5}"
            f"{line[11:21]}"
            f"{new_chain_id}"
            f"{new_resseq:>4}"
            f" {line[27:]}"
        )
        out_fh.write(new_line + '\n')

    if next_atom_serial > atom_serial_start:
        ter_serial = next_atom_serial
        out_fh.write(
            f"TER   {ter_serial:>5}      {last_resname:>3} {new_chain_id}{(next_resseq - 1):>4}\n"
        )
        next_atom_serial += 1

    return next_resseq, next_atom_serial, last_resname


def build_complex_pdb(job_dir, complex_name='complex_plip.pdb', force_rebuild=False):
    """
    读取 job_dir/peptide.pdb 与 job_dir/receptor.pdb，合并成一个 complex PDB。

    约定：
    - receptor 统一写成链 A，残基从 1 开始编号
    - peptide 统一写成链 L，残基从 receptor 结束后继续编号
    - 原子号在整个 complex 中全局唯一
    """
    peptide_pdb = os.path.join(job_dir, 'peptide.pdb')
    receptor_pdb = os.path.join(job_dir, 'receptor.pdb')
    complex_pdb = os.path.join(job_dir, complex_name)

    if not os.path.exists(peptide_pdb):
        raise FileNotFoundError(f"Missing peptide.pdb in {job_dir}")
    if not os.path.exists(receptor_pdb):
        raise FileNotFoundError(f"Missing receptor.pdb in {job_dir}")

    if (
        (not force_rebuild)
        and os.path.exists(complex_pdb)
        and os.path.getmtime(complex_pdb) >= max(os.path.getmtime(peptide_pdb), os.path.getmtime(receptor_pdb))
    ):
        return complex_pdb

    with open(complex_pdb, 'w') as out_fh:
        out_fh.write(f"REMARK merged from {os.path.basename(receptor_pdb)} + {os.path.basename(peptide_pdb)}\n")
        next_resseq, next_atom_serial, _ = rewrite_component_pdb(
            receptor_pdb, out_fh, new_chain_id='A', resseq_start=1, atom_serial_start=1
        )
        rewrite_component_pdb(
            peptide_pdb, out_fh, new_chain_id='L', resseq_start=next_resseq, atom_serial_start=next_atom_serial
        )
        out_fh.write('END\n')

    return complex_pdb


def process_job_dir(job_dir, complex_name='complex_plip.pdb', force_rebuild=False):
    """处理单个子目录：构建complex，运行PLIP，解析结果"""
    sample_name = os.path.basename(os.path.abspath(job_dir))
    xml_file = None
    temp_folder = None

    try:
        complex_pdb = build_complex_pdb(
            job_dir=job_dir,
            complex_name=complex_name,
            force_rebuild=force_rebuild
        )

        tmp_root = os.path.abspath('./tmp3')
        os.makedirs(tmp_root, exist_ok=True)

        unique_name = f"{sample_name}_{uuid.uuid4().hex[:10]}"
        output_xml = os.path.join(tmp_root, unique_name)
        temp_folder = output_xml
        xml_file = os.path.join(output_xml, 'complex_plip_report.xml')

        cmd = [
            'plip',
            '-f', complex_pdb,
            '--peptides', 'L',
            '-x',
            '-o', output_xml,
            '--silent'
        ]
        completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if completed.returncode != 0:
            print(f"Warning: PLIP command failed for {job_dir}", file=sys.stderr)
            return {'file_name': sample_name, 'interactions': {}}

        if not os.path.exists(xml_file):
            print(f"Error: XML file {xml_file} not found", file=sys.stderr)
            return {'file_name': sample_name, 'interactions': {}}

        return {
            'file_name': sample_name,
            'interactions': count_interactions(xml_file)
        }

    except Exception as e:
        print(f"Error processing {job_dir}: {e}", file=sys.stderr)
        return {'file_name': sample_name, 'interactions': {}}

    finally:
        if temp_folder and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)


def discover_job_dirs(input_dir):
    """发现 input_dir 下一级子目录中同时含有 peptide.pdb / receptor.pdb 的任务目录"""
    job_dirs = []

    for name in sorted(os.listdir(input_dir)):
        subdir = os.path.join(input_dir, name)
        if not os.path.isdir(subdir):
            continue

        peptide_pdb = os.path.join(subdir, 'peptide.pdb')
        receptor_pdb = os.path.join(subdir, 'receptor.pdb')
        if os.path.exists(peptide_pdb) and os.path.exists(receptor_pdb):
            job_dirs.append(subdir)
        else:
            print(
                f"Warning: skip {subdir} because peptide.pdb or receptor.pdb is missing",
                file=sys.stderr
            )

    return job_dirs


def callback(return_data, tbar, metadata_writer, log_file):
    """回调函数：写入CSV并更新进度条"""
    file_name = return_data.get('file_name', '')
    interaction_counts = return_data.get('interactions', {})
    standardized_counts = standardize_interaction_types(interaction_counts)

    row = [file_name]
    for header in FIXED_HEADERS[1:]:
        row.append(standardized_counts.get(header, 0))

    metadata_writer.writerow(row)
    tbar.update(1)

    if log_file:
        total_interactions = sum(standardized_counts.values())
        log_file.write(f"Processed: {file_name} - {total_interactions} interactions\n")
        log_file.flush()


def process_input_folder(input_dir, output_file='PLIP_metadata_CPSet.csv', n_jobs=-1,
                         log_file=None, complex_name='complex_plip.pdb', force_rebuild=False):
    """处理输入目录下的所有任务子目录"""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    job_dirs = discover_job_dirs(input_dir)
    total_files = len(job_dirs)
    print(f"Processing {total_files} job folders with {n_jobs} processes...")

    log = open(log_file, 'w') if log_file else None

    with open(output_file, 'w', newline='') as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(FIXED_HEADERS)

        with tqdm(total=total_files, desc='Processing folders') as tbar:
            partial_callback = partial(
                callback,
                tbar=tbar,
                metadata_writer=metadata_writer,
                log_file=log
            )

            with Pool(processes=n_jobs) as pool:
                results = []
                for job_dir in job_dirs:
                    result = pool.apply_async(
                        func=process_job_dir,
                        args=(job_dir, complex_name, force_rebuild),
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
        description='Merge peptide.pdb + receptor.pdb in each subfolder, run PLIP, and summarize interactions'
    )
    parser.add_argument('-i', '--input', required=True, help='Input root folder containing many subfolders')
    parser.add_argument('-o', '--output', default='PLIP_metadata_CPSet.csv', help='Output CSV file')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='Number of processes to use')
    parser.add_argument('--log', help='Log file path')
    parser.add_argument('--complex-name', default='complex_plip.pdb', help='Cached merged complex PDB filename')
    parser.add_argument('--force-rebuild', action='store_true', help='Rebuild cached complex PDB even if it already exists')

    args = parser.parse_args()

    start_time = time.time()

    process_input_folder(
        input_dir=args.input,
        output_file=args.output,
        n_jobs=args.jobs,
        log_file=args.log,
        complex_name=args.complex_name,
        force_rebuild=args.force_rebuild
    )

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

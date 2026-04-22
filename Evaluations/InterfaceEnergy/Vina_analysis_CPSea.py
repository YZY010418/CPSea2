import os
import re
import csv
import time
import gzip
import shutil
import argparse
import warnings
import subprocess
import uuid
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

warnings.filterwarnings("ignore")

PYTHONSH = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
PREPARE_LIGAND4 = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'
PREPARE_RECEPTOR4 = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'
VINA_BIN = '/data_hdd/home/yangziyi/autodock_vina_1_1_2_linux_x86/bin/vina'


def open_text_auto(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def get_base_name(path):
    base = os.path.basename(path)
    if base.endswith('.gz'):
        base = os.path.splitext(base)[0]
    if base.endswith('.pdb'):
        base = os.path.splitext(base)[0]
    return base


def parse_chain_id_from_pdb_line(line):
    if len(line) >= 22:
        return line[21].strip()
    return ''


def parse_residue_key_from_pdb_line(line):
    chain_id = parse_chain_id_from_pdb_line(line)
    resseq = line[22:26].strip() if len(line) >= 26 else ''
    icode = line[26].strip() if len(line) >= 27 else ''
    resname = line[17:20].strip() if len(line) >= 20 else ''
    return chain_id, resseq, icode, resname


def extract_receptor_ligand_pair_from_pdb(pdb_path, tmp_dir, ligand_chain='L', receptor_chain='R'):
    base_name = get_base_name(pdb_path)
    ligand_pdb = os.path.join(tmp_dir, f'{base_name}_{ligand_chain}.pdb')
    receptor_pdb = os.path.join(tmp_dir, f'{base_name}_{receptor_chain}.pdb')

    ligand_atoms = 0
    receptor_atoms = 0
    ligand_residues = set()
    receptor_residues = set()

    with open_text_auto(pdb_path) as fh, open(ligand_pdb, 'w') as lig_out, open(receptor_pdb, 'w') as rec_out:
        ligand_serial = 1
        receptor_serial = 1

        for line in fh:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            chain_id = parse_chain_id_from_pdb_line(line)
            if chain_id == ligand_chain:
                new_line = f"{line[:6]}{ligand_serial:>5d}{line[11:]}"
                lig_out.write(new_line)
                ligand_serial += 1
                ligand_atoms += 1
                ligand_residues.add(parse_residue_key_from_pdb_line(line))
            elif chain_id == receptor_chain:
                new_line = f"{line[:6]}{receptor_serial:>5d}{line[11:]}"
                rec_out.write(new_line)
                receptor_serial += 1
                receptor_atoms += 1
                receptor_residues.add(parse_residue_key_from_pdb_line(line))

        if ligand_atoms > 0:
            lig_out.write('TER\nEND\n')
        if receptor_atoms > 0:
            rec_out.write('TER\nEND\n')

    if ligand_atoms == 0:
        raise ValueError(f'No atoms found for ligand chain {ligand_chain} in {os.path.basename(pdb_path)}')
    if receptor_atoms == 0:
        raise ValueError(f'No atoms found for receptor chain {receptor_chain} in {os.path.basename(pdb_path)}')

    return base_name, ligand_chain, receptor_chain, ligand_pdb, receptor_pdb, len(ligand_residues), len(receptor_residues)


def run_subprocess(cmd, err_prefix):
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'{err_prefix}: {e.stderr}')


def read_pdb_coordinates(pdb_path):
    coords = []
    with open(pdb_path, 'r') as fh:
        for line in fh:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    x, y, z = map(float, parts[6:9])
                except Exception:
                    continue
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f'No atom coordinates found in {pdb_path}')
    return coords


def compute_vina_box_from_ligand_pdb(ligand_pdb, padding=8.0, min_size=20.0, max_size=126.0):
    coords = read_pdb_coordinates(ligand_pdb)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)

    size_x = min(max(span_x + padding, min_size), max_size)
    size_y = min(max(span_y + padding, min_size), max_size)
    size_z = min(max(span_z + padding, min_size), max_size)

    return {
        'center_x': center_x,
        'center_y': center_y,
        'center_z': center_z,
        'size_x': size_x,
        'size_y': size_y,
        'size_z': size_z,
    }


def parse_vina_affinity(stdout_text):
    m = re.search(r'Affinity:\s+([-\d.]+)', stdout_text)
    if m:
        return m.group(1)

    for line in stdout_text.splitlines():
        if re.match(r'^\s*1\s+[-\d.]+\s+', line):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def score_with_vina(receptor_pdb, ligand_pdb, tmp_dir, base_name, ligand_chain, score_only=False):
    ligand_pdbqt = os.path.join(tmp_dir, f'{base_name}_{ligand_chain}_ligand.pdbqt')
    receptor_pdbqt = os.path.join(tmp_dir, f'{base_name}_receptor.pdbqt')

    ligand_cmd = [PYTHONSH, PREPARE_LIGAND4, '-l', ligand_pdb, '-o', ligand_pdbqt]
    receptor_cmd = [PYTHONSH, PREPARE_RECEPTOR4, '-r', receptor_pdb, '-o', receptor_pdbqt]

    run_subprocess(ligand_cmd, f'prepare_ligand4 failed for chain {ligand_chain}')
    run_subprocess(receptor_cmd, 'prepare_receptor4 failed')

    vina_cmd = [
        VINA_BIN,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--cpu', '1',
    ]

    if score_only:
        vina_cmd.append('--score_only')
        err_prefix = f'vina --score_only failed for chain {ligand_chain}'
    else:
        box = compute_vina_box_from_ligand_pdb(ligand_pdb)
        vina_cmd.extend([
            '--local_only',
            '--center_x', f"{box['center_x']:.3f}",
            '--center_y', f"{box['center_y']:.3f}",
            '--center_z', f"{box['center_z']:.3f}",
            '--size_x', f"{box['size_x']:.3f}",
            '--size_y', f"{box['size_y']:.3f}",
            '--size_z', f"{box['size_z']:.3f}",
        ])
        err_prefix = f'vina --local_only failed for chain {ligand_chain}'

    result = run_subprocess(vina_cmd, err_prefix)
    return parse_vina_affinity(result.stdout)


def process_pdb_file(task):
    pdb_path, score_only, ligand_chain, receptor_chain = task
    tmp_dir = os.path.abspath(os.path.join('./tmp_vina_pdb_lr', str(uuid.uuid4())))
    os.makedirs(tmp_dir, exist_ok=True)

    base_name = get_base_name(pdb_path)

    try:
        base_name, ligand_chain, receptor_chain, ligand_pdb, receptor_pdb, n_ligand_res, n_receptor_res = extract_receptor_ligand_pair_from_pdb(
            pdb_path=pdb_path,
            tmp_dir=tmp_dir,
            ligand_chain=ligand_chain,
            receptor_chain=receptor_chain,
        )
        affinity = score_with_vina(
            receptor_pdb=receptor_pdb,
            ligand_pdb=ligand_pdb,
            tmp_dir=tmp_dir,
            base_name=base_name,
            ligand_chain=ligand_chain,
            score_only=score_only,
        )
        return [base_name, ligand_chain, receptor_chain, n_ligand_res, n_receptor_res, affinity, '']
    except Exception as e:
        return [base_name, ligand_chain, receptor_chain, '', '', '', str(e)]
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def list_pdb_files(input_dir, recursive=False):
    suffixes = ('.pdb', '.pdb.gz')
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for name in names:
                if name.endswith(suffixes):
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and name.endswith(suffixes):
                files.append(path)
    return sorted(files)


def process_pdb_dir(input_dir, output_file, n_jobs=1, recursive=False, score_only=False, ligand_chain='L', receptor_chain='R'):
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    pdb_files = list_pdb_files(input_dir, recursive=recursive)
    total_files = len(pdb_files)
    print(f'Found {total_files} PDB files in {input_dir}')
    print(f'Using {n_jobs} worker processes')
    print(f'Ligand chain: {ligand_chain}; Receptor chain: {receptor_chain}')
    print(f'Vina mode: {"score_only" if score_only else "local_only"}')

    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['input', 'ligand_chain', 'receptor_chain', 'n_ligand_residues', 'n_receptor_residues', 'affinity', 'error'])

        if total_files == 0:
            return

        tasks = [(path, score_only, ligand_chain, receptor_chain) for path in pdb_files]
        with Pool(processes=n_jobs) as pool:
            for row in tqdm(pool.imap_unordered(process_pdb_file, tasks, chunksize=1), total=total_files):
                writer.writerow(row)
                fh.flush()


def main():
    parser = argparse.ArgumentParser(
        description='Traverse a directory of .pdb/.pdb.gz files, split ligand/receptor chains from each file, and compute Vina affinity. Default mode is local_only; use --score_only to switch to score_only.'
    )
    parser.add_argument('-i', '--input', required=True, help='Input directory containing .pdb or .pdb.gz files')
    parser.add_argument('-o', '--output', default='check_vina_pdb_lr.csv', help='Output CSV file')
    parser.add_argument('-p', '--jobs', type=int, default=1, help='Number of worker processes; use -1 for all CPUs minus one')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for .pdb/.pdb.gz files under input directory')
    parser.add_argument('--ligand_chain', default='L', help='Ligand chain ID in input PDB files (default: L)')
    parser.add_argument('--receptor_chain', default='R', help='Receptor chain ID in input PDB files (default: R)')
    parser.add_argument('--score_only', action='store_true', help='Use Vina --score_only instead of the default --local_only mode')
    args = parser.parse_args()

    if len(args.ligand_chain) != 1 or len(args.receptor_chain) != 1:
        raise ValueError('ligand_chain and receptor_chain must each be a single-character chain ID for PDB format')

    start_time = time.time()
    process_pdb_dir(
        input_dir=args.input,
        output_file=args.output,
        n_jobs=args.jobs,
        recursive=args.recursive,
        score_only=args.score_only,
        ligand_chain=args.ligand_chain,
        receptor_chain=args.receptor_chain,
    )
    end_time = time.time()
    print(f'Total processing time: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
    main()

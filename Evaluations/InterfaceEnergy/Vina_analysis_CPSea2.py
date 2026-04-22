import os
import re
import csv
import sys
import time
import gzip
import shutil
import random
import hashlib
import argparse
import warnings
import subprocess
import uuid
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import gemmi

warnings.filterwarnings("ignore")

PYTHONSH = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
PREPARE_LIGAND4 = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'
PREPARE_RECEPTOR4 = '/data_hdd/home/yangziyi/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'
VINA_BIN = '/data_hdd/home/yangziyi/autodock_vina_1_1_2_linux_x86/bin/vina'


def normalize_ins_code(value):
    if value in (None, False, '', '.', '?', ' '):
        return ''
    return str(value)


def atom_name_to_pdb_field(atom_name, element):
    atom_name = str(atom_name)
    element = ('' if element is None else str(element)).strip()
    if len(atom_name) >= 4:
        return atom_name[:4]
    if len(element) == 1 and atom_name and atom_name[0].isalpha():
        return f" {atom_name:<3}"
    return f"{atom_name:>4}"


# Minimal PDB writer from mmCIF atom_site category.
def write_pdb_from_atom_site(atom_site, out_path, allowed_residue_keys=None):
    n_atoms = len(atom_site['id'])
    serial = 1
    prev_chain = None
    prev_res_key = None
    written = 0

    with open(out_path, 'w') as fh:
        for i in range(n_atoms):
            label_asym_id = str(atom_site['label_asym_id'][i])
            label_seq_id = str(atom_site['label_seq_id'][i])
            ins_code = normalize_ins_code(atom_site['pdbx_PDB_ins_code'][i])
            res_key = (label_asym_id, label_seq_id, ins_code)

            if allowed_residue_keys is not None and res_key not in allowed_residue_keys:
                continue

            group_pdb = str(atom_site['group_PDB'][i])
            record = 'HETATM' if group_pdb.upper().startswith('HET') else 'ATOM  '
            element = str(atom_site['type_symbol'][i]).strip()
            atom_name = str(atom_site['auth_atom_id'][i]).strip()
            res_name = str(atom_site['auth_comp_id'][i]).strip()
            chain_id = str(atom_site['auth_asym_id'][i]).strip() or label_asym_id or 'A'
            chain_id = chain_id[0]

            try:
                resseq = int(float(atom_site['auth_seq_id'][i]))
            except Exception:
                try:
                    resseq = int(float(label_seq_id))
                except Exception:
                    resseq = 1

            try:
                x = float(atom_site['Cartn_x'][i])
                y = float(atom_site['Cartn_y'][i])
                z = float(atom_site['Cartn_z'][i])
            except Exception:
                continue

            if prev_res_key is not None and prev_res_key != res_key and prev_chain == chain_id:
                pass
            if prev_chain is not None and prev_chain != chain_id:
                fh.write('TER\n')

            atom_name_field = atom_name_to_pdb_field(atom_name, element)
            altloc = ' '
            icode = ins_code[:1] if ins_code else ' '
            occupancy = 1.00
            temp_factor = 0.00

            line = (
                f"{record:<6}{serial:>5d} {atom_name_field}{altloc}{res_name:>3s} {chain_id:1s}"
                f"{resseq:>4d}{icode:1s}   "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{occupancy:>6.2f}{temp_factor:>6.2f}          "
                f"{element:>2s}\n"
            )
            fh.write(line)
            serial += 1
            written += 1
            prev_chain = chain_id
            prev_res_key = res_key

        if written > 0:
            fh.write('TER\nEND\n')

    return written


def read_cif_document(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as fh:
            text = fh.read()
        return gemmi.cif.read_string(text)
    return gemmi.cif.read_file(path)


def choose_ligand_id(candidate_ids, base_name, seed=None):
    candidate_ids = sorted(candidate_ids, key=lambda x: int(x))
    if not candidate_ids:
        raise ValueError('No candidate ligand IDs found.')
    if seed is None:
        return random.choice(candidate_ids)
    digest = hashlib.md5(f"{seed}:{base_name}".encode('utf-8')).hexdigest()
    idx = int(digest, 16) % len(candidate_ids)
    return candidate_ids[idx]


def extract_receptor_ligand_pair(cif_path, tmp_dir, seed=None):
    file_name = os.path.basename(cif_path)
    base_name = file_name
    if base_name.endswith('.gz'):
        base_name = os.path.splitext(base_name)[0]
    if base_name.endswith('.cif'):
        base_name = os.path.splitext(base_name)[0]

    doc = read_cif_document(cif_path)
    if len(doc) < 2:
        raise ValueError('Expected at least one receptor block and one ligand block.')

    receptor_block = None
    ligand_blocks = {}

    for block in doc:
        name = block.name
        if name.endswith('_receptor') and receptor_block is None:
            receptor_block = block
        m = re.search(r'_ligand(\d+)$', name)
        if m:
            ligand_blocks[m.group(1)] = block

    if receptor_block is None:
        receptor_block = doc[0]

    if not ligand_blocks:
        for block in doc[1:]:
            m = re.search(r'ligand(\d+)', block.name)
            if m:
                ligand_blocks[m.group(1)] = block

    if not ligand_blocks:
        raise ValueError('No ligand blocks found in mmCIF file.')

    mapping = receptor_block.get_mmcif_category('_cpsea_receptor_ligand_map.')
    if not mapping:
        raise ValueError('Missing _cpsea_receptor_ligand_map category in receptor block.')

    ligand_to_receptor_keys = {}
    n_rows = len(mapping['ligand_ids'])
    for i in range(n_rows):
        label_asym_id = str(mapping['label_asym_id'][i])
        label_seq_id = str(mapping['label_seq_id'][i])
        ins_code = normalize_ins_code(mapping['pdbx_PDB_ins_code'][i])
        res_key = (label_asym_id, label_seq_id, ins_code)
        ligand_ids_raw = str(mapping['ligand_ids'][i]).strip()
        if not ligand_ids_raw:
            continue
        for ligand_id in [x.strip() for x in ligand_ids_raw.split(',') if x.strip()]:
            ligand_to_receptor_keys.setdefault(ligand_id, set()).add(res_key)

    candidate_ids = sorted(set(ligand_blocks.keys()) & set(ligand_to_receptor_keys.keys()), key=lambda x: int(x))
    if not candidate_ids:
        raise ValueError('No overlapping ligand IDs between ligand blocks and receptor-ligand map.')

    chosen_ligand_id = choose_ligand_id(candidate_ids, base_name, seed=seed)
    receptor_keys = ligand_to_receptor_keys[chosen_ligand_id]
    ligand_block = ligand_blocks[chosen_ligand_id]

    receptor_atom_site = receptor_block.get_mmcif_category('_atom_site.')
    ligand_atom_site = ligand_block.get_mmcif_category('_atom_site.')
    if not receptor_atom_site or not ligand_atom_site:
        raise ValueError('Missing _atom_site category.')

    receptor_pdb = os.path.join(tmp_dir, f'{base_name}_R_lig{chosen_ligand_id}.pdb')
    ligand_pdb = os.path.join(tmp_dir, f'{base_name}_L_lig{chosen_ligand_id}.pdb')

    n_receptor_atoms = write_pdb_from_atom_site(receptor_atom_site, receptor_pdb, allowed_residue_keys=receptor_keys)
    n_ligand_atoms = write_pdb_from_atom_site(ligand_atom_site, ligand_pdb, allowed_residue_keys=None)

    if n_receptor_atoms == 0:
        raise ValueError(f'Chosen ligand {chosen_ligand_id} produced an empty receptor selection.')
    if n_ligand_atoms == 0:
        raise ValueError(f'Chosen ligand {chosen_ligand_id} produced an empty ligand.')

    return base_name, chosen_ligand_id, receptor_pdb, ligand_pdb, len(candidate_ids), len(receptor_keys)


def run_subprocess(cmd, err_prefix, decode_stderr=True):
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if decode_stderr else e.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f'{err_prefix}: {stderr}')


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


def score_with_vina(receptor_pdb, ligand_pdb, tmp_dir, base_name, ligand_id, score_only=False):
    ligand_pdbqt = os.path.join(tmp_dir, f'{base_name}_lig{ligand_id}_ligand.pdbqt')
    receptor_pdbqt = os.path.join(tmp_dir, f'{base_name}_lig{ligand_id}_receptor.pdbqt')

    ligand_cmd = [PYTHONSH, PREPARE_LIGAND4, '-l', ligand_pdb, '-o', ligand_pdbqt]
    receptor_cmd = [PYTHONSH, PREPARE_RECEPTOR4, '-r', receptor_pdb, '-o', receptor_pdbqt]

    run_subprocess(ligand_cmd, f'prepare_ligand4 failed for ligand {ligand_id}')
    run_subprocess(receptor_cmd, f'prepare_receptor4 failed for ligand {ligand_id}')

    vina_cmd = [
        VINA_BIN,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--cpu', '1',
    ]

    if score_only:
        vina_cmd.append('--score_only')
        err_prefix = f'vina --score_only failed for ligand {ligand_id}'
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
        err_prefix = f'vina --local_only failed for ligand {ligand_id}'

    result = run_subprocess(vina_cmd, err_prefix)
    affinity = parse_vina_affinity(result.stdout)
    return affinity


def process_cif_file(task):
    cif_path, seed, score_only = task
    tmp_dir = os.path.abspath(os.path.join('./tmp_vina_cpsea_cif', str(uuid.uuid4())))
    os.makedirs(tmp_dir, exist_ok=True)

    file_name = os.path.basename(cif_path)
    base_name = file_name
    if base_name.endswith('.gz'):
        base_name = os.path.splitext(base_name)[0]
    if base_name.endswith('.cif'):
        base_name = os.path.splitext(base_name)[0]

    try:
        base_name, ligand_id, receptor_pdb, ligand_pdb, n_candidates, n_receptor_res = extract_receptor_ligand_pair(
            cif_path=cif_path,
            tmp_dir=tmp_dir,
            seed=seed,
        )
        affinity = score_with_vina(receptor_pdb, ligand_pdb, tmp_dir, base_name, ligand_id, score_only=score_only)
        return [base_name, ligand_id, n_candidates, n_receptor_res, affinity, '']
    except Exception as e:
        return [base_name, '', '', '', '', str(e)]
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def list_cif_files(input_dir, recursive=False):
    suffixes = ('.cif', '.cif.gz')
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


def process_cif_dir(input_dir, output_file, n_jobs=1, seed=None, recursive=False, score_only=False):
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    cif_files = list_cif_files(input_dir, recursive=recursive)
    total_files = len(cif_files)
    print(f'Found {total_files} CIF files in {input_dir}')
    print(f'Using {n_jobs} worker processes')

    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['input', 'ligand_id', 'n_candidate_ligands', 'n_receptor_residues', 'affinity', 'error'])

        if total_files == 0:
            return

        tasks = [(path, seed, score_only) for path in cif_files]
        with Pool(processes=n_jobs) as pool:
            for row in tqdm(pool.imap_unordered(process_cif_file, tasks, chunksize=1), total=total_files):
                writer.writerow(row)
                fh.flush()


def main():
    parser = argparse.ArgumentParser(
        description='Randomly select one ligand-receptor pair from each CPSea mmCIF(.gz) file and compute Vina affinity. Default mode is local_only; use --score_only to switch to score_only.'
    )
    parser.add_argument('-i', '--input', required=True, help='Input directory containing .cif or .cif.gz files')
    parser.add_argument('-o', '--output', default='check_vina_CPSea_cif.csv', help='Output CSV file')
    parser.add_argument('-p', '--jobs', type=int, default=1, help='Number of worker processes; use -1 for all CPUs minus one')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for deterministic ligand selection per file')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for .cif/.cif.gz files under input directory')
    parser.add_argument('--score_only', action='store_true', help='Use Vina --score_only instead of the default --local_only mode')
    args = parser.parse_args()

    start_time = time.time()
    process_cif_dir(
        input_dir=args.input,
        output_file=args.output,
        n_jobs=args.jobs,
        seed=args.seed,
        recursive=args.recursive,
        score_only=args.score_only,
    )
    end_time = time.time()
    print(f'Total processing time: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import random
from collections import Counter
from multiprocessing import Pool

import gemmi

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]


def find_cif_gz_files(input_dir):
    """在输入目录下寻找所有 .cif.gz 文件（不递归）"""
    cif_files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.endswith(".cif.gz"):
            cif_files.append(path)
    return cif_files


def is_valid_value(x):
    return x not in (None, "", ".", "?")


def pick_first_present(cat, keys, default=None):
    """从 cat(dict) 中按优先级取第一个存在的列"""
    for k in keys:
        if k in cat:
            return cat[k]
    return default


def get_coordinate_blocks(doc):
    """
    返回所有包含 _atom_site category 的 block
    """
    coord_blocks = []
    for block in doc:
        cat = block.get_mmcif_category("_atom_site.")
        if cat and len(cat) > 0:
            coord_blocks.append(block)
    return coord_blocks


def count_residues_from_random_block(args):
    """
    对单个 .cif.gz:
    - 读取 CIF document
    - 找到所有带 _atom_site 的 block
    - 排除最后一个坐标 block
    - 从前面的坐标 block 中随机选一个
    - 统计该 block 中标准20种氨基酸残基数
    """
    cif_path, seed = args
    counter = Counter()

    try:
        doc = gemmi.cif.read(cif_path)
        coord_blocks = get_coordinate_blocks(doc)

        if len(coord_blocks) < 2:
            print(f"[Warning] Skip {cif_path}: coordinate blocks < 2, cannot exclude the last one.")
            return {
                "counter": Counter(),
                "accepted": False,
                "file": cif_path,
                "selected_block": None
            }

        rng = random.Random(seed)
        selected_idx = rng.randrange(0, len(coord_blocks) - 1)  # 排除最后一个
        block = coord_blocks[selected_idx]

        cat = block.get_mmcif_category("_atom_site.")
        if not cat:
            print(f"[Warning] Skip {cif_path}: selected block has no _atom_site.")
            return {
                "counter": Counter(),
                "accepted": False,
                "file": cif_path,
                "selected_block": block.name
            }

        # 优先使用 auth_*，没有再回退到 label_*
        comp_ids = pick_first_present(cat, ["auth_comp_id", "label_comp_id"])
        chain_ids = pick_first_present(cat, ["auth_asym_id", "label_asym_id"])
        seq_ids = pick_first_present(cat, ["auth_seq_id", "label_seq_id"])
        ins_codes = pick_first_present(cat, ["pdbx_PDB_ins_code"], None)

        if comp_ids is None or chain_ids is None or seq_ids is None:
            print(f"[Warning] Skip {cif_path}: missing essential _atom_site columns in block {block.name}.")
            return {
                "counter": Counter(),
                "accepted": False,
                "file": cif_path,
                "selected_block": block.name
            }

        n_rows = len(comp_ids)
        seen_residues = set()

        for i in range(n_rows):
            resname = str(comp_ids[i]).strip().upper()
            if resname not in AA3_TO_AA1:
                continue

            chain_id = str(chain_ids[i]).strip()
            seq_id = str(seq_ids[i]).strip()

            if ins_codes is not None and i < len(ins_codes):
                ins_code = str(ins_codes[i]).strip()
                if not is_valid_value(ins_code):
                    ins_code = ""
            else:
                ins_code = ""

            residue_key = (chain_id, seq_id, ins_code)
            if residue_key in seen_residues:
                continue

            seen_residues.add(residue_key)
            counter[resname] += 1

        return {
            "counter": counter,
            "accepted": True,
            "file": cif_path,
            "selected_block": block.name
        }

    except Exception as e:
        print(f"[Warning] Failed to process {cif_path}: {e}")
        return {
            "counter": Counter(),
            "accepted": False,
            "file": cif_path,
            "selected_block": None
        }


def main():
    parser = argparse.ArgumentParser(
        description="统计目录中 .cif.gz 文件的氨基酸组成：每个文件排除最后一个坐标block，从前面的block里随机选一个进行统计"
    )
    parser.add_argument("-i", "--input", required=True, help="输入目录，里面是一系列 .cif.gz 文件")
    parser.add_argument("-p", "--processes", type=int, default=1, help="进程数")
    parser.add_argument("-o", "--output", required=True, help="输出CSV路径")
    parser.add_argument("--seed", type=int, default=20260319, help="随机种子，默认 20260319")
    args = parser.parse_args()

    cif_files = find_cif_gz_files(args.input)
    if not cif_files:
        raise FileNotFoundError(f"No .cif.gz files found in: {args.input}")

    worker_args = [(cif_path, args.seed + i) for i, cif_path in enumerate(cif_files)]

    if args.processes > 1:
        with Pool(processes=args.processes) as pool:
            results = pool.map(count_residues_from_random_block, worker_args)
    else:
        results = [count_residues_from_random_block(x) for x in worker_args]

    total_counter = Counter()
    accepted_n = 0
    skipped_n = 0

    for r in results:
        if r["accepted"]:
            accepted_n += 1
            total_counter.update(r["counter"])
        else:
            skipped_n += 1

    total_count = sum(total_counter.get(aa, 0) for aa in AA_ORDER)

    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aa3", "aa1", "count", "frequency"])

        for aa3 in AA_ORDER:
            count = total_counter.get(aa3, 0)
            freq = count / total_count if total_count > 0 else 0.0
            writer.writerow([aa3, AA3_TO_AA1[aa3], count, f"{freq:.6f}"])

        writer.writerow(["TOTAL", "-", total_count, "1.000000" if total_count > 0 else "0.000000"])

    print(f"Found cif.gz files: {len(cif_files)}")
    print(f"Accepted files: {accepted_n}")
    print(f"Skipped files: {skipped_n}")
    print(f"Total residues counted: {total_count}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()

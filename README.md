## CPSea2: Synthetic data supporting both linear and cyclic peptide design

![CPSea2 overview](Figures/MainFigure.png)

😊 Hi there, this is  **CPSea2**, a synthetic data framework for (cyclic) peptide binder design.

🗺️ **Roadmap**
* Dataset Download: [**CPSea2 at Hugging Face**](https://huggingface.co/datasets/YZY010418/CPSea2)
* A Model Demo: [**PepAny**](https://github.com/YZY010418/PepAny)
* Previous Dataset: [**CPSea**](https://github.com/YZY010418/CPSea)

This repo provides protocols for the curation of CPbase, CPcap, and the merging between them. We also provide scripts for some evaluations on the dataset. You may visit [**Hugging Face**](https://huggingface.co/datasets/YZY010418/CPSea2) for downloading constructed datasets, and you may find our [**PepAny**](https://github.com/YZY010418/PepAny) as a model demo to explore the feasibility of constructing cyclic peptides with the merging protocol. 

**CPSea2** is named after our previous synthetic dataset for cyclic peptide binder design named [**CPSea**](https://github.com/YZY010418/CPSea). 

💎 **What's inherited**:

* The idea of mimicking inter-chain interactions by intra-chain interfaces.
* Overall pipeline of screening plausible binder-receptor pairs.

💪 **What's new**:

* Expanding the scope of peptide binders to linear peptides.
* Re-designing filters to make synthetic data align better to experimental references.
* Constructing cyclic peptides by merging linear peptides and capping structures.

---

## 🔍 Get CPbase

Because we didn't perform minimization on CPbase, we provide the `ResID maps` and `Metadata files` that describes the residue ID of the ligand and receptor chains in AFDB as well as some basic properties of these complexes. We recommend to reconstruct complex structures after downloading the AFDB50 source data, because directly upload/download structure files would cost TB level transition, and actually the new information here is only the residue ID of ligand and receptor chains. The AFDB50 dataset and the CPbase ResID map can be downloaded by

```bash
# Install huggingface tool
pip install -U huggingface_hub

# Download metadata and ResID maps
hf download YZY010418/CPSea2 \
  --repo-type dataset \
  --revision main \
  --include "CPbase_AFDB/Metadata/*" "CPbase_AFDB/ResID_Map/*" \
  --local-dir ./CPSea2

# Download source data
wget https://steineggerlab.s3.amazonaws.com/foldseek/afdb50.tar.gz
tar -xvzf afdb50.tar.gz
```

Use [`scripts/rebuild_cpbase.py`](scripts/rebuild_cpbase.py) to select a **CPbase_AFDB subset** from metadata and reconstruct its complexes from full AFDB50 source PDBs. It requires Python 3.10+ and only the standard library. It crops existing coordinates using ResID maps; it does not rerun structure mining or energy minimization, and does not reconstruct CPbase_PDB.

Run the following commands from this repository's root. Set the metadata, ResID map, and extracted AFDB50 paths to your local directories (the examples use `./CPSea2/CPbase_AFDB/{Metadata,ResID_Map}` and `./afdb50`). Metadata and ResID inputs must be separate TSV or TSV.gz files/directories; directory scanning is recursive.

```bash
# Copy the selection template and edit its rules for your use case.
cp scripts/rebuild_cpbase_template.json cpbase_selection.json

# Preview selection and check metadata/ResID joins and source file paths.
python3 scripts/rebuild_cpbase.py \
  --metadata ./CPSea2/CPbase_AFDB/Metadata \
  --resid ./CPSea2/CPbase_AFDB/ResID_Map \
  --source-dir ./afdb50 \
  --config cpbase_selection.json \
  --output ./CPbase_subset_preview \
  --dry-run

# After inspecting selected.tsv and summary.json, reconstruct into a NEW directory.
python3 scripts/rebuild_cpbase.py \
  --metadata ./CPSea2/CPbase_AFDB/Metadata \
  --resid ./CPSea2/CPbase_AFDB/ResID_Map \
  --source-dir ./afdb50 \
  --config cpbase_selection.json \
  --output ./CPbase_subset \
  --gzip
```

`--dry-run` does not read source coordinates or validate their atoms. Each output directory must not already exist. `--gzip` writes compressed PDBs; omit it for plain PDBs.

**Choosing the subset.** Edit `rules` in `cpbase_selection.json`; rules run in order on the remaining complexes. Template thresholds are examples and can be changed.

| Rule | Purpose | Example |
| --- | --- | --- |
| `range` | Numeric filter | `{"op":"range","field":"Length","min":5,"max":12}` |
| `top` | Top N per cluster | `{"op":"top","field":"relBSA","n":3,"order":"desc"}` |
| `relative` | Fraction of cluster maximum | `{"op":"relative","field":"relBSA","stat":"max","mode":"ratio","min":0.9}` |
| `aggregate` | Cluster-size filter | `{"op":"aggregate","stat":"count","min":3}` |

Clusters are defined by `(Source_ID, Domain_ID, Cluster_ID)`. Apply cluster-size filters before `top`; remove `top` to keep all passing complexes. Use `{"rules":[]}` to select all rows.

**Inputs and results.** ResID maps must contain `Source_ID`, `Domain_ID`, `Cluster_ID`, `Ligand_ID`, `receptor_res_ids`, and `ligand_res_ids`. Source structures must retain the original residue numbering; use full `.pdb`/`.pdb.gz` files, not domain-cropped PDBs or tar archives. For nonstandard filenames, use `--source-manifest`; see `python3 scripts/rebuild_cpbase.py --help` for options.

Successful structures appear under `structures/<shard>/`, with receptor chain `R`, ligand chain `L`, and ACE/NME caps reconstructed from the recorded precursor residues. `selected.tsv` records selected complexes and paths; `summary.json` records counts after each rule and reconstruction totals; `errors.tsv` records failed complexes. A reconstruction run with failures exits with status 1, so selection alone does not imply successful reconstruction. The output also preserves `config.json` and an audit database, `selection.sqlite`.


AFDB-multimer has been released recently, but it is not included in the current version of CPSea2 because we have not yet found a separately downloadable package. Nevertheless, we expect that these data can be processed using a script similar to the PDB pipeline, with an additional pLDDT-based confidence filter. We plan to explore the inclusion of AFDB-multimer data in future work.

## 🎲 Sample CPcap

Generate and minimize CPcap conformers in batches, stopping when cluster growth saturates.

```bash
python SampleCPCap/CappingConformerAuto.py \
  --name cap_name --smiles 'smiles' --out out_path \
  -p 8 --schrodinger-run /path/to/schrodinger/run
```

| Option | Purpose | Example |
| --- | --- | --- |
| `--batch-size` | Conformers per batch | `--batch-size 2000` |
| `--max-batches` | Sampling limit | `--max-batches 80` |
| `--cluster-rmsd` | Clustering RMSD cutoff | `--cluster-rmsd 0.5` |
| `--min-cluster-increase`, `--patience` | Stop on low cluster growth | `--min-cluster-increase 5 --patience 3` |
| `--energy-keep-percent` | Keep lowest-energy fraction (%) | `--energy-keep-percent 90` |
| `--seed-start` | First batch seed | `--seed-start 123` |

Set `--energy-keep-percent 100` to disable energy filtering. Use `--help` for advanced sampling and minimization options.

## 🔗 Merge CPcap and CPbase
Now we can construct cyclic peptide binders by searching CPbase-CPcap pairs whose terminal amides align to each other.

Because CPcap conformers are saved as .sdf files, we need to define which cap atoms correspond to the two terminal amide geometries used for alignment. Add these atom names to `CappingIndex.yaml`. Note that the atom-name CSV generated during CPcap sampling can help you and the scripts to map the atom names to atom indexes.

```yaml
Thioether-A: # Cap name used by --cap-name
  mode_1: # Multiple modes can be listed if one cap has several possible terminal-amide definitions.
    C_ter: "C2, N1, C1, O1, C3" # NME CA, N -> last-residue C, O, CA
    N_ter: "C7, C6, O2, N2, C5" # ACE CA, C, O -> first-residue N, CA
```

Then align CPcap conformers to CPbase terminal geometries and save the accepted capped complexes:

```bash
python Merge/CappingAlignBatch.py --base path_to_CPbase_cif_dir --cap path_to_CPcapping_sdf_dir --index CappingIndex.yaml --cap-name Thioether-A -o output_path -p num_processors
```

Useful arguments include:

* `--rmsd-cutoff` sets the maximum anchor RMSD accepted after alignment, for example `--rmsd-cutoff 0.5`.
* `--distance-cutoff` filters candidates by the 9D terminal descriptor distance before RMSD alignment, for example `--distance-cutoff 3.0`.
* `--clash-threshold` controls the VDW-scaled clash rejection threshold, for example `--clash-threshold 0.8`.
* `--cyclic-type-name` changes the output file prefix when you want it to differ from `--cap-name`.
* `--save-all` keeps all clash-free hits in the retrieved candidate set instead of only the default accepted outputs.

The merge step writes capped PDB complexes, a metadata CSV containing descriptor distances and anchor RMSD values for saved structures, and a summary CSV for monitoring progress.

Finally, relax the merged complexes with Schrodinger to remove local geometric irregularities:

```bash
$SCHRODINGER/run python3 SampleCPCap/relax_light.py -i merged_pdb_dir -o relaxed_pdb_dir -p num_processors
```

The relaxed output directory contains the final cyclic peptide binder complexes.


## 📈 Evaluations

We evaluate complex structures using tools including BSA, PLIP, GRAVY, Vina and FoldX. The SASA, PLIP, Vina, and FoldX scripts use `--input-type` to select the expected input layout, such as `pdb`, `cpsea2`, `cpsea2afdb`, `pepbench`, or `pdb-chain`. We also evaluate redundancy by clustering complexes based on receptor sequences using MMSeqs2. These analysis scripts can found in the `Evaluations` directory.

## 📞 Contact

Please contect yangzy23@mails.tsinghua.edu.cn if you have any questions. Thank you for your interest in our CPSea2!
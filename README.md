## CPSea2: Combine linear and cyclic peptide data for Arbitrary Cycliczation

![CPSea2 overview](Figures/MainFigure.png)

😊 Hi there, this is the repo for **CPSea2**, a framework for data synthesis of cyclic peptide binders with diverse terminal structures.

🗺️ This repo provides protocols for the curation of CPbase, CPcap, and the merging between them. We also provide scripts for some evaluations on the datasets. You may visit [**Hugging Face**](https://huggingface.co/datasets/YZY010418/CPSea2) for downloading constructed datasets, and you may find our generative model for structurally diverse and geometry plausible cyclic peptide binder design trained on CPbase_PDB named [**PepAny**](https://github.com/YZY010418/CPSea2). 

CPSea2 is named after our previous synthetic dataset for cyclic peptide binder design named [**CPSea**](https://github.com/YZY010418/CPSea). With the core idea of *mimicking inter-chain interactions by intra-protein interfaces*, we upgrade the whole pipeline aiming to be compatible with diverse cyclization structures with better cost-effectiveness and structural rationality:

💎 **What's inherited**:

* The idea of mimicking inter-chain interactions by intra-chain interfaces.
* Overall pipeline of screening plausible binder-receptor pairs.

💪 **What's new**:

* Constructing cyclic peptides by merging linear peptides and capping structures, insead of declare covalent bonds and expect energy minimization to recovery reasonable conformations.
* Expanding the scope of peptide binders to linear peptides, leading to a size increasement of over 100 times.
* Re-designing filters to make synthetic data align better to experimental references, such as contact density and interface hydrophobicity.

## 🔍 Find CPbase

As described in our paper, we pre-processed PDB and AFDB to remove redundancy and discard regions with low prediction confidence and rigidity, resulting in PDB70 and AFDB50_domains. We believe similar datasets could be generated following the described protocol, and please contact us for our source datasets used in our paper.

Because PDB structures have issues such as multiple chains, ins_code, missing atoms, ncAA, alternative conformations, etc., we introduced more guarding scripts for PDB, while AFDB is much easier. You may reproduce our curation of datasets by running:

```bash
python find_cpbase_mp_PDB.py -i source_data_names.list --root source_data_dir --saved_dir output_path --n_cpu num_processors_for_multiprocessing

python find_cpbase_mp_AFDB.py -i source_data_names.list --root source_data_dir --saved_dir output_path --n_cpu num_processors_for_multiprocessing
```

where source_data_names.list should be a list file with a stem name of a source data file in each row, and source_data_dir is the dir saving source data files (in the format of .pdb). The script will generate a metadata file and a directory saving the identified structures.

AFDB_multimer has been released recently, and has not been involved in current version of CPSea2 yet because we have not found a separate package for download. Nonetheless, we believe these data can be employed by a similar script as that used for PDB, with an additional filter on pLDDT. We will try including these data in future work.

## 🎲 Sample CPcap

We first generate plausible conformations for capping structures, then align and merge them with CPbase peptides. Alternatively, one may generate linear peptide binders as CPbase, so that more diverse terminal relative positions can be involved. Related discussions and instructions can be found in the **PepAny** repo.

To automatically get a proper number of conformation generation, we sample conformations in batch and check the number of clusters increased after a new batch of conformations is generated. When the increase number declines to a certain extent, we suppose that the plausible conformations have been adequately explored, and stop conformation sampling.

You may run this protocol for conformation sampling:

```bash
python CappingConformerAuto.py --name name_of_the_capping --smiles 'smiles' --out out_path -p num_processors_used_in_minimization_and_clustering --schrodinger-run path_for_schrodinger/run
```

You may also control the size of each batch, max num_batch, patience that controls the stop point selection, and some parameters in schrodinger minimization. 

Generated conformations are saved as sdf files. You need to mannually add the atom names of the terminal amides used for downstream alignment in `CappingIndex.yaml`, in the format below:

```yaml
Thioether-A: # The name of the capping structures
  mode_1: # maybe in the same structure, there are multiple possibility to define terminal amides. E.g. some amide pairs can either be the terminal amides for merging, or be the ACE/NME capping groups.
    C_ter: "C2, N1, C1, O1, C3" # should be in the order of NME CA, N -> LastAA C, O, CA
    N_ter: "C7, C6, O2, N2, C5" # should be in the order of ACE CA, C, O -> FirstAA N, CA
```

Now we can align and merge CPbase and CPcapping by running:

```bash
python CappingAlignBatch.py --base path_to_CPbase_cif_dir --cap path_to_CPcapping_sdf_dir --index CappingIndex.yaml --cap-name The_name_in_yaml -o output_path -p num_processors
```

This would generate many merged pdb files, a metadata file containing the descriptor distances and rmsd for each accepted pair, and a summary file for monitoring the running process.

We then used Schrodinger for energy minimization to eliminate potential geometric irregularities.

```bash
python $Schrodinger/run python3 relax_light.py -i input_dir -o output_dir -p num_processors
```

Now we have finished the merging process and have cyclic peptide binders in hands.

## 🔗 Merge CPcap and CPbase


## 📈 Evaluations

We evaluate complex structures using tools including BSA, PLIP, GRAVY, Vina and FoldX. We also evaluate the redundancy by clustering complexes based on receptor sequences using MMSeqs2. These analysis scripts can be used as following:

```bash
python CalcSASA.py -i input_pdb_dir -o output_csv -n num_processors
python plip_analysis.py -i input_pdb_dir -o output_csv -n num_processors
python CalcGRAVY.py -i input_pdb_dir -o output_csv -n num_processors
python Vina_analysis.py -i input_pdb_dir -o output_csv -n num_processors (--score_only)
python FoldX_analysis.py -i input_pdb_dir -o output_csv -p num_processors --repair --include_counts
python extract_receptor_sequences.py -i input_pdb_dir -o output_fasta -n num_processors
mmseqs easy-cluster fasta_file clusterRes tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
```

These results are further processed and intepreted as described in our paper.

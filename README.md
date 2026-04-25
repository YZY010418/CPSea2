## CPSea2: Combine linear and cyclic peptide data with more plausible structure

😊 Hi there, this is the repo for **CPSea2**, an *"upgraded"* synthetic dataset for cyclic peptide binder design from the previous [**CPSea**](https://github.com/YZY010418/CPSea), aiming to be compatible with diverse cyclization structures with better cost-effectiveness and structural rationality.

💎 **Core Consistency**:

* The idea of mimicking inter-chain interactions by intra-chain interfaces.
* Overall framework of how we screen out plausible binders.

💪 **Major Upgrades**:

* The idea of generating cyclic peptides by merging a linear peptide and a capping structure, enabling arbitraray cyclization structures with more reasonable geometry.
* Re-designed filters enabling more plausible synthetic complex data that align better to experimental datasets.

🗺️ This repo provides protocols for the curation of CPSea2, the merging between CPSea2 and CPcapping, and some metrics for evaluations on the dataset. You may download constructed CPSea2 in [**Zenodo**](https://github.com/YZY010418/CPSea2), and you may find the generative model trained on CPSea2 in [**PepAny**](https://github.com/YZY010418/CPSea2). 


## 🔍 FindCPBase

As described in our paper, we pre-processed PDB and AFDB to remove redundancy and discard regions with low prediction confidence and rigidity, resulting in PDB70 and AFDB50_domains. We believe similar datasets could be generated following the described protocol, and please contact us for our source datasets used in our paper.

Because PDB structures have issues such as multiple chains, ins_code, missing atoms, ncAA, alternative conformations, etc., we introduced more guarding scripts for PDB, while AFDB is much easier. You may reproduce our curation of datasets by running:

```bash
python find_cpbase_mp_PDB.py -i source_data_names.list --root source_data_dir --saved_dir output_path --n_cpu num_processors_for_multiprocessing

python find_cpbase_mp_AFDB.py -i source_data_names.list --root source_data_dir --saved_dir output_path --n_cpu num_processors_for_multiprocessing
```

where source_data_names.list should be a list file with a stem name of a source data file in each row, and source_data_dir is the dir saving source data files (in the format of .pdb). The script will generate a metadata file and a directory saving the identified structures.

AFDB_multimer has been released recently, and has not been involved in current version of CPSea2 yet because we have not found a separate package for download. Nonetheless, we believe these data can be employed by a similar script as that used for PDB, with an additional filter on pLDDT. We will try including these data in future work.

## 🔗 ApplyCPCapping

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

## 📈 Evaluations
We evaluate structures in the following aspects:
### SufficientContact

## 📞 Contact
If you have any questions, comments, or data requirements, please contact yangzy23@mails.tsinghua.edu.cn. We look forward to hear from the community!

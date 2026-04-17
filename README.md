## CPSea-2: Combine linear and cyclic peptide data with more plausible structure
The idea of constructing cyclic peptides
* **FindCPBase**. To synthesize complex structure data between linear peptide ligands and protein receptors. We provide scripts for both AFDB and PDB, where PDB have more complicated processings because there are more variations.
* **ApplyCPCapping**. First, plausible conformations of the capping structure of cyclic peptides are sampled. To merge the "CPBase" and "CPCapping" by searching the most aligned relative position of terminal amides, and check if additional clashes are introduced. This algorithm can be used both in the data preparation and after model generation.

## Receptor sequence extraction
Use [extract_receptor_sequences.py](/Users/bytedance/Desktop/CPSea2/extract_receptor_sequences.py) to scan a directory of multi-block `.cif.gz` files, extract the receptor sequence from the first structure block in each file, and write a FASTA for downstream `mmseqs` clustering.

Example:

```bash
python3 extract_receptor_sequences.py \
  -i /path/to/cif_gz_dir \
  -o /path/to/receptor_sequences.fasta \
  -c 32
```

The script writes failed parses to `OUTPUT_FASTA.errors.tsv`.

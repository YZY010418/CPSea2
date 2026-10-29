# AFDB residue ID repair

The AFDB writer now emits six columns:

```
Source_ID  Domain_ID  Cluster_ID  Ligand_ID  receptor_res_ids  ligand_res_ids
```

The columns are tab-separated. Input PDB names must end in `_<Domain_ID>.pdb`.
The writer splits at the final underscore, preserving both zero and multi-digit
domain numbers. It refuses to append this schema to an old five-column TSV.
Previously, `source_id[:-2]` discarded a fixed number of characters, losing the
domain identity and leaving a trailing underscore for two-digit domain IDs.

`repair_afdb_resid.py` repairs the known `Source_ID_ / NA` corruption only. It
requires original metadata and source domain PDBs, and rejects ambiguous or
unmatched records. Do not replace every `NA` with zero.

Prepare an anomaly file with no header: original TSV path, physical line number,
then all six original fields, separated by tabs. Prepare a metadata evidence file
containing original AFDB metadata rows for the affected proteins, including all
candidate domains. Keep the original TSVs unchanged during repair.

```bash
python scripts/repair_afdb_resid.py resolve \
  anomalies.tsv metadata_evidence.tsv /path/to/AF2_domains patches.json
python scripts/repair_afdb_resid.py apply patches.json /path/to/new_fixed_directory
```

Resolution checks the complete metadata ligand ID, cluster and ligand numbers,
core ligand endpoints and length, the historical truncation signature, and the
presence of every receptor/ligand residue in the candidate PDB. Ligand backbone
atoms must also be present. Exactly one candidate must satisfy these checks.

Application checks every original row against the manifest and writes only
affected shards to a new directory. It re-reads each output and reverses the
listed changes in memory: the resulting SHA-256 must equal the original input
SHA-256. Each completed file has an `.audit.json` sidecar. A `.partial` file is
not a verified result. Existing outputs are never overwritten. Original source
files and Hugging Face files are not changed by this utility.

Run the regression tests without the structure-mining dependencies:

```bash
python -m unittest discover -s tests -v
```

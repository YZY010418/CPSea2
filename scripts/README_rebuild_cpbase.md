# Reconstruct a CPbase subset from AFDB50

`rebuild_cpbase.py` selects complexes using metadata, joins them to ResID maps,
and crops **full AFDB structures** into receptor–ligand PDB files. It runs with
Python 3.10+ and the standard library; it does not run structure mining or relax.

Two levels are supported:

- **Complex**: one `LigandID`, identified by
  `(Source_ID, Domain_ID, Cluster_ID, Ligand_ID)`.
- **Cluster**: `(Source_ID, Domain_ID, Cluster_ID)`, the intermediate group of
  complexes in CPbase. This is not a higher-level receptor sequence cluster.

Domain IDs are used for joining records and distinguishing clusters. **They are
not appended to source filenames.** Source files must use the same residue
numbering and coordinates as the structures used to generate the ResID map.

## Run

Download and extract the AFDB50 source PDBs, CPbase_AFDB metadata, and ResID maps.
Pass metadata and ResID directories separately. Directories are scanned
recursively for `.tsv` and `.tsv.gz`; multiple files/directories can also be given.

```bash
python scripts/rebuild_cpbase.py \
  --metadata /data/CPbase_AFDB/Metadata \
  --resid /data/CPbase_AFDB/ResID_Map \
  --source-dir /data/afdb50 \
  --config scripts/rebuild_cpbase_template.json \
  --output /data/CPbase_subset_preview \
  --dry-run
```

`--dry-run` performs selection, checks joins and resolves source paths; it does
not read source coordinates or write PDBs. Inspect `selected.tsv` and
`summary.json`, then run without `--dry-run`, using a **new output directory**.
Add `--gzip` to write `.pdb.gz` instead of `.pdb`.

The example thresholds are illustrative and should be edited for your use case.
Use `{"rules": []}` to select every metadata row. There is **no automatic
one-per-cluster selection** unless a `top` rule requests it.

## Ordered selection rules

Rules run in JSON list order. Every rule sees only the complexes retained by
previous rules. Thus cluster statistics/ranks are recalculated after each step.
All bounds within one rule are combined with AND, as are successive rules.

Numeric metadata columns, including custom numeric columns, can be used as
`field`. `min`/`max` are inclusive; `gt`/`lt` are exclusive. At least one bound
is required for `range`, `relative`, and `aggregate`. Omit an unneeded bound.

### 1. Absolute range on each complex

```json
{"op": "range", "field": "Length", "min": 5, "max": 12}
```

For binary columns, `"min": 0, "max": 0` selects false. For example:

```json
{"op": "range", "field": "InterchainDisulfide", "min": 0, "max": 0}
```

### 2. Rank complexes within each cluster

Keep the three largest `relBSA` values per cluster:

```json
{"op": "top", "field": "relBSA", "n": 3, "order": "desc"}
```

Keep the lowest 20% of `GRAVY` values per cluster:

```json
{"op": "top", "field": "GRAVY", "fraction": 0.2, "order": "asc"}
```

Choose exactly one of `n` or `fraction`. Fractions are in `(0, 1]` and round up
per cluster. `order` defaults to `desc`. `ties` defaults to `truncate`, using
lexicographic `LigandID` order as a deterministic tie-breaker. Set
`"ties": "include"` to keep every complex tied at the cutoff (possibly more
than the requested count). Ranking excludes missing values.

### 3. Relative complex metric within each cluster

Keep complexes whose `relBSA` is at least 90% of their cluster's maximum:

```json
{"op": "relative", "field": "relBSA", "stat": "max", "mode": "ratio", "min": 0.9}
```

Keep complexes with `GRAVY` at or below their cluster mean:

```json
{"op": "relative", "field": "GRAVY", "stat": "mean", "mode": "difference", "max": 0}
```

Supported reference statistics: `max`, `min`, `mean`, `sum`.
`ratio` computes `value / reference`; `difference` computes `value - reference`.
Defaults are `stat=max`, `mode=ratio`. Use differences when ratios are not useful
for signed metrics. Ratios with a zero reference are undefined and excluded.

### 4. Retain/discard whole clusters by aggregate metrics

Keep all current members of clusters with at least three current complexes:

```json
{"op": "aggregate", "stat": "count", "min": 3}
```

Keep clusters with mean `relBSA >= 0.5`:

```json
{"op": "aggregate", "field": "relBSA", "stat": "mean", "min": 0.5}
```

Supported statistics: `count`, `min`, `max`, `mean`, `sum`.
`count` without `field` counts complexes; with `field`, it counts nonmissing
values. Other aggregates ignore missing values. An undefined aggregate does not
pass bounds. A passing aggregate rule retains all current members of that
cluster, including members whose individual metric was missing. Use a `range`
rule first if each member must have a valid metric.

`top`, `relative`, and `aggregate` accept `"scope": "cluster"` (the default).
There is no additional sequence-clustering input or hierarchy.

**Order matters:** put an aggregate rule before a top-1 rule to assess cluster
size before taking one representative. Put it after the top-1 rule and each
nonempty cluster has size one. Likewise, relative selection after an absolute
range compares only to the surviving members.

## Input schemas

### Metadata

Headered TSVs must include `LigandID`. Other columns retain their header names;
use these exact names in rules. Names must start with a letter and contain only
letters, digits and underscores. `LigandID` must follow:

```text
<Source_ID>_<Domain_ID>_cluster<Cluster_ID>_ligand<Ligand_ID>
```

The existing headerless 15-column AFDB metadata is interpreted as:

```text
LigandID OriginalPosition Length absBSA relBSA raBSA BSA_per_length rel_n_rBSA rel_c_rBSA GRAVY HelixRatio SheetRatio LigandAllUpstream LigandAllDownstream InterchainDisulfide
```

For 16-column files, `SeparateChain` is inserted immediately after `SheetRatio`.
All metadata shards must share the same schema. For a different headerless
schema, add a `metadata_columns` list to the config in the exact column order.
Headered inputs take precedence over this list.

Known numeric columns and every rule's `field` are parsed as finite numbers.
Missing tokens (empty, `NA`, `N/A`, `NULL`, `NONE`, `NAN`, `.`) become null;
other malformed numbers are errors. `LigandID`/`OriginalPosition` are identifiers,
not numeric filter fields. Duplicate complex IDs or duplicate numeric identities
are errors rather than silently counted twice.

### ResID maps

A header is required:

```text
Source_ID\tDomain_ID\tCluster_ID\tLigand_ID\treceptor_res_ids\tligand_res_ids
```

Residues use comma-separated numbers and inclusive ranges, e.g. `1-8,12,15-19`.
`ligand_res_ids` **includes both cap precursor residues**. The smallest ligand
ID supplies ACE atoms `CA,C,O`; the largest supplies NME atoms `N,CA`; interior
residues form the ligand core. This matches the current CPbase AFDB writer.

Each selected metadata record must match exactly one ResID row using all four
identity fields. Missing selected rows or duplicate selected ResID records are
errors. Invalid domain/cluster/ligand values (including the historical `NA`
corruption) must be repaired before use; domain IDs are never guessed.

### Full AFDB source PDBs

Recursive lookup accepts `<Source_ID>.pdb` and `<Source_ID>.pdb.gz`. For a source
such as `AF-P12345-F1`, it also accepts `AF-P12345-F1-model_v4.pdb[.gz]` (any
integer version). The `F1` fragment identity is preserved. If several files or
versions match one requested source, the run stops instead of choosing one.

For differently named files, provide `--source-manifest sources.tsv`, which
replaces automatic directory lookup. This tab-separated file has columns:

```text
Source_ID\tpath
AF-P12345-F1\t/data/afdb50/AF-P12345-F1-model_v4.pdb
```

Relative paths in that manifest are relative to its directory. The source
manifest can map multiple domain groups to the same complete AFDB PDB through
one Source_ID. A source path must exist. Input tar archives/mmCIF files are not
supported directly: use extracted PDB or PDB.gz files.

## Reconstruction and outputs

Coordinates and residue numbers are preserved. Output chains are receptor `R`
and ligand `L`. Cap residues are named ACE/NME, with source atom names retained.
Only the first PDB model is read. Missing backbone/cap atoms, overlapping
receptor/ligand selections, noncontiguous ligand ranges, insertion codes,
ambiguous chains, and duplicate atoms fail that complex. AFDB monomer inputs
are expected; alternate-location codes other than blank/A are rejected.
No atoms are synthesized, coordinates optimized, or bonds inferred.

A run creates:

- `structures/<shard>/<LigandID>.pdb[.gz]`: successful structures, with domain in
  the output ID to prevent collisions between distinct domain-derived complexes.
- `selected.tsv`: selected metadata, all four identity fields, ResID ranges,
  source path, and intended output path. Failed entries remain in this manifest;
  consult `errors.tsv` for status.
- `errors.tsv`: complex-specific reconstruction failures; a run with failures
  exits with status 1. Input/schema/join failures stop before reconstruction.
- `summary.json`: per-rule before/after counts and selected/written/failed counts.
- `config.json`: the configuration used.
- `selection.sqlite`: disk-backed selected metadata and joins for inspection.

The output directory must not already exist. No existing results are overwritten.
SQLite and streamed TSV reading keep the complete metadata collection off the
Python heap. Temporary SQLite files and its database require disk space;
filtering and reconstruction are currently single-process. Full AFDB50 runtime
has not been benchmarked. Dry-run and reconstruction in separate directories
perform separate scans. The saved database is an audit artifact, not a resume
cache.

Run focused tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_rebuild_cpbase.py' -v
```

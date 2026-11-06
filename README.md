<p align="center">
  <img src="mojoeMD_logo.png" alt="mojoeMD_logo" width="420">
</p>

# mojoeMD

`mojoeMD` is a JSON-driven molecular dynamics workflow for OpenMM. It prepares solvated systems, equilibrates them, runs production MD, and includes companion tools for trajectory analysis, contact analysis, interaction-energy screening, and noncanonical amino acid (NCAA) residue-template generation.

The project is aimed at protein systems that may include:

- Standard proteins
- Protein-ligand complexes
- Membrane protein-ligand complexes
- Glycoproteins with GLYCAM templates
- Embedded noncanonical amino acids supplied as OpenMM `ffxml`
- Combinations of glycoproteins, ligands, and NCAAs

## Repository Layout

```text
mojoeMD.py                                      Main OpenMM prepare/equilibrate/production pipeline
analysis/analyse_MD.py                         Trajectory post-processing and structural analyses
analysis/contact_analysis.py                   getcontacts wrapper, contact mapping, optional energies/plots
ncaa/generate_ncaa_ffxml_from_amber_mol2.py    NCAA ffxml/hydrogen-definition generator
examples/*.json                                Example simulation configs
analysis/*.json                                Example analysis/contact-analysis configs
examples/ncaa/                                 Example generated NCAA files
examples/ncaa_work/                            Example NCAA intermediate files
```

## What The Main Pipeline Does

`mojoeMD.py` has three stages:

1. `prepare`
   - Loads a receptor PDB with PDBFixer.
   - Replaces ordinary nonstandard residues unless NCAA mode preserves them.
   - Adds hydrogens with OpenMM `Modeller`.
   - Applies metal-aware histidine/cysteine protonation when retained metals are nearby.
   - Optionally adds a ligand from SDF using OpenFF/RDKit and either Espaloma or SMIRNOFF.
   - Optionally detects and renames glycan residues for GLYCAM.
   - Optionally patches NCAA internal and peptide bonds from supplied `ffxml`.
   - Adds solvent and ions.
   - Writes `prepared.pdb` and `system.xml`.

2. `equilibrate`
   - Loads `prepared.pdb` and `system.xml`.
   - Minimizes energy.
   - Heats from 5 K to 300 K under NVT.
   - Runs NPT equilibration with decreasing backbone restraints.
   - Writes `equilibrated.pdb`, `equilibrated.xml`, and `equilibrated_system.xml`.

3. `produce`
   - Loads the equilibrated system/state.
   - Runs production MD, optionally with enhanced sampling (see
     [Enhanced Sampling](#enhanced-sampling) below): well-tempered metadynamics, bias-exchange
     metadynamics, or automated CDR-loop bias-exchange for antibody Fabs.
   - Writes `production.dcd`, `production.csv`, `production.chk`, and `production_final.chk`.
   - Supports restart from the newest production checkpoint with `--restart`.

## Installation

Create an environment with the scientific MD stack used by the scripts. Exact package names vary by platform, but a typical Conda setup is:

```bash
conda create -n mojoemd -c conda-forge python=3.11 openmm pdbfixer mdtraj numpy scipy scikit-learn tqdm pandas matplotlib
conda activate mojoemd
```

For ligand support, also install OpenFF/RDKit/OpenMM force-field generators:

```bash
conda install -c conda-forge openff-toolkit rdkit openmmforcefields
```

For contact analysis, install `getcontacts` separately and either put its commands on `PATH` or set `getcontacts.path` in `analysis/contact.json`.

For automated CDR-loop detection on antibody Fabs (see [Enhanced Sampling](#enhanced-sampling)), also install `anarcii`:

```bash
pip install anarcii
```

## Quick Start

Run a complete standard protein workflow:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json
```

Run only preparation:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --prepare
```

Run production from an existing equilibrated state:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --produce
```

Restart production from the latest checkpoint:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --produce --restart
```

Configuration values from JSON become command-line defaults. Any explicit CLI argument overrides the config value.

## Example Simulation Config

```json
{
  "receptor": "inputs/protein.pdb",
  "output_dir": "runs/vanilla_protein",
  "prepare": true,
  "equilibrate": true,
  "produce": true,
  "ph": 7.4,
  "water_model": "tip3p",
  "timestep_fs": 4.0,
  "hydrogen_mass": 1.5,
  "prod_ns": 100.0
}
```

Common fields:

- `receptor`: input PDB.
- `ligand`: optional ligand SDF.
- `ligand_resname`: optional 3-character ligand residue name.
- `output_dir`: run directory.
- `prepare`, `equilibrate`, `produce`: stage toggles.
- `prod_ns`: production length in nanoseconds.
- `timestep_fs`: integrator timestep.
- `hydrogen_mass`: hydrogen mass for H-mass repartitioning.
- `ph`: pH used during hydrogen addition.
- `water_model`: solvent model passed to `Modeller.addSolvent`.
- `water_ff`: OpenMM water force-field XML file.
- `membrane`: build a lipid membrane and water box with `Modeller.addMembrane`.
- `lipid_type`: lipid type for OpenMM membrane building, for example `POPC`.
- `lipid_ff`: OpenMM lipid force-field XML file used for membrane systems.
- `membrane_padding`: minimum membrane/water padding in nanometers.
- `membrane_center_z`: explicit membrane center along z in nanometers.
- `membrane_center_chain_ids`: optional chain IDs used to auto-center the membrane.
- `membrane_center_residue_range`: optional residue-id range used to auto-center the membrane.
- `membrane_auto_thickness`: hydrophobic slab thickness used for automatic TM-window detection.
- `metal_aware_protonation`: enabled by default; forces metal-aware residue variants such as histidine tautomers and deprotonated cysteine when donor atoms are close to a retained metal.
- `metal_coordination_restraints`: optional harmonic restraints between retained metals and nearby N/O/S donor atoms.
- `metal_coordination_cutoff`: metal-donor detection cutoff in Angstrom.
- `metal_coordination_k`: restraint force constant in kJ/mol/nm^2.
- `platform`: optional OpenMM platform name (`CUDA`, `OpenCL`, `CPU`, `Reference`). Defaults to
  auto-detecting the fastest usable platform; override this if auto-detection picks a platform
  that is compiled in but not actually usable on a given machine (seen in practice with some
  OpenCL driver setups that work for a trivial system but fail on a real solvated one).

## Protein-Ligand Systems

Use a config such as `examples/02_protein_ligand.json`:

```json
{
  "receptor": "inputs/protein.pdb",
  "ligand": "inputs/ligand.sdf",
  "ligand_resname": "LIG",
  "output_dir": "runs/protein_ligand",
  "prepare": true,
  "equilibrate": true,
  "produce": true,
  "lig_ff": "espaloma"
}
```

`lig_ff` can be:

- `espaloma`
- `smirnoff`

The ligand SDF must contain at least one conformer.

For metalloprotein or metal-coordinated ligand systems, use a config such as
`examples/10_metal_coordinated_protein_ligand.json`. Metal-aware protonation is
enabled by default, and that example also enables harmonic metal coordination
restraints.

## Enhanced Sampling

The production stage supports three mutually exclusive enhanced-sampling modes on top of
plain MD, all built on OpenMM's `openmm.app.metadynamics` and all biasing backbone phi/psi
torsions. Each is configured with its own top-level JSON key; omit all three for plain MD.

### Well-tempered metadynamics

Bias 1-3 backbone torsions in a single walker. Useful for accelerating sampling of a specific
known loop or side-chain-adjacent backbone motion.

```json
"metadynamics": {
  "enabled": true,
  "cvs": [
    {"type": "phi", "chain": "A", "residue": 45, "bias_width_degrees": 25},
    {"type": "psi", "chain": "A", "residue": 45, "bias_width_degrees": 25}
  ],
  "bias_factor": 10.0,
  "height_kjmol": 1.2,
  "deposition_interval_ps": 1.0,
  "save_interval_ps": 100.0
}
```

```bash
python mojoeMD.py -c examples/12_metadynamics_phi_psi.json
```

Outputs per run: `metad_bias/` (bias checkpoints, also used to resume with `--restart`),
`metad_cv.csv` (CV trajectory in degrees), `metad_free_energy.npy` + `metad_free_energy_axes.json`
(free-energy surface, written at the end of the run).

### Bias-exchange metadynamics (BE-META)

Run several walkers in one process, each biasing its own set of torsions (any number per
walker — more than 3 automatically splits into multiple internal bias groups), periodically
attempting Metropolis-criterion swaps of full configurations between walkers. Useful when you
want several CVs explored in parallel without forcing them into one shared bias.

```json
"bias_exchange": {
  "enabled": true,
  "exchange_interval_ps": 2.0,
  "walkers": [
    {"cvs": [{"type": "phi", "chain": "A", "residue": 45, "bias_width_degrees": 25}]},
    {"cvs": [{"type": "psi", "chain": "A", "residue": 45, "bias_width_degrees": 25}]}
  ]
}
```

```bash
python mojoeMD.py -c examples/13_bias_exchange_metadynamics.json
```

Outputs per run: one subdirectory per walker (`walker_00/`, `walker_01/`, ... or a custom
label), each with its own `production.dcd`/`.csv`/checkpoints, `metad_bias/`, `metad_cv.csv`,
and per-bias-group free-energy files; plus a top-level `bias_exchange_log.csv` recording every
exchange attempt (Δ and accept/reject) for monitoring exchange acceptance rates.

### Automated CDR-loop bias-exchange for antibody Fabs

For an antibody Fab or similar, automatically detects the CDR loop residues with
[`anarcii`](https://github.com/oxpig/ANARCII) (antibody numbering, no manual residue picking
required) and runs one bias-exchange walker per CDR loop, biasing every backbone phi/psi
torsion in that loop together. Heavy/light chain identity is auto-detected from sequence by
`anarcii`, not assumed from the input PDB's own chain IDs — useful since those can be
mislabeled or inconsistent across deposited structures.

```json
"auto_cdr_bias_exchange": {
  "enabled": true,
  "loops": ["H1", "H2", "H3", "L1", "L2", "L3"],
  "bias_width_degrees": 25,
  "bias_factor": 10.0,
  "exchange_interval_ps": 2.0
}
```

```bash
python mojoeMD.py -c examples/14_auto_cdr_bias_exchange_fab.json
```

`loops` defaults to all six CDRs (`H1`/`H2`/`H3`/`L1`/`L2`/`L3`, IMGT-numbered) and can be
restricted to a subset, e.g. `["H3"]` for just the heavy-chain CDR3 loop. Each requested loop
becomes one walker labeled `CDR-H1`, `CDR-H2`, etc., with its own output subdirectory, following
the same bias-exchange output layout described above. CDR residue detection runs once at the
start of production and is logged (loop name, number of residues, number of torsions biased);
a loop with no detected residues is skipped with a warning rather than failing the run.

## Membrane Systems

Use a config such as `examples/11_membrane_ligand.json`:

```json
{
  "receptor": "inputs/membrane_protein.pdb",
  "ligand": "inputs/ligand.sdf",
  "output_dir": "runs/membrane_ligand",
  "prepare": true,
  "membrane": true,
  "lipid_type": "POPC",
  "lipid_ff": "amber14/lipid17.xml",
  "water_ff": "amber14/tip3p.xml",
  "membrane_center_z": null,
  "membrane_center_chain_ids": [],
  "membrane_center_residue_range": null,
  "membrane_auto_thickness": 3.0
}
```

Membrane mode prepares the protein and optional ligand first, then calls OpenMM
`Modeller.addMembrane()` to build the lipid bilayer, water, and ions around the
complex. The input structure should already be oriented for a membrane in the XY
plane.

If `membrane_center_z` is set, that explicit z value is used. If it is `null`
and no chain/range selection is provided, mojoeMD scans the oriented structure
for a hydrophobic, z-spanning sequence window and centers the bilayer on that
predicted transmembrane slab. `membrane_auto_thickness` controls the slab window
in nanometers. If the heuristic is wrong for a difficult structure, set
`membrane_center_chain_ids` and `membrane_center_residue_range` to anchor the
center on known transmembrane residues.

OpenMM's built-in membrane builder creates a single-lipid membrane from the
selected `lipid_type`. For mixed or asymmetric membrane compositions, build the
membrane externally and then use mojoeMD with force-field files that support the
lipid residue names in that prepared structure.

## Glycoproteins

Set `glycoprotein: true` and provide GLYCAM resources:

```json
{
  "glycoprotein": true,
  "glycam_xml": "amber14/GLYCAM_06j-1.xml",
  "glycam_hydrogens": "glycam-hydrogens.xml"
}
```

The preparation code detects supported glycan links from PDB `LINK` records and/or distance heuristics, assigns GLYCAM templates, renames glycosylated ASN residues to `NLN` variants, and writes glycan diagnostics under:

```text
<output_dir>/glycan_prepare/
```

Useful glycoprotein debug flags:

```bash
python mojoeMD.py -c examples/09_prepare_only_debug_glycoprotein.json
```

Debug output includes a pre-hydrogen PDB and residue/template bond dumps.

## Noncanonical Amino Acids

NCAA mode preserves explicitly selected nonstandard residues and loads matching
OpenMM residue templates. Identify each target by residue name, chain ID,
residue number, and insertion code where applicable. Do not assume every
`HETATM` residue is an NCAA: crystallographic ligands, terminal modifications,
metals, and known modified amino acids require separate decisions.

Before parameterization, inspect all nonstandard residues and determine whether
each one should be preserved, replaced, removed, or parameterized. For example,
Amber14 already contains force-field templates for hydroxyproline (`HYP`), but
OpenMM's default hydrogen definitions may still be insufficient to add its
hydrogens. Conversely, an unsupported terminal residue such as pyroglutamate
(`PCA`) requires a terminally appropriate custom template if it is retained.

A preparation configuration looks like:

```json
{
  "ncaa": true,
  "ncaa_resnames": ["NCA"],
  "ncaa_ffxml": ["ffxml/NCA.xml"],
  "ncaa_mode": "ffxml"
}
```

During preparation, the pipeline:

- Tells PDBFixer which nonstandard residue names must remain unchanged.
- Loads custom NCAA `ffxml` residue templates.
- Loads sibling hydrogen-definition files named like `NCA_hydrogens.xml` when present.
- Patches missing internal NCAA bonds from the residue `Bond` records in the `ffxml`.
- Patches peptide bonds adjacent to NCAA residues.
- Adds hydrogens with OpenMM `Modeller` and requires every residue to match a
  loaded force-field template before solvation.

Names listed in `ncaa_resnames` are preserved. If a known modified residue must
remain present, list it too—even when its force-field template is already part
of Amber—or PDBFixer may replace it. Only custom XML files belong in
`ncaa_ffxml`.

## Generating NCAA Templates

The helper extracts a local charge fragment and converts an AmberTools MOL2
containing AM1-BCC charges into an OpenMM residue template. The extracted
fragment normally includes the NCAA plus neighboring residues so that its
charges reflect the peptide environment.

Extract a fragment first:

```bash
python ncaa/generate_ncaa_ffxml_from_amber_mol2.py \
  --protein-pdb inputs/protein_ncaa.pdb \
  --resname ALM \
  --chain A \
  --resid 42 \
  --extract-fragment \
  --fragment-only \
  --workdir ncaa_work
```

Before running Antechamber, ensure the fragment contains its complete and
chemically correct hydrogen set. Antechamber assigns charges and atom types; it
does not reliably add missing hydrogens. A heavy-atom-only fragment can produce
the wrong net charge, an odd-electron calculation, or an incomplete OpenMM
template.

Run Antechamber on the complete fragment, explicitly setting the charge of the
**whole extracted fragment**, not merely the target residue. One representative
command is:

```bash
antechamber \
  -i ncaa_work/ALM_fragment_complete_h.pdb -fi pdb \
  -o ncaa_work/ALM_fragment_amber.mol2 -fo mol2 \
  -c bcc -nc 0 -at gaff2 -j 5 -dr no -pf y -s 2
```

Check the electron count and SQM convergence rather than changing `-nc` merely
to silence an error. If altered SQM controls are scientifically justified, put
them explicitly in the command and retain the command and `sqm.out` with the
parameterization record.

Then build the template:

```bash
python ncaa/generate_ncaa_ffxml_from_amber_mol2.py \
  --protein-pdb inputs/protein_ncaa.pdb \
  --mol2 ncaa_work/ALM_amber.mol2 \
  --resname ALM \
  --chain A \
  --resid 42 \
  --outdir ncaa \
  --target-charge 0 \
  --charge-correction uniform \
  --type-source protein \
  --atom-type-map ncaa_work/ALM_atom_type_map.json \
  --force
```

Outputs:

- `ncaa/ALM.xml`
- `ncaa/ALM_hydrogens.xml`
- `ncaa/ALM_charge_report.json`

### Charges and atom types

AM1-BCC supplies partial charges, but it does not supply the bonded and
Lennard-Jones parameters needed by OpenMM. The generator therefore needs atom
types already defined by a loaded force field:

- Prefer an explicit `--atom-type-map` based on the NCAA's chemistry.
- `--type-source protein` can reuse suitable ff14SB types for genuinely
  protein-like functional groups.
- `--allow-approximate-types` enables heuristic GAFF/MOL2-to-ff14SB mappings.
  This is a compatibility aid for testing, not automatic publication-grade
  parameterization.
- `--type-source mol2` emits prefixed types, but those types are useful only
  when a separate force-field XML defines their masses and nonbonded/bonded
  parameters.

Set `--target-charge` to the intended formal charge of the NCAA residue itself.
The generator reports the raw charge extracted from the MOL2, the correction
method, and the final residue-template charge. Neutralization is a deliberate
chemical choice, not a default that is correct for every NCAA.

Charge results can depend on neighboring residues and protonation state. If the
same NCAA appears in substantially different chemical contexts, parameterize
and review each context independently or deliberately adopt one transferable,
validated parameter set.

### Required validation

Do not consider XML generation alone a successful parameterization. Run the
complete mojoeMD preparation stage and verify that:

1. The intended nonstandard residues were preserved and unwanted ones were
   removed or replaced explicitly.
2. The NCAA has every expected heavy atom and hydrogen with unique names.
3. Internal and adjacent peptide bonds are present.
4. Every residue matches exactly one force-field template after hydrogen
   addition.
5. The reported NCAA template charge matches the intended formal charge.
6. `prepared.pdb` and the deserialized `system.xml` contain the same number of
   atoms/particles.
7. The preparation log, source fragment, hydrogenated fragment, MOL2, atom-type
   map, XML files, configuration, commands, tool versions, and checksums are
   retained for reproducibility.

Parameters produced with approximate type mappings are appropriate for
pipeline development and compatibility testing. Novel chemistry should receive
independent charge, bonded-parameter, protonation, and conformational validation
before production or publication-quality simulations.

## Trajectory Analysis

Run:

```bash
python analysis/analyse_MD.py -c analysis/analyse.json
```

The analysis pipeline can:

- Strip water and ions.
- Repair periodic boundary artifacts for multichain complexes.
- Pull small molecules back near a protein anchor selection.
- Center and align the trajectory.
- Compute RMSD.
- Compute RMSF and write B-factor-mapped PDBs.
- Compute global and residue-wise SASA.
- Compute dihedral timeseries and summaries.
- Run PCA.
- Run hierarchical clustering and write representative structures.

Typical outputs include:

```text
analysis_out/rmsd.csv
analysis_out/processed_topology.pdb
analysis_out/processed.dcd
analysis_out/rmsf_atoms.csv
analysis_out/rmsf_residues.csv
analysis_out/bfactor_rmsf.pdb
analysis_out/sasa_global.csv
analysis_out/pca_projection.csv
analysis_out/clusters.csv
analysis_out/cluster_representatives/
```

## Contact Analysis

Run:

```bash
python analysis/contact_analysis.py -c analysis/contact.json
```

`contact_analysis.py` can:

- Run `getcontacts` dynamically or statically.
- Parse existing getcontacts TSV files.
- Filter contacts by type and frequency.
- Exclude `wb2` water-bridge contacts.
- Map contact scores onto PDB B-factors.
- Write PyMOL scripts for contact visualization.
- Optionally compute simple nonbonded interaction-energy estimates from `system.xml` or Amber `prmtop`.
- Optionally write residue energy plots and an interactive flareplot.

Key outputs:

```text
contact_analysis_out/getcontacts_A_B.tsv
contact_analysis_out/contacts_filtered.csv
contact_analysis_out/contacts_mapped_bfactor.pdb
contact_analysis_out/contacts.pml
contact_analysis_out/interaction_energies.csv
contact_analysis_out/energies_mapped_bfactor.pdb
contact_analysis_out/energy_contacts.pml
contact_analysis_out/interaction_summary_residue_energies.png
contact_analysis_out/interaction_summary_flareplot.html
```

The energy calculation is a lightweight Coulomb + Lennard-Jones screen. It is useful for diagnostics and ranking, but it is not a rigorous binding free-energy method.

## Important Notes

- Input PDB quality matters. Missing atoms, ambiguous chain IDs, unsupported glycan linkages, or unmatched NCAA residues will stop preparation.
- Metal sites need special care. Metal-aware protonation prevents hydrogens from being added to coordinating histidine nitrogens or cysteine sulfurs, but monatomic metal ions are still nonbonded particles unless `metal_coordination_restraints` or a dedicated metal-site model is used.
- `mojoeMD.py` currently uses Amber14 protein and TIP3P-FB force-field XML files by default.
- `water_model` is passed to OpenMM solvent building; keep it compatible with the loaded water XML.
- Production reports DCD frames every 50 ps.
- Production checkpointing is configured at a long interval; final checkpoint is always written at the end of a successful run.
- GPU platforms use mixed precision, and CUDA sets deterministic forces by default.
- `metadynamics`, `bias_exchange`, and `auto_cdr_bias_exchange` are mutually exclusive; enabling
  more than one of them fails validation before any stage runs.
- Bias-exchange and auto-CDR bias-exchange run every walker sequentially in one process (no
  MPI/multi-process support), so wall-clock time scales with the number of walkers.

## Development Check

Basic syntax check:

```bash
python -m py_compile mojoeMD.py analysis/analyse_MD.py analysis/contact_analysis.py ncaa/generate_ncaa_ffxml_from_amber_mol2.py
```

Most functional checks require installed MD dependencies and realistic molecular input files.

## License

MIT License

Copyright (c) 2026 mojoeMD authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

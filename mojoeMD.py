#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pdbfixer

from openmm import (
    CustomBondForce,
    CustomExternalForce,
    CustomTorsionForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    State,
    XmlSerializer,
)
from openmm.app import (
    PME,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
    Topology,
    CheckpointReporter,
)
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    Quantity,
    amu,
    bar,
    femtoseconds,
    kelvin,
    kilojoules_per_mole,
    molar,
    nanometer,
    nanoseconds,
    picosecond,
    picoseconds,
)
from tqdm import tqdm

try:
    from openmmtools.utils import get_fastest_platform
except ImportError:
    def get_fastest_platform() -> Platform:
        from openmm import System, VerletIntegrator, Context
        from openmm.unit import picoseconds

        preferred = ["CUDA", "OpenCL", "CPU", "Reference"]
        available = {
            Platform.getPlatform(i).getName(): Platform.getPlatform(i)
            for i in range(Platform.getNumPlatforms())
        }

        for name in preferred:
            platform = available.get(name)
            if platform is None:
                continue

            # A platform can be compiled in but not actually usable (e.g. no working OpenCL
            # driver/device), so verify it by creating a throwaway Context rather than trusting
            # Platform.getNumPlatforms() alone.
            try:
                test_system = System()
                test_system.addParticle(1.0)
                Context(test_system, VerletIntegrator(1.0 * picoseconds), platform)
            except Exception:
                continue

            return platform

        raise RuntimeError("No usable OpenMM platform is available.")

def _find_spec_safe(name: str):
    # importlib.util.find_spec raises ModuleNotFoundError (instead of returning None) for a
    # dotted name whose parent package isn't installed at all, e.g. "openff.toolkit" when
    # "openff" doesn't exist.
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


_has_openff = (
    _find_spec_safe("openff.toolkit") is not None
    and _find_spec_safe("rdkit") is not None
    and _find_spec_safe("openmmforcefields") is not None
)


def setup_logging(log_file: str | Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def resolve_config_path(value: str | None, config_dir: Path | None) -> str | None:
    """Resolve a config path relative to its JSON file when that file exists."""
    if value is None or config_dir is None:
        return value

    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)

    config_relative = config_dir / path
    if config_relative.exists():
        return str(config_relative)

    return value


def resolve_config_paths(args: argparse.Namespace, config_dir: Path | None) -> None:
    if config_dir is None:
        return

    args.receptor = resolve_config_path(args.receptor, config_dir)
    args.ligand = resolve_config_path(args.ligand, config_dir)

    if args.ncaa_ffxml:
        args.ncaa_ffxml = [
            resolve_config_path(ffxml, config_dir)
            for ffxml in args.ncaa_ffxml
        ]

    args.glycam_hydrogens = resolve_config_path(args.glycam_hydrogens, config_dir)
    args.water_ff = resolve_config_path(args.water_ff, config_dir)
    args.lipid_ff = resolve_config_path(args.lipid_ff, config_dir)


def validate_existing_file(path: str | None, label: str) -> bool:
    if path is None:
        return True

    if Path(path).expanduser().exists():
        return True

    logging.error(f"{label} file does not exist: {path}")
    return False


def validate_openmm_app_data_file(path: str | None, label: str) -> bool:
    if path is None:
        return True

    if Path(path).expanduser().exists():
        return True

    app_spec = importlib.util.find_spec("openmm.app")
    if app_spec and app_spec.origin:
        data_path = Path(app_spec.origin).parent / "data" / path
        if data_path.exists():
            return True

    logging.error(f"{label} file does not exist: {path}")
    return False


def parse_int_residue_id(residue_id: str) -> int | None:
    try:
        return int(residue_id)
    except (TypeError, ValueError):
        return None


KYTE_DOOLITTLE_HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}


def infer_membrane_center_z_from_hydrophobic_slab(
    modeller: Modeller,
    thickness_nm: float,
    window_size: int = 21,
) -> float | None:
    positions_nm = modeller.positions.value_in_unit(nanometer)
    residues_by_chain: dict[str, list[tuple[int, str, float]]] = defaultdict(list)

    for atom in modeller.topology.atoms():
        residue = atom.residue
        residue_name = residue.name.upper()
        if residue_name not in KYTE_DOOLITTLE_HYDROPATHY:
            continue
        if atom.name != "CA":
            continue

        residue_id = parse_int_residue_id(residue.id)
        if residue_id is None:
            continue

        residues_by_chain[residue.chain.id].append((
            residue_id,
            residue_name,
            positions_nm[atom.index][2],
        ))

    if not residues_by_chain:
        return None

    best = None
    min_z_span_nm = min(1.4, thickness_nm * 0.5)

    for chain_id, residues in residues_by_chain.items():
        residues = sorted(residues)
        if len(residues) < window_size:
            continue

        for start in range(len(residues) - window_size + 1):
            window = residues[start:start + window_size]
            first_resid = window[0][0]
            last_resid = window[-1][0]

            if last_resid - first_resid > window_size + 5:
                continue

            z_values = [item[2] for item in window]
            z_span = max(z_values) - min(z_values)
            if z_span < min_z_span_nm:
                continue

            hydropathy = sum(KYTE_DOOLITTLE_HYDROPATHY[item[1]] for item in window) / window_size
            if hydropathy < 1.0:
                continue

            score = hydropathy * z_span
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "chain_id": chain_id,
                    "first_resid": first_resid,
                    "last_resid": last_resid,
                    "hydropathy": hydropathy,
                    "z_span": z_span,
                    "center_z": 0.5 * (min(z_values) + max(z_values)),
                }

    if best is None:
        return None

    logging.info(
        "Auto-selected membrane center z from hydrophobic TM-window heuristic: "
        f"{best['center_z']:.2f} nm "
        f"(chain={best['chain_id']}, residues={best['first_resid']}-{best['last_resid']}, "
        f"hydropathy={best['hydropathy']:.2f}, z_span={best['z_span']:.2f} nm)"
    )
    return best["center_z"]


def choose_membrane_center_z(modeller: Modeller, args: argparse.Namespace) -> float:
    if args.membrane_center_z is not None:
        return args.membrane_center_z

    selected_positions = []
    positions_nm = modeller.positions.value_in_unit(nanometer)
    residue_range = args.membrane_center_residue_range
    chain_ids = set(args.membrane_center_chain_ids or [])

    if residue_range or chain_ids:
        for atom in modeller.topology.atoms():
            residue = atom.residue

            if chain_ids and residue.chain.id not in chain_ids:
                continue

            if residue_range:
                residue_id = parse_int_residue_id(residue.id)
                if residue_id is None:
                    continue
                if residue_id < residue_range[0] or residue_id > residue_range[1]:
                    continue

            selected_positions.append(positions_nm[atom.index])

        if not selected_positions:
            raise ValueError(
                "No atoms matched the membrane-centering selection. "
                "Check --membrane_center_chain_ids and --membrane_center_residue_range."
            )

        z_values = [position[2] for position in selected_positions]
        membrane_center_z = 0.5 * (min(z_values) + max(z_values))
        selection = []
        if chain_ids:
            selection.append("chains=" + ",".join(sorted(chain_ids)))
        if residue_range:
            selection.append(f"residues={residue_range[0]}-{residue_range[1]}")
        logging.info(
            "Auto-selected membrane center z from "
            f"{' '.join(selection)}: {membrane_center_z:.2f} nm"
        )
        return membrane_center_z

    membrane_center_z = infer_membrane_center_z_from_hydrophobic_slab(
        modeller,
        args.membrane_auto_thickness,
    )
    if membrane_center_z is not None:
        return membrane_center_z

    z_values = [position[2] for position in positions_nm]
    membrane_center_z = 0.5 * (min(z_values) + max(z_values))
    logging.warning(
        "Could not infer a hydrophobic membrane slab; falling back to all prepared atoms "
        f"for membrane center z: {membrane_center_z:.2f} nm"
    )

    return membrane_center_z


def validate_input_files(args: argparse.Namespace) -> None:
    ok = validate_existing_file(args.receptor, "Receptor")

    if args.ligand:
        ok = validate_existing_file(args.ligand, "Ligand") and ok

    if args.ncaa_ffxml:
        for ffxml in args.ncaa_ffxml:
            ok = validate_existing_file(ffxml, "NCAA ffxml") and ok

    if args.glycoprotein:
        ok = validate_openmm_app_data_file(args.glycam_hydrogens, "GLYCAM hydrogen definitions") and ok

    ok = validate_openmm_app_data_file(args.water_ff, "Water force field") and ok

    if args.membrane:
        ok = validate_openmm_app_data_file(args.lipid_ff, "Lipid force field") and ok

    if args.metadynamics and args.metadynamics.get("enabled"):
        ok = validate_metadynamics_config(args.metadynamics) and ok

    if args.bias_exchange and args.bias_exchange.get("enabled"):
        if args.metadynamics and args.metadynamics.get("enabled"):
            logging.error("metadynamics.enabled and bias_exchange.enabled cannot both be true.")
            ok = False

        ok = validate_bias_exchange_config(args.bias_exchange) and ok

    if args.auto_cdr_bias_exchange and args.auto_cdr_bias_exchange.get("enabled"):
        if args.metadynamics and args.metadynamics.get("enabled"):
            logging.error("metadynamics.enabled and auto_cdr_bias_exchange.enabled cannot both be true.")
            ok = False

        if args.bias_exchange and args.bias_exchange.get("enabled"):
            logging.error("bias_exchange.enabled and auto_cdr_bias_exchange.enabled cannot both be true.")
            ok = False

        ok = validate_auto_cdr_bias_exchange_config(args.auto_cdr_bias_exchange) and ok

        if importlib.util.find_spec("anarcii") is None:
            logging.error(
                "auto_cdr_bias_exchange.enabled is true but the 'anarcii' package is not installed. "
                "Install it with: pip install anarcii"
            )
            ok = False

    if not ok:
        sys.exit(1)


def validate_cv_list(cvs: list[dict] | None, context_label: str) -> bool:
    if not cvs:
        logging.error(f"{context_label}: no 'cvs' were provided.")
        return False

    ok = True

    for i, cv in enumerate(cvs):
        if cv.get("type") not in ("phi", "psi"):
            logging.error(f"{context_label}.cvs[{i}]: 'type' must be 'phi' or 'psi', got {cv.get('type')!r}.")
            ok = False

        if cv.get("residue") is None:
            logging.error(f"{context_label}.cvs[{i}]: 'residue' is required.")
            ok = False

    return ok


def validate_metadynamics_config(metad_config: dict) -> bool:
    cvs = metad_config.get("cvs")

    if cvs and len(cvs) > 3:
        logging.error(f"metadynamics supports at most 3 collective variables, got {len(cvs)}.")
        return False

    return validate_cv_list(cvs, "metadynamics")


def validate_bias_exchange_config(bias_exchange_config: dict) -> bool:
    walkers = bias_exchange_config.get("walkers")

    if not walkers or len(walkers) < 2:
        logging.error("bias_exchange.enabled is true but fewer than 2 'walkers' were provided.")
        return False

    ok = True

    for i, walker_cfg in enumerate(walkers):
        # Walkers auto-chunk into groups of <=3 CVs (build_metadynamics_groups), so unlike
        # top-level metadynamics there is no hard cap on a walker's total CV count here.
        if not validate_cv_list(walker_cfg.get("cvs"), f"bias_exchange.walkers[{i}]"):
            ok = False

    return ok


def validate_auto_cdr_bias_exchange_config(auto_cfg: dict) -> bool:
    loops = auto_cfg.get("loops")

    if loops is None:
        return True

    invalid = set(loops) - VALID_CDR_LOOP_NAMES

    if invalid:
        logging.error(
            f"auto_cdr_bias_exchange.loops contains invalid loop name(s) {sorted(invalid)}; "
            f"valid names are {sorted(VALID_CDR_LOOP_NAMES)}."
        )
        return False

    return True


def select_platform(platform_name: str | None = None) -> Platform:
    if platform_name is None or platform_name.lower() == "fastest":
        platform = get_fastest_platform()
        logging.info(f"Using fastest available platform: {platform.getName()}")
    else:
        platform = Platform.getPlatformByName(platform_name)
        logging.info(f"Using specified platform: {platform.getName()}")

    if platform.getName() in ["CUDA", "OpenCL"]:
        platform.setPropertyDefaultValue("Precision", "mixed")
        if platform.getName() == "CUDA":
            platform.setPropertyDefaultValue("DeterministicForces", "true")

    return platform


class LoggingReporter:
    def __init__(self, report_interval: int = 10000) -> None:
        self._reportInterval = report_interval
        self._last_reported_time = None
        self._last_reported_ns = 0.0

    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, False)

    def report(self, simulation: Simulation, state: State) -> None:
        current_time = time.perf_counter()
        dt = simulation.integrator.getStepSize()
        current_nanoseconds = simulation.currentStep * dt.value_in_unit(nanoseconds)

        if self._last_reported_time is not None:
            elapsed_wall_time = current_time - self._last_reported_time
            elapsed_ns = current_nanoseconds - self._last_reported_ns
            compute_time_s_ns = elapsed_wall_time / elapsed_ns if elapsed_ns > 0 else 0.0
            logging.info(
                f"Step {simulation.currentStep}, ns: {current_nanoseconds:.3f}, "
                f"compute_time: {compute_time_s_ns:.2f} s/ns"
            )
        else:
            logging.info(f"Step {simulation.currentStep}, ns: {current_nanoseconds:.3f}, compute_time: initializing...")

        self._last_reported_time = current_time
        self._last_reported_ns = current_nanoseconds


def strip_amber_caps(modeller: Modeller) -> None:
    cap_residues = [
        residue
        for residue in modeller.topology.residues()
        if residue.name.upper() in {"ACE", "NME"}
    ]

    if not cap_residues:
        logging.info("Terminal cap stripping requested, but no ACE/NME residues were found.")
        return

    labels = [f"{residue.name} {residue.chain.id}:{residue.id}" for residue in cap_residues]
    modeller.delete(cap_residues)
    logging.info("Removed terminal caps: " + ", ".join(labels))


def prepare_protein(
    pdb_file: str | Path,
    ncaa: bool = False,
    ncaa_resnames: list[str] | None = None,
) -> pdbfixer.PDBFixer:
    logging.info(f"Preparing protein from {pdb_file}...")

    fixer = pdbfixer.PDBFixer(str(pdb_file))

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()

    ncaa_resnames_set = set(ncaa_resnames or [])

    if ncaa:
        logging.info("NCAA mode enabled.")

        if ncaa_resnames_set:
            logging.info(f"Preserving NCAA residue names: {', '.join(sorted(ncaa_resnames_set))}")
        else:
            logging.warning(
                "NCAA mode enabled but no --ncaa_resnames were provided. "
                "All nonstandard residues will be preserved."
            )

        kept = []
        replace = []

        for residue, suggested_replacement in fixer.nonstandardResidues:
            if not ncaa_resnames_set or residue.name in ncaa_resnames_set:
                kept.append(residue.name)
            else:
                replace.append((residue, suggested_replacement))

        fixer.nonstandardResidues = replace

        if replace:
            logging.info(
                "Replacing nonstandard residues not marked as NCAA: "
                + ", ".join(sorted({res.name for res, _ in replace}))
            )
            fixer.replaceNonstandardResidues()

        if kept:
            logging.info("Kept NCAA/nonstandard residues unchanged: " + ", ".join(sorted(set(kept))))

    else:
        fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    logging.info("Skipping PDBFixer hydrogen addition; hydrogens will be added with Modeller/ForceField.")
    logging.info("Protein preparation complete.")
    return fixer


def patch_ncaa_internal_bonds_from_ffxml(
    topology: Topology,
    ncaa_resnames: list[str],
    ncaa_ffxml_files: list[str],
) -> None:
    """Patch missing intra-residue NCAA bonds using Bond records from ffxml.

    This is needed because OpenMM/PDBFixer often does not infer internal bonds
    for unknown residue names such as ALM.
    """
    ncaa_set = set(ncaa_resnames or [])

    if not ncaa_set:
        logging.warning("NCAA mode enabled but no residue names given; skipping NCAA internal-bond patching.")
        return

    if not ncaa_ffxml_files:
        logging.warning("No NCAA ffxml files provided; skipping NCAA internal-bond patching.")
        return

    residue_bonds: dict[str, list[tuple[str, str]]] = {}

    for ffxml in ncaa_ffxml_files:
        ffxml_path = Path(ffxml)

        if not ffxml_path.exists():
            logging.error(f"NCAA ffxml file does not exist: {ffxml_path}")
            sys.exit(1)

        tree = ET.parse(ffxml_path)
        root = tree.getroot()

        for residue_elem in root.findall(".//Residue"):
            resname = residue_elem.attrib.get("name")

            if resname not in ncaa_set:
                continue

            bonds = []

            for bond_elem in residue_elem.findall("Bond"):
                a1 = bond_elem.attrib.get("atomName1")
                a2 = bond_elem.attrib.get("atomName2")

                if a1 and a2:
                    bonds.append((a1, a2))

            residue_bonds[resname] = bonds
            logging.info(f"Read {len(bonds)} internal bond definition(s) for NCAA residue {resname} from {ffxml_path}")

    existing_bonds = {frozenset((bond[0].index, bond[1].index)) for bond in topology.bonds()}
    added = 0

    for residue in topology.residues():
        if residue.name not in ncaa_set:
            continue

        atoms = {atom.name: atom for atom in residue.atoms()}

        if residue.name not in residue_bonds:
            logging.warning(f"No internal bond definitions found for NCAA residue {residue.name}")
            continue

        for atom1_name, atom2_name in residue_bonds[residue.name]:
            if atom1_name not in atoms or atom2_name not in atoms:
                continue

            atom1 = atoms[atom1_name]
            atom2 = atoms[atom2_name]
            bond_key = frozenset((atom1.index, atom2.index))

            if bond_key not in existing_bonds:
                topology.addBond(atom1, atom2)
                existing_bonds.add(bond_key)
                added += 1
                logging.info(
                    f"Added NCAA internal bond: "
                    f"{residue.name}{residue.id}:{atom1_name} -- {atom2_name}"
                )

    logging.info(f"Patched {added} NCAA internal bond(s) from ffxml.")


def patch_ncaa_peptide_bonds(topology: Topology, ncaa_resnames: list[str]) -> None:
    """Patch missing peptide bonds around NCAA residues.

    Adds:
      previous residue C -- NCAA N
      NCAA C -- next residue N

    when either residue in an adjacent pair is an NCAA.
    """
    ncaa_set = set(ncaa_resnames or [])

    if not ncaa_set:
        logging.warning("NCAA mode enabled but no residue names given; skipping peptide-bond patching.")
        return

    existing_bonds = {frozenset((bond[0].index, bond[1].index)) for bond in topology.bonds()}

    def atom_by_name(residue, atom_name: str):
        for atom in residue.atoms():
            if atom.name == atom_name:
                return atom
        return None

    added = 0

    for chain in topology.chains():
        residues = list(chain.residues())

        for prev_res, next_res in zip(residues[:-1], residues[1:]):
            if prev_res.name not in ncaa_set and next_res.name not in ncaa_set:
                continue

            c_atom = atom_by_name(prev_res, "C")
            n_atom = atom_by_name(next_res, "N")

            if c_atom is None or n_atom is None:
                logging.warning(
                    f"Could not patch peptide bond {prev_res.name}{prev_res.id} -> "
                    f"{next_res.name}{next_res.id}: missing C or N atom."
                )
                continue

            bond_key = frozenset((c_atom.index, n_atom.index))

            if bond_key not in existing_bonds:
                topology.addBond(c_atom, n_atom)
                existing_bonds.add(bond_key)
                added += 1
                logging.info(
                    f"Added peptide bond: {prev_res.name}{prev_res.id}:C -- "
                    f"{next_res.name}{next_res.id}:N"
                )

    logging.info(f"Patched {added} NCAA-adjacent peptide bond(s).")


def load_ncaa_hydrogen_definitions(ncaa_ffxml_files: list[str]) -> None:
    """Load optional hydrogen-definition files next to NCAA ffxml files.

    Example:
      ncaa/ALM.xml -> ncaa/ALM_hydrogens.xml
    """
    for ffxml in ncaa_ffxml_files:
        ffxml_path = Path(ffxml)
        hxml_path = ffxml_path.with_name(ffxml_path.stem + "_hydrogens.xml")

        if hxml_path.exists():
            logging.info(f"Loading NCAA hydrogen definitions: {hxml_path}")
            Modeller.loadHydrogenDefinitions(str(hxml_path))
        else:
            logging.warning(f"No NCAA hydrogen definition file found: {hxml_path}")


def pdb_safe_residue_name(name: str) -> str:
    """Return a PDB-safe 3-character residue name."""
    cleaned = "".join(char for char in name.upper() if char.isalnum())
    if not cleaned:
        return "LIG"
    return cleaned[:3]


def prepare_ligand(lig_sdf_file: str | Path, residue_name: str | None = None):
    if not _has_openff:
        raise ImportError("OpenFF Toolkit and RDKit are required for ligand preparation.")

    from openff.toolkit import Molecule
    from rdkit.Chem import SDMolSupplier

    logging.info(f"Loading ligand from {lig_sdf_file}...")
    rdkit_mol = SDMolSupplier(str(lig_sdf_file))[0]

    if rdkit_mol is None:
        raise ValueError(f"Could not read ligand from SDF file: {lig_sdf_file}")

    ligand = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
    ligand.name = pdb_safe_residue_name(residue_name or Path(lig_sdf_file).stem)
    logging.info(f"Using ligand residue name: {ligand.name}")
    return ligand


def parametrize_ligand(ligand, forcefield: ForceField, lig_ff: str = "espaloma") -> tuple[Topology, Quantity]:
    from openff.toolkit import Topology as offTopology
    from openff.units.openmm import to_openmm as offquantity_to_openmm
    from openmmforcefields.generators import EspalomaTemplateGenerator, SMIRNOFFTemplateGenerator

    logging.info(f"Parameterizing ligand with {lig_ff}...")

    if lig_ff == "espaloma":
        template_generator = EspalomaTemplateGenerator(molecules=ligand, forcefield="espaloma-0.3.1")
    elif lig_ff == "smirnoff":
        template_generator = SMIRNOFFTemplateGenerator(molecules=ligand, forcefield="openff-2.1.0")
    else:
        raise ValueError("Unsupported ligand force field. Choose 'espaloma' or 'smirnoff'.")

    forcefield.registerTemplateGenerator(template_generator.generator)

    ligand_off_topology = offTopology.from_molecules(molecules=[ligand])
    ligand_omm_topology = ligand_off_topology.to_openmm()
    for residue in ligand_omm_topology.residues():
        residue.name = pdb_safe_residue_name(ligand.name)

    if ligand.conformers is None or len(ligand.conformers) == 0:
        raise ValueError("Ligand must have at least one conformer for parameterization.")

    ligand_positions = offquantity_to_openmm(ligand.conformers[0])

    logging.info("Ligand parameterization complete.")
    return ligand_omm_topology, ligand_positions


def check_unmatched_residues(
    forcefield: ForceField,
    topology: Topology,
    args: argparse.Namespace,
    ignored_residues: set | None = None,
) -> None:
    ignored_residues = ignored_residues or set()
    unmatched = [
        residue
        for residue in forcefield.getUnmatchedResidues(topology)
        if residue not in ignored_residues
    ]

    if not unmatched:
        logging.info("All residues matched by the force field.")
        return

    unmatched_names = sorted({res.name for res in unmatched})
    logging.warning("Unmatched residues before createSystem/addSolvent: " + ", ".join(unmatched_names))

    if args.ncaa:
        ncaa_set = set(args.ncaa_resnames or [])
        missing_ncaa = [res for res in unmatched if not ncaa_set or res.name in ncaa_set]

        if missing_ncaa:
            logging.error(
                "Some NCAA residues still have no matching force-field template: "
                + ", ".join(sorted({res.name for res in missing_ncaa}))
            )
            logging.error(
                "For embedded NCAAs, provide or fix a residue-template ffxml via "
                "--ncaa_ffxml ncaa/ALM.xml"
            )
            sys.exit(1)

    logging.error(
        "Unmatched residues found. Use --ncaa with --ncaa_resnames and --ncaa_ffxml "
        "if these are intended noncanonical residues."
    )
    sys.exit(1)


METAL_ELEMENTS = {
    "LI", "NA", "K", "RB", "CS",
    "BE", "MG", "CA", "SR", "BA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "Y", "ZR", "MO", "RU", "RH", "PD", "AG", "CD",
    "W", "PT", "AU", "HG",
    "AL", "GA", "IN", "SN", "PB",
}
WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP3", "TIP3P"}
BULK_ION_RESNAMES = {"NA", "CL", "K", "RB", "CS", "LI", "F", "BR", "I"}
COORDINATING_DONOR_ELEMENTS = {"N", "O", "S"}


def atom_element_symbol(atom) -> str:
    if atom.element is not None and atom.element.symbol:
        return atom.element.symbol.upper()

    name = atom.name.strip()
    if name and name[0].isdigit():
        name = name[1:]

    two = name[:2].upper()
    if two in METAL_ELEMENTS or two in {"CL", "BR"}:
        return two

    return name[:1].upper()


def position_distance_nm(pos1, pos2) -> float:
    try:
        p1 = pos1.value_in_unit(nanometer)
    except AttributeError:
        p1 = pos1

    try:
        p2 = pos2.value_in_unit(nanometer)
    except AttributeError:
        p2 = pos2

    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(3)))


def is_metal_atom(atom) -> bool:
    return atom_element_symbol(atom) in METAL_ELEMENTS


def is_bulk_solvent_or_ion_residue(residue) -> bool:
    return residue.name.upper() in WATER_RESNAMES | BULK_ION_RESNAMES


def build_metal_aware_protonation_variants(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[str | None]:
    """Force residue protonation variants for common metal-coordinating side chains."""
    residues = list(topology.residues())
    variants: list[str | None] = [None] * len(residues)
    metal_atoms = [atom for atom in topology.atoms() if is_metal_atom(atom)]

    if not metal_atoms:
        logging.info("Metal-aware protonation: no metal atoms detected.")
        return variants

    cutoff_nm = cutoff_angstrom * 0.1
    logging.info(
        "Metal-aware protonation: checking histidine-metal contacts "
        f"within {cutoff_angstrom:.2f} A."
    )

    for residue in residues:
        residue_name = residue.name.upper()

        if residue_name in {"CYS", "CYM", "CYX"}:
            atoms = {atom.name: atom for atom in residue.atoms()}
            sg = atoms.get("SG")

            if sg is None:
                continue

            contacts = []

            for metal in metal_atoms:
                distance = position_distance_nm(positions[sg.index], positions[metal.index])
                if distance <= cutoff_nm:
                    contacts.append((metal, distance))

            if contacts:
                variants[residue.index] = "CYX"
                nearest = min(contacts, key=lambda item: item[1])
                logging.info(
                    "Metal-aware protonation: forcing "
                    f"{residue.name}{residue.id} chain {residue.chain.id} to CYX "
                    f"(SG coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
                )

            continue

        if residue_name not in {"HIS", "HID", "HIE", "HIP"}:
            continue

        atoms = {atom.name: atom for atom in residue.atoms()}
        nd1 = atoms.get("ND1")
        ne2 = atoms.get("NE2")

        if nd1 is None or ne2 is None:
            continue

        nd1_contacts = []
        ne2_contacts = []

        for metal in metal_atoms:
            nd1_distance = position_distance_nm(positions[nd1.index], positions[metal.index])
            ne2_distance = position_distance_nm(positions[ne2.index], positions[metal.index])

            if nd1_distance <= cutoff_nm:
                nd1_contacts.append((metal, nd1_distance))
            if ne2_distance <= cutoff_nm:
                ne2_contacts.append((metal, ne2_distance))

        if nd1_contacts and ne2_contacts:
            logging.warning(
                "Metal-aware protonation: both ND1 and NE2 are close to metal(s) for "
                f"{residue.name}{residue.id} chain {residue.chain.id}; leaving pH-based variant. "
                "This may need an imidazolate/custom template."
            )
            continue

        if nd1_contacts:
            variants[residue.index] = "HIE"
            nearest = min(nd1_contacts, key=lambda item: item[1])
            logging.info(
                "Metal-aware protonation: forcing "
                f"{residue.name}{residue.id} chain {residue.chain.id} to HIE "
                f"(ND1 coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
            )
        elif ne2_contacts:
            variants[residue.index] = "HID"
            nearest = min(ne2_contacts, key=lambda item: item[1])
            logging.info(
                "Metal-aware protonation: forcing "
                f"{residue.name}{residue.id} chain {residue.chain.id} to HID "
                f"(NE2 coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
            )

    return variants


def build_metal_aware_histidine_variants(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[str | None]:
    """Backward-compatible wrapper for the broader metal-aware protonation logic."""
    return build_metal_aware_protonation_variants(topology, positions, cutoff_angstrom)


def find_metal_coordination_bonds(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[tuple[int, int, float, str]]:
    cutoff_nm = cutoff_angstrom * 0.1
    atoms = list(topology.atoms())
    metal_atoms = [
        atom
        for atom in atoms
        if is_metal_atom(atom) and atom.residue.name.upper() not in BULK_ION_RESNAMES
    ]
    donor_atoms = [
        atom
        for atom in atoms
        if atom_element_symbol(atom) in COORDINATING_DONOR_ELEMENTS
        and not is_metal_atom(atom)
        and not is_bulk_solvent_or_ion_residue(atom.residue)
    ]

    bonds = []

    for metal in metal_atoms:
        for donor in donor_atoms:
            distance_nm = position_distance_nm(positions[metal.index], positions[donor.index])
            if distance_nm <= cutoff_nm:
                label = (
                    f"{metal.name}:{metal.residue.name}{metal.residue.id}:chain {metal.residue.chain.id} -- "
                    f"{donor.name}:{donor.residue.name}{donor.residue.id}:chain {donor.residue.chain.id}"
                )
                bonds.append((metal.index, donor.index, distance_nm, label))

    return bonds


def add_metal_coordination_restraints(
    system,
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
    force_constant: float,
) -> None:
    bonds = find_metal_coordination_bonds(topology, positions, cutoff_angstrom)

    if not bonds:
        logging.warning("Metal coordination restraints requested, but no metal-donor contacts were detected.")
        return

    restraint = CustomBondForce("0.5*k_metal_coordination*(r-r0)^2")
    restraint.addGlobalParameter(
        "k_metal_coordination",
        force_constant * kilojoules_per_mole / nanometer**2,
    )
    restraint.addPerBondParameter("r0")

    for metal_index, donor_index, distance_nm, label in bonds:
        restraint.addBond(metal_index, donor_index, [distance_nm])
        logging.info(
            "Metal coordination restraint: "
            f"{label}, r0={distance_nm * 10:.2f} A, k={force_constant:.1f} kJ/mol/nm^2"
        )

    system.addForce(restraint)
    logging.info(f"Added {len(bonds)} metal coordination restraint(s).")



# ============================================================
# GLYCOPROTEIN / GLYCAM HELPERS
# ============================================================

PDB_TO_GLYCAM_SUFFIXES = {
    "ABE": ("AE", "AF"),
    "ADA": ("OA", "OB"),
    "AHR": ("AA", "AB", "AD", "AU"),
    "ALL": ("NA", "NB"),
    "ALT": ("EA", "EB"),
    "ARA": ("AA", "AB", "AD", "AU"),
    "BAC": ("BC",),
    "BGC": ("GB",),
    "BMA": ("MB",),
    "FRU": ("CA", "CB", "CD", "CU"),
    "FUC": ("FA",),
    "FUL": ("FA",),
    "GAL": ("LA", "LB"),
    "GCS": ("YN", "YS"),
    "GCU": ("ZA", "ZB"),
    "GLA": ("LA", "LB"),
    "GLC": ("GA", "GB"),
    "GL0": ("KA", "KB"),
    "GUL": ("KA", "KB"),
    "GUP": ("KA", "KB"),
    "IDS": ("UA", "UB"),
    "LYX": ("DA", "DB", "DD", "DU"),
    "LXC": ("XA", "XB", "XD", "XU"),
    "MAL": ("GA",),
    "MAN": ("MA",),
    "NAG": ("YB",),
    "NDG": ("YB",),
    "NGA": ("VA", "VB"),
    "QUI": ("QA", "QB"),
    "PSI": ("PA", "PB", "PD", "PU"),
    "RAM": ("HA", "HB"),
    "RIB": ("RA", "RB", "RD", "RU"),
    "SGN": ("YB",),
    "SIA": ("SA", "SB"),
    "SOR": ("BA", "BB", "BD", "BU"),
    "TAG": ("JA", "JB", "JD", "JU"),
    "TAL": ("TA", "TB"),
    "TYV": ("TV", "Tv"),
    "XYP": ("XA", "XB", "XD", "XU"),
    "XYS": ("XA", "XB", "XD", "XU"),
    "XYL": ("XA", "XB", "XD", "XU"),
}

GLYCAN_RESNAMES = set(PDB_TO_GLYCAM_SUFFIXES) | {
    "API", "BEM", "COL", "DHA", "DIG", "GMH", "IDO", "KDN", "KDO", "LGU", "MAV", "MUR", "NEU",
    "OLI", "PAR",
}

PROTEIN_GLYCAN_LINKS = {
    ("ASN", "ND2", "NAG", "C1"),
    ("ASN", "ND2", "NDG", "C1"),
    ("ASN", "ND2", "BMA", "C1"),
}

GLYCAN_LINK_DISTANCE_CUTOFF = 1.90
GLYCAN_GLYCAN_DISTANCE_CUTOFF = 1.85
GLYCAN_RING_ATOM = "C1"

GLYCAN_ANOMERIC_ATOMS = {
    "SIA": "C2",
    "NEU": "C2",
    "KDN": "C2",
    "KDO": "C2",
}


def _glyco_guess_element(atom_name: str, element_field: str = "") -> str:
    if element_field.strip():
        return element_field.strip().upper()

    atom_name = atom_name.strip()
    if atom_name and atom_name[0].isdigit():
        atom_name = atom_name[1:]

    if len(atom_name) >= 2:
        first2 = atom_name[:2].upper()
        if first2 in {"CL", "BR", "NA", "MG", "FE", "ZN"}:
            return first2

    return atom_name[0].upper() if atom_name else ""


def _glyco_is_hydrogen_pdb_atom_line(line: str) -> bool:
    atom_name = line[12:16].strip()
    element = _glyco_guess_element(line[12:16], line[76:78] if len(line) >= 78 else "")
    return element == "H" or atom_name.startswith("H")


def _glyco_parse_pdb(pdb_file: str | Path) -> tuple[list[dict], list[dict]]:
    atoms: list[dict] = []
    links: list[dict] = []

    with open(pdb_file) as handle:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"}:
                atoms.append({
                    "record": record,
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip() or " ",
                    "resid": int(line[22:26]),
                    "icode": line[26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": _glyco_guess_element(line[12:16], line[76:78] if len(line) >= 78 else ""),
                })

            elif record == "LINK":
                try:
                    links.append({
                        "atom1": line[12:16].strip(),
                        "resname1": line[17:20].strip(),
                        "chain1": line[21].strip() or " ",
                        "resid1": int(line[22:26]),
                        "icode1": line[26].strip(),
                        "atom2": line[42:46].strip(),
                        "resname2": line[47:50].strip(),
                        "chain2": line[51].strip() or " ",
                        "resid2": int(line[52:56]),
                        "icode2": line[56].strip(),
                    })
                except Exception:
                    continue

    return atoms, links


def _glyco_parse_link_line(line: str) -> dict | None:
    try:
        return {
            "atom1": line[12:16].strip(),
            "resname1": line[17:20].strip(),
            "chain1": line[21].strip() or " ",
            "resid1": int(line[22:26]),
            "icode1": line[26].strip(),
            "atom2": line[42:46].strip(),
            "resname2": line[47:50].strip(),
            "chain2": line[51].strip() or " ",
            "resid2": int(line[52:56]),
            "icode2": line[56].strip(),
        }
    except Exception:
        return None


def _glyco_residue_key(atom: dict) -> tuple[str, int, str, str]:
    return atom["chain"], atom["resid"], atom["icode"], atom["resname"]


def _glyco_residue_label(key: tuple[str, int, str, str]) -> str:
    chain, resid, icode, resname = key
    return f"{resname}:{chain}:{resid}{icode}"


def _glyco_parse_residue_label(label: str) -> tuple[str, str, str]:
    return tuple(label.split(":", 2))  # type: ignore[return-value]


def _glyco_group_residues(atoms: list[dict]) -> dict[tuple[str, int, str, str], list[dict]]:
    residues: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for atom in atoms:
        residues[_glyco_residue_key(atom)].append(atom)
    return residues


def _glyco_get_atom(residue_atoms: list[dict], atom_name: str) -> dict | None:
    for atom in residue_atoms:
        if atom["name"] == atom_name:
            return atom
    return None


def _glyco_distance(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _glyco_anomeric_atom(resname: str) -> str:
    return GLYCAN_ANOMERIC_ATOMS.get(resname, GLYCAN_RING_ATOM)


def _glyco_is_glycosidic_atom_pair(atom1: dict, atom2: dict) -> bool:
    """Restrict inferred glycan links to anomeric-carbon/oxygen glycosidic bonds."""
    if atom1["name"] == _glyco_anomeric_atom(atom1["resname"]) and atom2["element"] == "O" and atom2["name"].startswith("O"):
        return True
    if atom2["name"] == _glyco_anomeric_atom(atom2["resname"]) and atom1["element"] == "O" and atom1["name"].startswith("O"):
        return True
    return False


def _glyco_is_supported_protein_glycan_pair(key1: tuple, atom1: str, key2: tuple, atom2: str) -> bool:
    resname1 = key1[3]
    resname2 = key2[3]

    for protein_res, protein_atom, glycan_res, glycan_atom in PROTEIN_GLYCAN_LINKS:
        if resname1 == protein_res and atom1 == protein_atom and resname2 == glycan_res and atom2 == glycan_atom:
            return True
        if resname2 == protein_res and atom2 == protein_atom and resname1 == glycan_res and atom1 == glycan_atom:
            return True

    return False


def _glyco_is_supported_link_record(key1: tuple, atom1: dict | None, key2: tuple, atom2: dict | None) -> bool:
    if atom1 is None or atom2 is None:
        return False

    key1_is_glycan = key1[3] in GLYCAN_RESNAMES
    key2_is_glycan = key2[3] in GLYCAN_RESNAMES

    if key1_is_glycan and key2_is_glycan:
        return _glyco_is_glycosidic_atom_pair(atom1, atom2)

    if key1_is_glycan != key2_is_glycan:
        return _glyco_is_supported_protein_glycan_pair(key1, atom1["name"], key2, atom2["name"])

    return False


def _glyco_detect_links_from_records(residues: dict, pdb_links: list[dict]) -> list[dict]:
    links = []

    for link in pdb_links:
        key1 = (link["chain1"], link["resid1"], link["icode1"], link["resname1"])
        key2 = (link["chain2"], link["resid2"], link["icode2"], link["resname2"])

        if key1 not in residues or key2 not in residues:
            continue
        if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
            continue

        atom1 = _glyco_get_atom(residues[key1], link["atom1"])
        atom2 = _glyco_get_atom(residues[key2], link["atom2"])
        if not _glyco_is_supported_link_record(key1, atom1, key2, atom2):
            logging.info(
                "Ignoring unsupported glycan LINK record: "
                f"{_glyco_residue_label(key1)}:{link['atom1']} -- "
                f"{_glyco_residue_label(key2)}:{link['atom2']}"
            )
            continue
        dist = round(_glyco_distance(atom1, atom2), 3) if atom1 and atom2 else None

        links.append({
            "residue1": _glyco_residue_label(key1),
            "atom1": link["atom1"],
            "residue2": _glyco_residue_label(key2),
            "atom2": link["atom2"],
            "distance": dist,
            "source": "LINK",
        })

    return links


def write_pdb_with_supported_glycan_links_only(pdb_file: str | Path, out_pdb: Path) -> Path:
    atoms, _ = _glyco_parse_pdb(pdb_file)
    heavy_atoms = [atom for atom in atoms if atom["element"] != "H" and not atom["name"].startswith("H")]
    residues = _glyco_group_residues(heavy_atoms)
    atoms_by_serial = {atom["serial"]: atom for atom in heavy_atoms}
    skipped_links = 0
    skipped_conects = 0
    skipped_hydrogens = 0

    def keep_conect_pair(src_serial: int, target_serial: int) -> bool:
        nonlocal skipped_conects

        atom1 = atoms_by_serial.get(src_serial)
        atom2 = atoms_by_serial.get(target_serial)

        if atom1 is None or atom2 is None:
            skipped_conects += 1
            return False

        key1 = _glyco_residue_key(atom1)
        key2 = _glyco_residue_key(atom2)

        if key1 == key2:
            return True

        if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
            return True

        if _glyco_is_supported_link_record(key1, atom1, key2, atom2):
            return True

        skipped_conects += 1
        logging.info(
            "Removing unsupported glycan CONECT before PDBFixer: "
            f"{_glyco_residue_label(key1)}:{atom1['name']} -- "
            f"{_glyco_residue_label(key2)}:{atom2['name']}"
        )
        return False

    with open(pdb_file) as handle, open(out_pdb, "w") as out:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"} and _glyco_is_hydrogen_pdb_atom_line(line):
                skipped_hydrogens += 1
                continue

            if record == "CONECT":
                try:
                    src_serial = int(line[6:11])
                    target_serials = [
                        int(line[i:i + 5])
                        for i in range(11, len(line), 5)
                        if line[i:i + 5].strip()
                    ]
                except ValueError:
                    out.write(line)
                    continue

                kept_targets = [
                    target_serial
                    for target_serial in target_serials
                    if keep_conect_pair(src_serial, target_serial)
                ]

                if kept_targets:
                    out.write(f"CONECT{src_serial:5d}" + "".join(f"{serial:5d}" for serial in kept_targets) + "\n")
                continue

            if record != "LINK":
                out.write(line)
                continue

            link = _glyco_parse_link_line(line)
            if link is None:
                out.write(line)
                continue

            key1 = (link["chain1"], link["resid1"], link["icode1"], link["resname1"])
            key2 = (link["chain2"], link["resid2"], link["icode2"], link["resname2"])

            if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
                out.write(line)
                continue

            atom1 = _glyco_get_atom(residues.get(key1, []), link["atom1"])
            atom2 = _glyco_get_atom(residues.get(key2, []), link["atom2"])
            if _glyco_is_supported_link_record(key1, atom1, key2, atom2):
                out.write(line)
            else:
                skipped_links += 1
                logging.info(
                    "Removing unsupported glycan LINK before PDBFixer: "
                    f"{_glyco_residue_label(key1)}:{link['atom1']} -- "
                    f"{_glyco_residue_label(key2)}:{link['atom2']}"
                )

    logging.info(
        "Wrote PDBFixer input with "
        f"{skipped_links} unsupported glycan LINK record(s), "
        f"{skipped_conects} unsupported/invalid CONECT edge(s), and "
        f"{skipped_hydrogens} hydrogen atom record(s) removed: {out_pdb}"
    )
    return out_pdb


def _glyco_detect_protein_glycan_links(residues: dict) -> list[dict]:
    found = []
    residue_keys = list(residues.keys())

    for key1 in residue_keys:
        for key2 in residue_keys:
            if key1 == key2:
                continue

            resname1 = key1[3]
            resname2 = key2[3]

            for protein_res, protein_atom, glycan_res, glycan_atom in PROTEIN_GLYCAN_LINKS:
                if resname1 != protein_res or resname2 != glycan_res:
                    continue

                atom1 = _glyco_get_atom(residues[key1], protein_atom)
                atom2 = _glyco_get_atom(residues[key2], glycan_atom)
                if atom1 is None or atom2 is None:
                    continue

                d = _glyco_distance(atom1, atom2)
                if d <= GLYCAN_LINK_DISTANCE_CUTOFF:
                    found.append({
                        "residue1": _glyco_residue_label(key1),
                        "atom1": protein_atom,
                        "residue2": _glyco_residue_label(key2),
                        "atom2": glycan_atom,
                        "distance": round(d, 3),
                        "source": "distance",
                    })

    return found


def _glyco_detect_glycan_glycan_links(residues: dict) -> list[dict]:
    found = []
    glycan_keys = [key for key in residues if key[3] in GLYCAN_RESNAMES]

    for i, key1 in enumerate(glycan_keys):
        for key2 in glycan_keys[i + 1:]:
            best = None
            best_d = 999.0

            for atom1 in residues[key1]:
                for atom2 in residues[key2]:
                    if atom1["element"] == "H" or atom2["element"] == "H":
                        continue
                    if not _glyco_is_glycosidic_atom_pair(atom1, atom2):
                        continue
                    d = _glyco_distance(atom1, atom2)
                    if d < best_d:
                        best_d = d
                        best = atom1, atom2

            if best is None or best_d > GLYCAN_GLYCAN_DISTANCE_CUTOFF:
                continue

            atom1, atom2 = best
            found.append({
                "residue1": _glyco_residue_label(key1),
                "atom1": atom1["name"],
                "residue2": _glyco_residue_label(key2),
                "atom2": atom2["name"],
                "distance": round(best_d, 3),
                "source": "distance",
            })

    return found


def _glyco_unique_links(all_links: list[dict]) -> list[dict]:
    unique = []
    seen = set()

    for link in all_links:
        key = tuple(sorted([(link["residue1"], link["atom1"]), (link["residue2"], link["atom2"])]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(link)

    return unique


def _glyco_build_graph(links: list[dict]) -> tuple[dict[str, list[str]], dict[frozenset, dict]]:
    graph: dict[str, list[str]] = defaultdict(list)
    edge_data: dict[frozenset, dict] = {}

    for link in links:
        r1, r2 = link["residue1"], link["residue2"]
        graph[r1].append(r2)
        graph[r2].append(r1)
        edge_data[frozenset((r1, r2))] = link

    return graph, edge_data


def _glyco_connected_components(graph: dict[str, list[str]]) -> list[list[str]]:
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue
        queue = deque([node])
        component = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(sorted(component))

    return components


def _glyco_is_glycan_label(label: str) -> bool:
    resname, _, _ = _glyco_parse_residue_label(label)
    return resname in GLYCAN_RESNAMES


def _glyco_choose_anchor(component: list[str]) -> str:
    protein_nodes = [node for node in component if not _glyco_is_glycan_label(node)]
    return sorted(protein_nodes)[0] if protein_nodes else sorted(component)[0]


def _glyco_orient_link(parent: str, child: str, edge_data: dict[frozenset, dict]) -> dict:
    link = edge_data[frozenset((parent, child))]
    if link["residue1"] == parent and link["residue2"] == child:
        return {"parent": parent, "parent_atom": link["atom1"], "child": child, "child_atom": link["atom2"], "distance": link.get("distance"), "source": link["source"]}
    return {"parent": parent, "parent_atom": link["atom2"], "child": child, "child_atom": link["atom1"], "distance": link.get("distance"), "source": link["source"]}


def _glyco_build_oriented_tree(root: str, graph: dict[str, list[str]], edge_data: dict[frozenset, dict]) -> dict:
    visited = set()

    def walk(node: str, parent: str | None = None) -> dict:
        visited.add(node)
        children = []
        oriented_links = []
        for neighbor in sorted(graph[node]):
            if neighbor == parent or neighbor in visited:
                continue
            oriented_links.append(_glyco_orient_link(node, neighbor, edge_data))
            children.append(walk(neighbor, node))
        return {"residue": node, "children": children, "links_to_children": oriented_links}

    return walk(root)


def _glyco_component_records(components: list[list[str]], graph: dict[str, list[str]], edge_data: dict[frozenset, dict]) -> list[dict]:
    records = []
    for i, component in enumerate(components, 1):
        anchor = _glyco_choose_anchor(component)
        records.append({
            "component_id": i,
            "anchor": anchor,
            "nodes": component,
            "glycan_nodes": [n for n in component if _glyco_is_glycan_label(n)],
            "protein_nodes": [n for n in component if not _glyco_is_glycan_label(n)],
            "tree": _glyco_build_oriented_tree(anchor, graph, edge_data),
        })
    return records


def _glyco_template_external_atom_index(forcefield: ForceField) -> dict[str, dict[frozenset[str], list[str]]]:
    index: dict[str, dict[frozenset[str], list[str]]] = defaultdict(lambda: defaultdict(list))

    for template_name, template in forcefield._templates.items():
        if len(template_name) != 3:
            continue

        suffix = template_name[-2:]
        external_atoms = frozenset(template.atoms[i].name for i in template.externalBonds)
        if not external_atoms:
            continue
        if not any(atom in external_atoms for atom in ("C1", "C2", "O1")):
            continue

        index[suffix][external_atoms].append(template_name)

    return index


def _glyco_choose_template_from_index(
    resname: str,
    external_atoms: set[str],
    template_index: dict[str, dict[frozenset[str], list[str]]],
) -> tuple[str | None, str, str]:
    suffixes = PDB_TO_GLYCAM_SUFFIXES.get(resname, ())
    if not suffixes:
        return None, "unsupported", f"No PDB-to-GLYCAM residue-family mapping for {resname}"

    ext_key = frozenset(external_atoms)
    candidates = []
    for suffix in suffixes:
        candidates.extend(template_index.get(suffix, {}).get(ext_key, []))

    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        template = candidates[0]
        return template, "high", f"Matched GLYCAM template {template} by external atoms {sorted(external_atoms)}"
    if len(candidates) > 1:
        return None, "ambiguous", f"External atoms {sorted(external_atoms)} match multiple GLYCAM templates: {', '.join(candidates)}"

    return None, "unsupported", (
        f"No GLYCAM template for {resname} with external atoms {sorted(external_atoms)} "
        f"in family/families {', '.join(suffixes)}"
    )


def _glyco_choose_template_from_connectivity(
    resname: str,
    external_atoms: set[str],
    incoming_links: list[tuple],
) -> tuple[str | None, str, str] | None:
    """Resolve common ambiguous GLYCAM terminal templates from parent linkage motifs."""
    if resname == "GAL" and external_atoms == {"C1"}:
        for parent, parent_atom, child_atom, _link in incoming_links:
            parent_resname = parent.split(":")[0]
            if parent_resname in {"NAG", "YB", "0YB", "3YB", "4YB", "6YB", "UYB", "VYB", "WYB", "QYB"}:
                if parent_atom == "O4" and child_atom == "C1":
                    return (
                        "0LB",
                        "motif",
                        "Resolved terminal Gal-beta-1,4-GlcNAc motif from NAG O4 -> GAL C1 linkage",
                    )

    return None


def _glyco_assign_templates(
    links: list[dict],
    glycan_residues: set[str],
    template_index: dict[str, dict[frozenset[str], list[str]]],
) -> dict[str, dict]:
    external_atoms: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, list] = defaultdict(list)
    outgoing: dict[str, list] = defaultdict(list)

    for link in links:
        r1, a1 = link["residue1"], link["atom1"]
        r2, a2 = link["residue2"], link["atom2"]

        if r1 in glycan_residues:
            external_atoms[r1].add(a1)
        if r2 in glycan_residues:
            external_atoms[r2].add(a2)

        if a2 == "C1":
            parent, parent_atom, child, child_atom = r1, a1, r2, a2
        elif a1 == "C1":
            parent, parent_atom, child, child_atom = r2, a2, r1, a1
        else:
            continue

        incoming[child].append((parent, parent_atom, child_atom, link))
        outgoing[parent].append((child, parent_atom, child_atom, link))

    assignments: dict[str, dict] = {}

    for residue in sorted(glycan_residues):
        resname = residue.split(":")[0]
        ext = external_atoms.get(residue, set())
        template, confidence, reason = _glyco_choose_template_from_index(resname, ext, template_index)
        if template is None and confidence == "ambiguous":
            connectivity_choice = _glyco_choose_template_from_connectivity(resname, ext, incoming.get(residue, []))
            if connectivity_choice is not None:
                template, confidence, reason = connectivity_choice

        assignments[residue] = {
            "resname": resname,
            "template": template,
            "confidence": confidence,
            "reason": reason,
            "external_atoms": sorted(ext),
            "expected_external_atoms": sorted(ext) if template else [],
            "incoming_links": [
                {"parent": p, "parent_atom": pa, "child_atom": ca, "source": l.get("source"), "distance": l.get("distance")}
                for p, pa, ca, l in incoming.get(residue, [])
            ],
            "outgoing_links": [
                {"child": c, "parent_atom": pa, "child_atom": ca, "source": l.get("source"), "distance": l.get("distance")}
                for c, pa, ca, l in outgoing.get(residue, [])
            ],
            "substituted_positions": sorted(a for a in ext if a.startswith("O")),
        }

    return assignments


def _glyco_tree_to_ascii(tree: dict, edge_data: dict[frozenset, dict], template_map: dict[str, str], prefix: str = "", is_last: bool = True, parent: str | None = None) -> list[str]:
    label = tree["residue"]
    if label in template_map:
        _, chain, resid = _glyco_parse_residue_label(label)
        display = f"{template_map[label]}:{chain}:{resid}"
    else:
        display = label

    if parent is None:
        line = display
    else:
        link = _glyco_orient_link(parent, label, edge_data)
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{display}  ({link['parent_atom']}-{link['child_atom']}, d={link['distance'] if link['distance'] is not None else '-'}, {link['source']})"

    lines = [line]
    child_prefix = prefix
    if parent is not None:
        child_prefix += "    " if is_last else "│   "

    children = tree["children"]
    for i, child in enumerate(children):
        lines.extend(_glyco_tree_to_ascii(child, edge_data, template_map, child_prefix, i == len(children) - 1, label))
    return lines


def _glyco_write_outputs(outdir: Path, glycan_residues: set[str], links: list[dict], component_records: list[dict], graph: dict[str, list[str]], glycam_assignments: dict[str, dict], edge_data: dict[frozenset, dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    data = {
        "glycan_residues": sorted(glycan_residues),
        "links": links,
        "glycam_assignments": glycam_assignments,
        "components": component_records,
        "graph": {k: sorted(v) for k, v in graph.items()},
    }

    with open(outdir / "glycans.json", "w") as out:
        json.dump(data, out, indent=2)

    with open(outdir / "glycans.tsv", "w") as out:
        out.write("residue1\tatom1\tresidue2\tatom2\tdistance\tsource\n")
        for link in links:
            out.write(f"{link['residue1']}\t{link['atom1']}\t{link['residue2']}\t{link['atom2']}\t{link.get('distance', '-')}\t{link['source']}\n")

    with open(outdir / "glycam_assignments.tsv", "w") as out:
        out.write("residue\tresname\ttemplate\tconfidence\treason\texternal_atoms\n")
        for residue, a in glycam_assignments.items():
            out.write(f"{residue}\t{a['resname']}\t{a['template']}\t{a['confidence']}\t{a['reason']}\t{','.join(a['external_atoms'])}\n")

    template_map = {residue: a["template"] for residue, a in glycam_assignments.items() if a.get("template")}
    with open(outdir / "glycan_trees.txt", "w") as out:
        for record in component_records:
            out.write(f"Component {record['component_id']} | anchor={record['anchor']} | glycan_residues={len(record['glycan_nodes'])}\n")
            for line in _glyco_tree_to_ascii(record["tree"], edge_data, template_map):
                out.write(line + "\n")
            out.write("\n")


def detect_glycans_for_prepare(
    pdb_file: str | Path,
    outdir: Path,
    forcefield: ForceField,
    use_link_records: bool = True,
    use_distance_detection: bool = True,
) -> dict:
    logging.info("Detecting covalently bound glycans for GLYCAM preparation...")
    atoms, pdb_links = _glyco_parse_pdb(pdb_file)
    residues = _glyco_group_residues(atoms)
    glycan_residues = {_glyco_residue_label(k) for k in residues if k[3] in GLYCAN_RESNAMES}

    all_links: list[dict] = []
    if use_link_records:
        all_links.extend(_glyco_detect_links_from_records(residues, pdb_links))
    if use_distance_detection:
        all_links.extend(_glyco_detect_protein_glycan_links(residues))
        all_links.extend(_glyco_detect_glycan_glycan_links(residues))

    links = _glyco_unique_links(all_links)
    graph, edge_data = _glyco_build_graph(links)
    components = _glyco_connected_components(graph)
    component_records = _glyco_component_records(components, graph, edge_data)
    template_index = _glyco_template_external_atom_index(forcefield)
    assignments = _glyco_assign_templates(links, glycan_residues, template_index)

    _glyco_write_outputs(outdir, glycan_residues, links, component_records, graph, assignments, edge_data)

    logging.info(f"Detected {len(glycan_residues)} glycan residues")
    logging.info(f"Detected {len(links)} covalent glycan links")
    logging.info(f"Detected {len(component_records)} glycan components")

    template_map = {residue: a["template"] for residue, a in assignments.items() if a.get("template")}
    for record in component_records:
        logging.info(f"Glycan {record['component_id']} | anchor={record['anchor']} | glycan_residues={len(record['glycan_nodes'])}")
        for line in _glyco_tree_to_ascii(record["tree"], edge_data, template_map):
            logging.info("  " + line)

    return {
        "glycan_residues": sorted(glycan_residues),
        "links": links,
        "glycam_assignments": assignments,
        "components": component_records,
        "graph": {k: sorted(v) for k, v in graph.items()},
    }


def find_glycosylated_asn_labels_from_glycan_data(glycans_data: dict) -> set[str]:
    labels = set()
    for link in glycans_data.get("links", []):
        r1, a1 = link["residue1"], link["atom1"]
        r2, a2 = link["residue2"], link["atom2"]
        if r1.startswith("ASN:") and a1 == "ND2" and r2.startswith("NAG:") and a2 == "C1":
            labels.add(r1)
        elif r2.startswith("ASN:") and a2 == "ND2" and r1.startswith("NAG:") and a1 == "C1":
            labels.add(r2)
    return labels


def rename_glycosylated_asn_to_nln(modeller: Modeller, glycans_data: dict) -> None:
    glyco_asn_labels = find_glycosylated_asn_labels_from_glycan_data(glycans_data)
    renamed = []
    for residue in modeller.topology.residues():
        label_as_asn = f"ASN:{residue.chain.id}:{residue.id}"
        if label_as_asn in glyco_asn_labels:
            old = residue.name
            residue.name = "NLN"
            renamed.append((label_as_asn, old, residue.name))
    for label, old, new in renamed:
        logging.info(f"Renamed glycosylated residue {label}: {old} -> {new}")
    logging.info(f"Renamed {len(renamed)} glycosylated ASN residues to NLN")


def rename_glycan_atoms_to_glycam(modeller: Modeller) -> None:
    atom_rename = {"NAG": {"C7": "C2N", "O7": "O2N", "C8": "CME"}}
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for atom in modeller.topology.atoms():
        residue = atom.residue
        mapping = atom_rename.get(residue.name)
        if mapping and atom.name in mapping:
            old = atom.name
            atom.name = mapping[old]
            grouped[f"{residue.name}:{residue.chain.id}:{residue.id}"].append((old, atom.name))
    for label, changes in grouped.items():
        logging.info(f"GLYCAM atom renaming for {label}: " + ", ".join(f"{o}->{n}" for o, n in changes))
    logging.info(f"Renamed {sum(len(v) for v in grouped.values())} glycan atoms to GLYCAM atom names")


def rename_glycan_residues_to_glycam_templates(modeller: Modeller, glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    renamed = []
    for residue in modeller.topology.residues():
        label = f"{residue.name}:{residue.chain.id}:{residue.id}"
        if label not in assignments:
            continue
        template = assignments[label].get("template")
        if not template:
            logging.warning(f"No GLYCAM template assignment for {label}; leaving residue name unchanged")
            continue
        old = residue.name
        residue.name = template
        renamed.append((label, old, template))
    for label, old, new in renamed:
        logging.info(f"Renamed glycan residue {label}: {old} -> {new}")
    logging.info(f"Renamed {len(renamed)} glycan residues to GLYCAM template names")


def validate_glycam_assignments(glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    unsupported = [
        (residue, data)
        for residue, data in sorted(assignments.items())
        if not data.get("template")
    ]

    if not unsupported:
        logging.info("All detected glycan residues have GLYCAM template assignments.")
        return

    logging.error("Unsupported glycan residues detected; cannot safely continue with GLYCAM preparation.")
    for residue, data in unsupported:
        logging.error(
            f"  {residue}: resname={data.get('resname')} "
            f"external_atoms={data.get('external_atoms')} reason={data.get('reason')}"
        )
    logging.error(
        "Add an explicit GLYCAM assignment rule for these residues/linkages, "
        "or supply a curated PDB with supported GLYCAM residue/template names."
    )
    sys.exit(1)


def build_nln_residue_template_map(topology: Topology, glycans_data: dict) -> dict:
    glyco_asn_labels = find_glycosylated_asn_labels_from_glycan_data(glycans_data)
    residue_templates = {}
    for residue in topology.residues():
        label_as_asn = f"ASN:{residue.chain.id}:{residue.id}"
        label_as_nln = f"NLN:{residue.chain.id}:{residue.id}"
        if label_as_asn in glyco_asn_labels or label_as_nln in glyco_asn_labels:
            ext = external_atoms_for_residue(topology, residue)
            if {"N", "C", "ND2"}.issubset(ext):
                template = "NLN"
            elif {"C", "ND2"}.issubset(ext) and "N" not in ext:
                template = "NNLN"
            elif {"N", "ND2"}.issubset(ext) and "C" not in ext:
                template = "CNLN"
            else:
                template = "NLN"
                logging.warning(
                    f"Could not infer NLN terminal variant for {residue.name}:{residue.chain.id}:{residue.id}; "
                    f"external_atoms={sorted(ext)}. Falling back to NLN."
                )
            residue_templates[residue] = template
            logging.info(
                f"Using explicit glycosylated ASN template "
                f"{residue.name}:{residue.chain.id}:{residue.id} -> {template}"
            )
    logging.info(f"Using {len(residue_templates)} explicit glycosylated ASN residue template assignments")
    return residue_templates


def expected_external_atoms_for_glycam_template(template: str) -> set[str] | None:
    expected = {
        "0YB": {"C1"}, "3YB": {"C1", "O3"}, "4YB": {"C1", "O4"},
        "6YB": {"C1", "O6"}, "UYB": {"C1", "O4", "O6"},
        "VYB": {"C1", "O3", "O6"}, "WYB": {"C1", "O3", "O4"}, "QYB": {"C1", "O3", "O4", "O6"},
        "0MA": {"C1"}, "3MA": {"C1", "O3"}, "4MA": {"C1", "O4"}, "6MA": {"C1", "O6"},
        "VMA": {"C1", "O3", "O6"}, "WMA": {"C1", "O3", "O4"}, "UMA": {"C1", "O4", "O6"}, "QMA": {"C1", "O3", "O4", "O6"},
        "0MB": {"C1"}, "3MB": {"C1", "O3"}, "4MB": {"C1", "O4"}, "6MB": {"C1", "O6"},
        "VMB": {"C1", "O3", "O6"}, "WMB": {"C1", "O3", "O4"}, "UMB": {"C1", "O4", "O6"}, "QMB": {"C1", "O3", "O4", "O6"},
        "0FA": {"C1"},
    }
    return expected.get(template)


def external_atoms_for_residue(topology: Topology, residue) -> set[str]:
    ext = set()
    for a1, a2 in topology.bonds():
        if a1.residue == residue and a2.residue != residue:
            ext.add(a1.name)
        elif a2.residue == residue and a1.residue != residue:
            ext.add(a2.name)
    return ext


def log_glycam_sanity_checks(modeller: Modeller, glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    assignment_by_chain_resid = {}
    for old_label, data in assignments.items():
        _, chain, resid = _glyco_parse_residue_label(old_label)
        assignment_by_chain_resid[(chain, resid)] = data

    passed = 0
    warnings = 0
    logging.info("GLYCAM topology sanity checks:")
    for residue in modeller.topology.residues():
        key = (residue.chain.id, residue.id)
        if key not in assignment_by_chain_resid:
            continue
        template = assignment_by_chain_resid[key].get("template")
        expected_atoms = assignment_by_chain_resid[key].get("expected_external_atoms") or []
        expected = set(expected_atoms) if expected_atoms else expected_external_atoms_for_glycam_template(template)
        found = external_atoms_for_residue(modeller.topology, residue)
        label = f"{residue.name}:{residue.chain.id}:{residue.id}"
        if expected is None:
            logging.warning(f"  {label}: no sanity rule for template {template}")
            warnings += 1
        elif expected == found:
            logging.info(f"  [OK] {label}: expected external={sorted(expected)} found={sorted(found)}")
            passed += 1
        else:
            logging.error(f"  [FAIL] {label}: expected external={sorted(expected)} found={sorted(found)}")
            warnings += 1
    logging.info(f"GLYCAM sanity checks passed for {passed} glycan residues")
    if warnings:
        logging.warning(f"{warnings} glycan sanity checks need inspection")


def dump_glyco_debug_files(modeller: Modeller, residue_templates: dict, outdir: Path, target_index: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    debug_pdb = outdir / "debug_before_hydrogens.pdb"
    debug_residue = outdir / f"debug_residue_{target_index}.txt"
    debug_templates = outdir / "debug_template_residues.txt"

    with open(debug_pdb, "w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    with open(debug_residue, "w") as out:
        wrote = False
        for residue in modeller.topology.residues():
            if residue.index != target_index:
                continue
            wrote = True
            out.write(f"Residue index: {residue.index}\n")
            out.write(f"Residue: {residue.name}:{residue.chain.id}:{residue.id}\n\n")
            out.write("Atoms:\n")
            for atom in residue.atoms():
                out.write(f"  {atom.index:5d} {atom.name}\n")
            out.write("\nBonds involving this residue:\n")
            for a1, a2 in modeller.topology.bonds():
                if a1.residue == residue or a2.residue == residue:
                    out.write(f"  {a1.residue.name}:{a1.residue.chain.id}:{a1.residue.id}:{a1.name} -- {a2.residue.name}:{a2.residue.chain.id}:{a2.residue.id}:{a2.name}\n")
            break
        if not wrote:
            out.write(f"No residue with topology index {target_index} found.\n")

    with open(debug_templates, "w") as out:
        for residue, template in residue_templates.items():
            out.write("=" * 80 + "\n")
            out.write(f"Residue index: {residue.index}\nResidue: {residue.name}:{residue.chain.id}:{residue.id}\nTemplate: {template}\n\n")
            out.write("Atoms:\n")
            for atom in residue.atoms():
                out.write(f"  {atom.index:5d} {atom.name}\n")
            out.write("\nBonds involving this residue:\n")
            for a1, a2 in modeller.topology.bonds():
                if a1.residue == residue or a2.residue == residue:
                    out.write(f"  {a1.residue.name}:{a1.residue.chain.id}:{a1.residue.id}:{a1.name} -- {a2.residue.name}:{a2.residue.chain.id}:{a2.residue.id}:{a2.name}\n")
            out.write("\n")

    logging.info(f"Wrote glycoprotein debug PDB: {debug_pdb}")
    logging.info(f"Wrote glycoprotein debug residue dump: {debug_residue}")
    logging.info(f"Wrote glycoprotein debug template dump: {debug_templates}")


def prepare_glycoprotein_modeller(
    args: argparse.Namespace,
    forcefield: ForceField,
    glyco_outdir: Path,
) -> tuple[Modeller, dict]:
    glycans_data = detect_glycans_for_prepare(
        args.receptor,
        glyco_outdir,
        forcefield,
        use_link_records=not args.glyco_no_link_records,
        use_distance_detection=not args.glyco_no_distance_detection,
    )
    pdbfixer_input = write_pdb_with_supported_glycan_links_only(
        args.receptor,
        glyco_outdir / "pdbfixer_supported_links.pdb",
    )

    fixer = prepare_protein(
        pdbfixer_input,
        ncaa=args.ncaa,
        ncaa_resnames=args.ncaa_resnames,
    )
    modeller = Modeller(fixer.topology, fixer.positions)

    validate_glycam_assignments(glycans_data)
    rename_glycosylated_asn_to_nln(modeller, glycans_data)
    rename_glycan_atoms_to_glycam(modeller)
    rename_glycan_residues_to_glycam_templates(modeller, glycans_data)

    if args.ncaa:
        patch_ncaa_internal_bonds_from_ffxml(modeller.topology, args.ncaa_resnames, args.ncaa_ffxml)
        patch_ncaa_peptide_bonds(modeller.topology, args.ncaa_resnames)

    residue_templates = build_nln_residue_template_map(modeller.topology, glycans_data)
    log_glycam_sanity_checks(modeller, glycans_data)

    if args.glyco_debug:
        dump_glyco_debug_files(modeller, residue_templates, glyco_outdir, args.glyco_debug_residue_index)

    variants = None
    if args.metal_aware_protonation:
        variants = build_metal_aware_protonation_variants(
            modeller.topology,
            modeller.positions,
            args.metal_coordination_cutoff,
        )

    logging.info("Adding hydrogens using Modeller/ForceField templates in GLYCAM mode...")
    modeller.addHydrogens(forcefield, pH=args.ph, variants=variants, residueTemplates=residue_templates)
    logging.info(f"Atoms after GLYCAM hydrogen addition: {modeller.topology.getNumAtoms()}")

    return modeller, glycans_data

def run_prepare(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 1: SYSTEM PREPARATION ---")

    forcefield_files = ["amber14/protein.ff14SB.xml"]

    if args.membrane:
        logging.info("Membrane mode enabled: adding lipid force field.")
        forcefield_files.append(args.lipid_ff)

    forcefield_files.append(args.water_ff)

    if args.glycoprotein:
        logging.info("Glycoprotein mode enabled: adding GLYCAM force field.")
        forcefield_files.append(args.glycam_xml)

    if args.ncaa and args.ncaa_ffxml:
        logging.info("Loading NCAA ffxml files:")
        for ffxml in args.ncaa_ffxml:
            logging.info(f"  - {ffxml}")
            forcefield_files.append(ffxml)

    forcefield = ForceField(*forcefield_files)

    if args.glycoprotein:
        logging.info(f"Loading GLYCAM hydrogen definitions: {args.glycam_hydrogens}")
        Modeller.loadHydrogenDefinitions(args.glycam_hydrogens)

    if args.ncaa and args.ncaa_ffxml:
        load_ncaa_hydrogen_definitions(args.ncaa_ffxml)

    if args.glycoprotein:
        glyco_outdir = Path(args.output_dir) / "glycan_prepare"
        modeller, _glycans_data = prepare_glycoprotein_modeller(args, forcefield, glyco_outdir)
    else:
        protein_fixer = prepare_protein(
            args.receptor,
            ncaa=args.ncaa,
            ncaa_resnames=args.ncaa_resnames,
        )

        modeller = Modeller(protein_fixer.topology, protein_fixer.positions)

        if args.strip_caps:
            strip_amber_caps(modeller)

        if args.ncaa:
            patch_ncaa_internal_bonds_from_ffxml(
                modeller.topology,
                args.ncaa_resnames,
                args.ncaa_ffxml,
            )

            patch_ncaa_peptide_bonds(
                modeller.topology,
                args.ncaa_resnames,
            )

        variants = None
        if args.metal_aware_protonation:
            variants = build_metal_aware_protonation_variants(
                modeller.topology,
                modeller.positions,
                args.metal_coordination_cutoff,
            )

        logging.info("Adding hydrogens using Modeller/ForceField templates...")
        modeller.addHydrogens(forcefield, pH=args.ph, variants=variants)

    ligand_residues_to_ignore = set()

    if args.ligand:
        if not _has_openff:
            logging.error("Cannot process ligand. Please install openff-toolkit and rdkit.")
            sys.exit(1)

        ligand_mol = prepare_ligand(args.ligand, residue_name=args.ligand_resname)
        ligand_topology, ligand_positions = parametrize_ligand(ligand_mol, forcefield, args.lig_ff)
        existing_residue_count = sum(1 for _ in modeller.topology.residues())
        modeller.add(ligand_topology, ligand_positions)
        ligand_residues_to_ignore = {
            residue
            for residue in modeller.topology.residues()
            if residue.index >= existing_residue_count
        }
        for residue in sorted(ligand_residues_to_ignore, key=lambda r: r.index):
            logging.info(
                "Ligand residue will be handled by the registered template generator: "
                f"{residue.name}:{residue.chain.id}:{residue.id}"
            )
        logging.info("Ligand added to the system.")

    check_unmatched_residues(forcefield, modeller.topology, args, ignored_residues=ligand_residues_to_ignore)

    if args.membrane:
        membrane_center_z = choose_membrane_center_z(modeller, args)

        logging.info(
            "Adding membrane, solvent, and ions "
            f"(lipid={args.lipid_type}, padding={args.membrane_padding:.2f} nm, "
            f"center_z={membrane_center_z:.2f} nm)..."
        )
        modeller.addMembrane(
            forcefield,
            lipidType=args.lipid_type,
            membraneCenterZ=membrane_center_z * nanometer,
            minimumPadding=args.membrane_padding * nanometer,
            positiveIon=args.positive_ion,
            negativeIon=args.negative_ion,
            ionicStrength=args.ionic_strength * molar,
            neutralize=True,
        )
    else:
        logging.info("Adding solvent and ions...")
        modeller.addSolvent(
            forcefield,
            model=args.water_model,
            padding=args.solvent_padding * nanometer,
            positiveIon=args.positive_ion,
            negativeIon=args.negative_ion,
            ionicStrength=args.ionic_strength * molar,
            neutralize=True,
        )
    logging.info(f"System size: {modeller.topology.getNumAtoms()} atoms.")

    logging.info("Creating OpenMM System object...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=args.hydrogen_mass * amu,
    )

    if args.metal_coordination_restraints:
        add_metal_coordination_restraints(
            system,
            modeller.topology,
            modeller.positions,
            args.metal_coordination_cutoff,
            args.metal_coordination_k,
        )

    prepared_pdb_path = os.path.join(args.output_dir, "prepared.pdb")
    system_xml_path = os.path.join(args.output_dir, "system.xml")

    logging.info(f"Saving prepared structure to {prepared_pdb_path}")
    with open(prepared_pdb_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    logging.info(f"Saving serialized System to {system_xml_path}")
    with open(system_xml_path, "w") as f:
        f.write(XmlSerializer.serialize(system))

    logging.info("--- PREPARATION STAGE COMPLETE ---")

def run_equilibrate(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 2: SYSTEM EQUILIBRATION ---")

    prepared_pdb_path = os.path.join(args.output_dir, "prepared.pdb")
    system_xml_path = os.path.join(args.output_dir, "system.xml")

    if not all(os.path.exists(p) for p in [prepared_pdb_path, system_xml_path]):
        logging.error("Cannot find prepared files. Please run the --prepare step first.")
        sys.exit(1)

    logging.info(f"Loading topology from {prepared_pdb_path}")
    pdb = PDBFile(prepared_pdb_path)
    topology = pdb.topology
    positions = pdb.positions

    logging.info(f"Loading system from {system_xml_path}")
    with open(system_xml_path, "r") as f:
        system = XmlSerializer.deserialize(f.read())

    platform = select_platform(args.platform)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, args.timestep_fs * femtoseconds)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    logging.info("Running energy minimization...")
    simulation.minimizeEnergy()
    logging.info("Minimization complete.")

    logging.info("Heating system to 300K (NVT)...")
    simulation.context.setVelocitiesToTemperature(5 * kelvin)

    for temp in tqdm(range(5, 305, 5), desc="NVT Heating", ncols=80):
        integrator.setTemperature(temp * kelvin)
        simulation.step(500)

    integrator.setTemperature(300 * kelvin)
    logging.info("Heating complete.")

    logging.info("Equilibrating system (NPT) with decreasing restraints...")
    system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))

    restraint = CustomExternalForce("0.5*k_backbone_restraint*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k_backbone_restraint", 4000.0 * kilojoules_per_mole / nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    backbone_indices = [
        a.index
        for a in topology.atoms()
        if a.residue.name not in ("HOH", "NA", "CL") and a.name in ("CA", "C", "N", "O")
    ]

    logging.info(f"Applying position restraints to {len(backbone_indices)} backbone atoms.")

    initial_positions = simulation.context.getState(getPositions=True).getPositions()

    for i in backbone_indices:
        restraint.addParticle(i, initial_positions[i])

    system.addForce(restraint)
    simulation.context.reinitialize(preserveState=True)

    restraint_levels = [4000.0, 2000.0, 1000.0, 500.0, 250.0, 100.0, 50.0, 0.0]

    for k in tqdm(restraint_levels, desc="NPT Equilibration", ncols=80):
        simulation.context.setParameter("k_backbone_restraint", k)
        simulation.step(12500)

    logging.info("NPT equilibration complete.")

    equil_system_path = os.path.join(args.output_dir, "equilibrated_system.xml")
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")
    equil_pdb_path = os.path.join(args.output_dir, "equilibrated.pdb")

    logging.info(f"Saving equilibrated system with barostat and restraints to {equil_system_path}")
    with open(equil_system_path, "w") as f:
        f.write(XmlSerializer.serialize(system))

    logging.info(f"Saving equilibrated state to {equil_state_path}")
    simulation.saveState(equil_state_path)

    logging.info(f"Saving equilibrated structure to {equil_pdb_path}")
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    with open(equil_pdb_path, "w") as f:
        PDBFile.writeFile(topology, final_positions, f, keepIds=True)

    logging.info("--- EQUILIBRATION STAGE COMPLETE ---")


# ============================================================
# ENHANCED SAMPLING / METADYNAMICS HELPERS
# ============================================================

BACKBONE_DIHEDRAL_TYPES = ("phi", "psi")


def residue_label(residue) -> str:
    """PDB residue id combined with its insertion code, e.g. "27A" for resSeq 27 / icode A.

    OpenMM's Residue.id is only the numeric resSeq; the insertion code lives in the separate
    Residue.insertionCode attribute. PDB-deposited antibody structures very commonly carry
    insertion-coded CDR residues (e.g. Kabat/Chothia 27, 27A, 27B, ...), so residue.id alone is
    not unique within a chain for those residues.
    """
    icode = (residue.insertionCode or "").strip()
    return f"{residue.id}{icode}"


def resolve_phi_psi_atoms(
    topology: Topology,
    cv_type: str,
    residue_id,
    chain_id: str | None = None,
) -> tuple[int, int, int, int]:
    if cv_type not in BACKBONE_DIHEDRAL_TYPES:
        raise ValueError(f"Unsupported torsion CV type '{cv_type}'. Must be one of {BACKBONE_DIHEDRAL_TYPES}.")

    matches = []

    for chain in topology.chains():
        if chain_id is not None and chain.id != str(chain_id):
            continue

        residues = list(chain.residues())

        for i, residue in enumerate(residues):
            if residue_label(residue) != str(residue_id):
                continue

            if cv_type == "phi":
                if i == 0:
                    raise ValueError(
                        f"Cannot compute phi for residue {residue_id} in chain {chain.id}: "
                        "it is the first residue in its chain (no preceding C atom)."
                    )
                prev_residue = residues[i - 1]
                try:
                    atoms = (
                        next(a.index for a in prev_residue.atoms() if a.name == "C"),
                        next(a.index for a in residue.atoms() if a.name == "N"),
                        next(a.index for a in residue.atoms() if a.name == "CA"),
                        next(a.index for a in residue.atoms() if a.name == "C"),
                    )
                except StopIteration:
                    raise ValueError(
                        f"Could not find N/CA/C backbone atoms for phi of residue {residue_id} "
                        f"in chain {chain.id}."
                    )
            else:
                if i == len(residues) - 1:
                    raise ValueError(
                        f"Cannot compute psi for residue {residue_id} in chain {chain.id}: "
                        "it is the last residue in its chain (no following N atom)."
                    )
                next_residue = residues[i + 1]
                try:
                    atoms = (
                        next(a.index for a in residue.atoms() if a.name == "N"),
                        next(a.index for a in residue.atoms() if a.name == "CA"),
                        next(a.index for a in residue.atoms() if a.name == "C"),
                        next(a.index for a in next_residue.atoms() if a.name == "N"),
                    )
                except StopIteration:
                    raise ValueError(
                        f"Could not find N/CA/C backbone atoms for psi of residue {residue_id} "
                        f"in chain {chain.id}."
                    )

            matches.append((chain.id, atoms))

    if not matches:
        chain_hint = f" in chain {chain_id}" if chain_id is not None else ""
        raise ValueError(f"No residue with id {residue_id}{chain_hint} was found in the topology.")

    if len(matches) > 1:
        chains_found = ", ".join(c for c, _ in matches)
        raise ValueError(
            f"Residue id {residue_id} is ambiguous: found in chains {chains_found}. "
            "Specify 'chain' in the metadynamics CV config to disambiguate."
        )

    return matches[0][1]


def build_metadynamics(
    system,
    topology: Topology,
    metad_config: dict,
    temperature: Quantity,
    dt_ps: float,
) -> tuple[Metadynamics, list[str], list[tuple[int, int, int, int]]]:
    variables = []
    cv_labels = []
    cv_atoms = []

    for cv in metad_config["cvs"]:
        cv_type = cv["type"]
        residue_id = cv["residue"]
        chain_id = cv.get("chain")
        bias_width_degrees = cv.get("bias_width_degrees", 25.0)
        grid_width = cv.get("grid_width")

        atoms = resolve_phi_psi_atoms(topology, cv_type, residue_id, chain_id)

        force = CustomTorsionForce("theta")
        force.addTorsion(*atoms)

        variables.append(
            BiasVariable(
                force,
                -math.pi,
                math.pi,
                math.radians(bias_width_degrees),
                periodic=True,
                gridWidth=grid_width,
            )
        )

        label = f"{cv_type}_{chain_id or ''}{residue_id}"
        cv_labels.append(label)
        cv_atoms.append(atoms)
        logging.info(
            f"Metadynamics CV '{label}': atoms={atoms}, bias_width={bias_width_degrees} deg, "
            f"grid_width={grid_width or 'auto'}"
        )

    bias_factor = metad_config.get("bias_factor", 10.0)
    height_kjmol = metad_config.get("height_kjmol", 1.2)
    deposition_interval_ps = metad_config.get("deposition_interval_ps", 1.0)
    save_interval_ps = metad_config.get("save_interval_ps", 100.0)

    frequency = max(1, round(deposition_interval_ps / dt_ps))
    save_frequency = max(frequency, round(save_interval_ps / dt_ps))
    if save_frequency % frequency != 0:
        save_frequency = frequency * round(save_frequency / frequency)
        logging.warning(
            f"Rounded metadynamics save_interval_ps so it is a multiple of the deposition "
            f"interval: save_frequency={save_frequency} steps, frequency={frequency} steps."
        )

    bias_dir = metad_config["_bias_dir"]
    os.makedirs(bias_dir, exist_ok=True)

    meta = Metadynamics(
        system,
        variables,
        temperature,
        biasFactor=bias_factor,
        height=height_kjmol * kilojoules_per_mole,
        frequency=frequency,
        saveFrequency=save_frequency,
        biasDir=bias_dir,
    )

    logging.info(
        f"Metadynamics enabled: CVs={cv_labels}, bias_factor={bias_factor}, "
        f"height={height_kjmol} kJ/mol, frequency={frequency} steps, "
        f"save_frequency={save_frequency} steps, bias_dir={bias_dir}"
    )

    return meta, cv_labels, cv_atoms


@dataclass
class MetadynamicsGroup:
    meta: Metadynamics
    cv_labels: list[str]
    cv_atoms: list[tuple[int, int, int, int]]


def build_metadynamics_groups(
    system,
    topology: Topology,
    metad_config: dict,
    temperature: Quantity,
    dt_ps: float,
    bias_dir: str,
) -> list[MetadynamicsGroup]:
    """Split metad_config['cvs'] into chunks of at most 3 (OpenMM's per-bias-force CV limit),
    building one Metadynamics object per chunk so a single walker can bias an arbitrary number
    of torsions.
    """
    cvs = metad_config["cvs"]
    groups = []

    for chunk_index, start in enumerate(range(0, len(cvs), 3)):
        chunk_config = dict(metad_config)
        chunk_config["cvs"] = cvs[start : start + 3]
        chunk_config["_bias_dir"] = os.path.join(bias_dir, f"group_{chunk_index:02d}")

        meta, cv_labels, cv_atoms = build_metadynamics(system, topology, chunk_config, temperature, dt_ps)
        groups.append(MetadynamicsGroup(meta=meta, cv_labels=cv_labels, cv_atoms=cv_atoms))

    return groups


# ============================================================
# CDR-LOOP DETECTION (anarcii) FOR AUTOMATED FAB BIAS-EXCHANGE
# ============================================================
#
# IMGT CDR boundaries, identical for heavy and light chains, verified directly against the
# anarci package's own source (anarci/schemes.py): CDR1 = 27-38, CDR2 = 56-65, CDR3 = 105-117
# (inclusive).
IMGT_CDR_RANGES = {
    "1": (27, 38),
    "2": (56, 65),
    "3": (105, 117),
}

VALID_CDR_LOOP_NAMES = {"H1", "H2", "H3", "L1", "L2", "L3"}


def write_protein_only_pdb(topology: Topology, positions: Quantity, out_path: str) -> None:
    modeller = Modeller(topology, positions)
    to_delete = [r for r in modeller.topology.residues() if is_bulk_solvent_or_ion_residue(r)]
    modeller.delete(to_delete)

    with open(out_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


def run_anarcii_numbering(pdb_path: str) -> dict:
    from anarcii import Anarcii  # Lazy import: anarcii is only required for this opt-in feature.

    model = Anarcii(seq_type="antibody")
    return model.number(pdb_path)


def find_cdr_residues(topology: Topology, numbering_results: dict) -> dict[str, list[tuple[str, str]]]:
    """Maps anarcii's PDB numbering output back to (chain_id, residue_id) pairs per CDR loop.

    numbering_results is keyed by (model_index, chain_id); each entry's 'numbering' list holds
    ((imgt_number, insertion_code), residue_letter) pairs in the same order as the chain's
    actual residues (gaps marked '-' carry no residue). Since this is run on a solvent/ion-
    stripped copy of the same structure the production walkers use, residue order for protein
    chains is unaffected and the zip below lines up 1:1 with the real Topology residues.
    """
    cdr_loops: dict[str, list[tuple[str, str]]] = {}
    chains_by_id = {chain.id: chain for chain in topology.chains()}

    for (_model_index, chain_id), entry in numbering_results.items():
        if entry.get("numbering") is None:
            continue

        chain_type = entry.get("chain_type")
        if chain_type not in ("H", "K", "L"):
            continue

        prefix = "H" if chain_type == "H" else "L"

        chain = chains_by_id.get(chain_id)
        if chain is None:
            continue

        residues = list(chain.residues())
        query_start, query_end = entry["query_start"], entry["query_end"]
        residues_in_domain = residues[query_start : query_end + 1]

        non_gap_entries = [pos for pos, letter in entry["numbering"] if letter != "-"]

        if len(non_gap_entries) != len(residues_in_domain):
            logging.warning(
                f"anarcii numbering length mismatch for chain {chain_id}: "
                f"{len(non_gap_entries)} numbered positions vs {len(residues_in_domain)} residues "
                "in the detected domain; skipping this chain for CDR detection."
            )
            continue

        for (imgt_number, _insertion_code), residue in zip(non_gap_entries, residues_in_domain):
            for cdr_number, (start, end) in IMGT_CDR_RANGES.items():
                if start <= imgt_number <= end:
                    loop_name = f"{prefix}{cdr_number}"
                    cdr_loops.setdefault(loop_name, []).append((chain_id, residue_label(residue)))
                    break

    return cdr_loops


def build_cdr_bias_exchange_config(args: argparse.Namespace, topology: Topology, positions: Quantity) -> dict:
    auto_cfg = args.auto_cdr_bias_exchange

    protein_only_pdb_path = os.path.join(args.output_dir, "cdr_detection_protein_only.pdb")
    write_protein_only_pdb(topology, positions, protein_only_pdb_path)

    numbering_results = run_anarcii_numbering(protein_only_pdb_path)
    cdr_loops = find_cdr_residues(topology, numbering_results)

    requested_loops = auto_cfg.get("loops", sorted(VALID_CDR_LOOP_NAMES))
    bias_width_degrees = auto_cfg.get("bias_width_degrees", 25.0)

    walkers = []
    walker_labels = []

    for loop_name in requested_loops:
        residues = cdr_loops.get(loop_name)

        if not residues:
            logging.warning(f"No residues found for CDR loop {loop_name}; skipping.")
            continue

        cvs = [
            {"type": torsion_type, "chain": chain_id, "residue": residue_id, "bias_width_degrees": bias_width_degrees}
            for chain_id, residue_id in residues
            for torsion_type in ("phi", "psi")
        ]

        walker_cfg = {"cvs": cvs}
        for key in ("bias_factor", "height_kjmol", "deposition_interval_ps", "save_interval_ps"):
            if key in auto_cfg:
                walker_cfg[key] = auto_cfg[key]

        walkers.append(walker_cfg)
        walker_labels.append(f"CDR-{loop_name}")

        logging.info(f"CDR-{loop_name}: {len(residues)} residues, {len(cvs)} torsions biased.")

    return {
        "enabled": True,
        "exchange_interval_ps": auto_cfg.get("exchange_interval_ps", 2.0),
        "walkers": walkers,
        "_walker_labels": walker_labels,
    }


class MetadynamicsCVReporter:
    def __init__(self, meta: Metadynamics, cv_labels: list[str], csv_path: str, report_interval: int) -> None:
        self._meta = meta
        self._cv_labels = cv_labels
        self._reportInterval = report_interval
        self._csv_path = csv_path

        with open(self._csv_path, "w") as f:
            f.write("step,time_ps," + ",".join(f"{label}_deg" for label in cv_labels) + "\n")

    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, False)

    def report(self, simulation: Simulation, state: State) -> None:
        dt_ps = simulation.integrator.getStepSize().value_in_unit(picoseconds)
        time_ps = simulation.currentStep * dt_ps
        values_deg = [math.degrees(v) for v in self._meta.getCollectiveVariables(simulation)]

        with open(self._csv_path, "a") as f:
            f.write(f"{simulation.currentStep},{time_ps:.3f}," + ",".join(f"{v:.3f}" for v in values_deg) + "\n")


class MultiGroupMetadynamicsCVReporter:
    """Like MetadynamicsCVReporter, but logs CVs from every bias group of a multi-group walker."""

    def __init__(self, groups: list["MetadynamicsGroup"], csv_path: str, report_interval: int) -> None:
        self._groups = groups
        self._reportInterval = report_interval
        self._csv_path = csv_path

        all_labels = [label for group in groups for label in group.cv_labels]
        with open(self._csv_path, "w") as f:
            f.write("step,time_ps," + ",".join(f"{label}_deg" for label in all_labels) + "\n")

    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, False)

    def report(self, simulation: Simulation, state: State) -> None:
        dt_ps = simulation.integrator.getStepSize().value_in_unit(picoseconds)
        time_ps = simulation.currentStep * dt_ps

        all_values_deg = [
            math.degrees(v)
            for group in self._groups
            for v in group.meta.getCollectiveVariables(simulation)
        ]

        with open(self._csv_path, "a") as f:
            f.write(
                f"{simulation.currentStep},{time_ps:.3f}," + ",".join(f"{v:.3f}" for v in all_values_deg) + "\n"
            )


# ============================================================
# BIAS-EXCHANGE METADYNAMICS
# ============================================================
#
# Multiple walkers, each biasing a different torsion CV, periodically attempt to swap
# configurations. Because every walker shares the same force field and topology, the
# unbiased potential energy cancels out of the exchange criterion, leaving only the
# walkers' bias values:
#
#   delta = beta * [V_i(CV_i at x_j) + V_j(CV_j at x_i) - V_i(CV_i at x_i) - V_j(CV_j at x_j)]
#   accept with probability min(1, exp(-delta))
#
# so no full-system energy evaluation is needed for the exchange test itself.


def compute_dihedral_radians(positions: np.ndarray, atom_indices: tuple[int, int, int, int]) -> float:
    i, j, k, l = atom_indices
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(b2 / np.linalg.norm(b2), n1)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return float(math.atan2(y, x))


def evaluate_metadynamics_bias(meta: Metadynamics, cv_values_radians: list[float]) -> float:
    """Linearly interpolate Metadynamics' accumulated bias at an arbitrary CV point.

    OpenMM's public Metadynamics API only evaluates the bias through the live CustomCVForce
    table inside a Context (cubic-spline interpolation). This reads the private
    _totalBias/_limits arrays instead, so the bias-exchange Metropolis test can be evaluated
    for a *foreign* configuration without needing a Context for it. Only this acceptance test
    uses the (linear, slightly less accurate than OpenMM's spline) approximation; the biased
    dynamics themselves are driven entirely by OpenMM's own meta.step().
    """
    variables = meta.variables
    n = len(variables)
    bias = meta._totalBias

    fractions = []
    for i, v in enumerate(variables):
        x = (cv_values_radians[i] - v.minValue) / (v.maxValue - v.minValue)
        if v.periodic:
            x = x % 1.0
        fractions.append(x * (v.gridWidth - 1))

    # numpy axis order is reversed relative to variable order (see Metadynamics._addGaussian).
    axis_fracs = list(reversed(fractions))
    grid_widths = [v.gridWidth for v in reversed(variables)]
    periodics = [v.periodic for v in reversed(variables)]

    lowers = [int(math.floor(f)) for f in axis_fracs]
    weights = [f - lo for f, lo in zip(axis_fracs, lowers)]

    total = 0.0
    for corner in itertools.product((0, 1), repeat=n):
        idx = []
        weight = 1.0
        for c, lo, w, gw, periodic in zip(corner, lowers, weights, grid_widths, periodics):
            gi = lo + c
            if periodic:
                gi = gi % gw
            else:
                gi = min(max(gi, 0), gw - 1)
            idx.append(gi)
            weight *= w if c == 1 else (1.0 - w)
        total += weight * bias[tuple(idx)]

    return float(total)


def _walker_bias_at(walker: "BiasExchangeWalker", positions: np.ndarray) -> float:
    """Sum this walker's bias (across all of its groups) evaluated at a given configuration."""
    total = 0.0
    for group in walker.groups:
        cv_values = [compute_dihedral_radians(positions, atoms) for atoms in group.cv_atoms]
        total += evaluate_metadynamics_bias(group.meta, cv_values)
    return total


def metropolis_exchange_delta(
    walker_i: "BiasExchangeWalker",
    walker_j: "BiasExchangeWalker",
    positions_i: np.ndarray,
    positions_j: np.ndarray,
    temperature: Quantity,
) -> float:
    kT = (MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(kilojoules_per_mole)

    v_i_i = _walker_bias_at(walker_i, positions_i)
    v_i_j = _walker_bias_at(walker_i, positions_j)
    v_j_i = _walker_bias_at(walker_j, positions_i)
    v_j_j = _walker_bias_at(walker_j, positions_j)

    return (v_i_j + v_j_i - v_i_i - v_j_j) / kT


@dataclass
class BiasExchangeWalker:
    index: int
    label: str
    simulation: Simulation
    groups: list[MetadynamicsGroup]
    output_dir: str
    checkpoint_path: str
    final_checkpoint_path: str


def step_bias_exchange_walker(walker: BiasExchangeWalker, steps: int) -> None:
    """Advance a (possibly multi-group) walker by `steps` integration steps, depositing a
    Gaussian for every bias group whose own deposition interval has elapsed.

    This replicates Metadynamics.step()'s internal deposit logic (same private _addGaussian/
    _syncWithDisk members already relied on elsewhere in this module) across however many
    groups the walker holds, advancing the shared integrator exactly once per sub-chunk rather
    than once per group — calling each group's own .step() separately would wrongly advance
    time once per group.
    """
    simulation = walker.simulation
    steps_to_go = steps

    while steps_to_go > 0:
        next_steps = steps_to_go

        for group in walker.groups:
            frequency = group.meta.frequency
            current = simulation.currentStep
            distance = frequency if current % frequency == 0 else frequency - (current % frequency)
            next_steps = min(next_steps, distance)

        simulation.step(next_steps)

        for group in walker.groups:
            meta = group.meta

            if simulation.currentStep % meta.frequency == 0:
                force_group_index = meta._force.getForceGroup()
                position = meta._force.getCollectiveVariableValues(simulation.context)
                energy = simulation.context.getState(energy=True, groups={force_group_index}).getPotentialEnergy()
                height = meta.height * np.exp(-energy / (MOLAR_GAS_CONSTANT_R * meta._deltaT))
                meta._addGaussian(position, height, simulation.context)

            if meta.saveFrequency is not None and simulation.currentStep % meta.saveFrequency == 0:
                meta._syncWithDisk()

        steps_to_go -= next_steps


def attempt_exchange(
    walker_i: BiasExchangeWalker,
    walker_j: BiasExchangeWalker,
    temperature: Quantity,
) -> tuple[bool, float]:
    state_i = walker_i.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    state_j = walker_j.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)

    positions_i = state_i.getPositions(asNumpy=True).value_in_unit(nanometer)
    positions_j = state_j.getPositions(asNumpy=True).value_in_unit(nanometer)

    delta = metropolis_exchange_delta(walker_i, walker_j, positions_i, positions_j, temperature)
    accepted = delta <= 0.0 or random.random() < math.exp(-delta)

    if accepted:
        box_i = state_i.getPeriodicBoxVectors()
        box_j = state_j.getPeriodicBoxVectors()

        walker_i.simulation.context.setPositions(state_j.getPositions())
        walker_i.simulation.context.setVelocities(state_j.getVelocities())
        walker_i.simulation.context.setPeriodicBoxVectors(*box_j)

        walker_j.simulation.context.setPositions(state_i.getPositions())
        walker_j.simulation.context.setVelocities(state_i.getVelocities())
        walker_j.simulation.context.setPeriodicBoxVectors(*box_i)

    return accepted, delta


def build_bias_exchange_walkers(
    args: argparse.Namespace,
    topology: Topology,
    system_xml_string: str,
    temperature: Quantity,
    dt_ps: float,
    platform: Platform,
) -> list[BiasExchangeWalker]:
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")
    walker_labels = args.bias_exchange.get("_walker_labels")
    walkers = []

    for i, walker_cfg in enumerate(args.bias_exchange["walkers"]):
        label = walker_labels[i] if walker_labels else f"walker_{i:02d}"
        walker_dir = os.path.join(args.output_dir, label)
        os.makedirs(walker_dir, exist_ok=True)

        system = XmlSerializer.deserialize(system_xml_string)

        metad_config = dict(walker_cfg)
        bias_dir = os.path.join(walker_dir, metad_config.get("bias_dir", "metad_bias"))

        groups = build_metadynamics_groups(system, topology, metad_config, temperature, dt_ps, bias_dir)

        integrator = LangevinMiddleIntegrator(temperature, 1 / picosecond, args.timestep_fs * femtoseconds)
        simulation = Simulation(topology, system, integrator, platform)

        checkpoint_path = os.path.join(walker_dir, "production.chk")
        final_checkpoint_path = os.path.join(walker_dir, "production_final.chk")

        if args.restart:
            restart_candidates = [p for p in [final_checkpoint_path, checkpoint_path] if os.path.exists(p)]

            if not restart_candidates:
                logging.error(f"--restart was requested, but no checkpoint file was found for walker {label}.")
                sys.exit(1)

            latest_checkpoint = max(restart_candidates, key=os.path.getmtime)
            logging.info(f"Walker {label}: restart requested. Loading checkpoint from {latest_checkpoint}")
            simulation.loadCheckpoint(latest_checkpoint)
        else:
            logging.info(f"Walker {label}: loading state from {equil_state_path}")
            simulation.loadState(equil_state_path)

        walkers.append(
            BiasExchangeWalker(
                index=i,
                label=label,
                simulation=simulation,
                groups=groups,
                output_dir=walker_dir,
                checkpoint_path=checkpoint_path,
                final_checkpoint_path=final_checkpoint_path,
            )
        )

    return walkers


def run_production_bias_exchange(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 3: PRODUCTION MD (BIAS-EXCHANGE METADYNAMICS) ---")

    equil_system_path = os.path.join(args.output_dir, "equilibrated_system.xml")
    equil_pdb_path = os.path.join(args.output_dir, "equilibrated.pdb")
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")

    if not all(os.path.exists(p) for p in [equil_system_path, equil_pdb_path]):
        logging.error("Cannot find equilibrated files. Please run --prepare and --equilibrate first.")
        sys.exit(1)

    if not args.restart and not os.path.exists(equil_state_path):
        logging.error("Cannot find equilibrated state. Please run --equilibrate first, or use --restart.")
        sys.exit(1)

    logging.info(f"Loading topology from {equil_pdb_path}")
    pdb = PDBFile(equil_pdb_path)
    topology = pdb.topology

    if args.auto_cdr_bias_exchange and args.auto_cdr_bias_exchange.get("enabled"):
        logging.info("Auto CDR bias-exchange enabled: detecting CDR loops via anarcii.")
        args.bias_exchange = build_cdr_bias_exchange_config(args, topology, pdb.positions)

    logging.info(f"Loading system from {equil_system_path}")
    with open(equil_system_path, "r") as f:
        system_xml_string = f.read()

    platform = select_platform(args.platform)
    dt_ps = args.timestep_fs / 1000.0
    temperature = 300 * kelvin

    production_steps = int(args.prod_ns * 1000 / dt_ps)
    report_interval_ps = 50
    report_interval = int(report_interval_ps / dt_ps)
    checkpoint_interval_ps = 100000.0
    checkpoint_interval = int(checkpoint_interval_ps / dt_ps)

    exchange_interval_ps = args.bias_exchange.get("exchange_interval_ps", 2.0)
    exchange_interval = max(1, round(exchange_interval_ps / dt_ps))

    walkers = build_bias_exchange_walkers(args, topology, system_xml_string, temperature, dt_ps, platform)

    for walker in walkers:
        dcd_path = os.path.join(walker.output_dir, "production.dcd")
        log_path = os.path.join(walker.output_dir, "production.csv")
        metad_cv_path = os.path.join(walker.output_dir, "metad_cv.csv")

        total_steps_for_reporter = walker.simulation.currentStep + production_steps

        walker.simulation.reporters.append(DCDReporter(dcd_path, report_interval))
        walker.simulation.reporters.append(
            StateDataReporter(
                log_path,
                report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                density=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=total_steps_for_reporter,
                separator=",",
            )
        )
        walker.simulation.reporters.append(CheckpointReporter(walker.checkpoint_path, checkpoint_interval))
        walker.simulation.reporters.append(LoggingReporter())
        walker.simulation.reporters.append(
            MultiGroupMetadynamicsCVReporter(walker.groups, metad_cv_path, report_interval)
        )

    exchange_log_path = os.path.join(args.output_dir, "bias_exchange_log.csv")
    with open(exchange_log_path, "w") as f:
        f.write("cycle,step,walker_i,walker_j,delta,accepted\n")

    logging.info(
        f"Starting {args.prod_ns} ns bias-exchange production with {len(walkers)} walkers "
        f"({production_steps} steps, exchange every {exchange_interval} steps)..."
    )

    steps_done = 0
    cycle = 0

    while steps_done < production_steps:
        steps_this_cycle = min(exchange_interval, production_steps - steps_done)

        for walker in walkers:
            step_bias_exchange_walker(walker, steps_this_cycle)

        steps_done += steps_this_cycle

        offset = cycle % 2
        pairs = [(i, i + 1) for i in range(offset, len(walkers) - 1, 2)]

        with open(exchange_log_path, "a") as f:
            for i, j in pairs:
                accepted, delta = attempt_exchange(walkers[i], walkers[j], temperature)
                f.write(f"{cycle},{steps_done},{i},{j},{delta:.4f},{int(accepted)}\n")
                logging.info(
                    f"Exchange attempt cycle={cycle} walkers=({walkers[i].label},{walkers[j].label}) "
                    f"delta={delta:.3f} accepted={accepted}"
                )

        cycle += 1

    for walker in walkers:
        logging.info(f"Walker {walker.label}: saving final checkpoint to {walker.final_checkpoint_path}")
        walker.simulation.saveCheckpoint(walker.final_checkpoint_path)

        for group_idx, group in enumerate(walker.groups):
            free_energy_path = os.path.join(walker.output_dir, f"group_{group_idx:02d}_free_energy.npy")
            axes_path = os.path.join(walker.output_dir, f"group_{group_idx:02d}_free_energy_axes.json")

            np.save(free_energy_path, group.meta.getFreeEnergy().value_in_unit(kilojoules_per_mole))

            with open(axes_path, "w") as f:
                json.dump(
                    [
                        {
                            "label": label,
                            "min_degrees": math.degrees(variable.minValue),
                            "max_degrees": math.degrees(variable.maxValue),
                            "grid_width": variable.gridWidth,
                        }
                        for label, variable in zip(group.cv_labels, group.meta.variables)
                    ],
                    f,
                    indent=2,
                )

            logging.info(f"Walker {walker.label}: saved group {group_idx} free energy surface to {free_energy_path}")

    logging.info("--- PRODUCTION STAGE COMPLETE (BIAS-EXCHANGE METADYNAMICS) ---")


def run_production(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 3: PRODUCTION MD ---")

    equil_system_path = os.path.join(args.output_dir, "equilibrated_system.xml")
    equil_pdb_path = os.path.join(args.output_dir, "equilibrated.pdb")
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")

    checkpoint_path = os.path.join(args.output_dir, "production.chk")
    final_checkpoint_path = os.path.join(args.output_dir, "production_final.chk")

    if not all(os.path.exists(p) for p in [equil_system_path, equil_pdb_path]):
        logging.error("Cannot find equilibrated files. Please run --prepare and --equilibrate first.")
        sys.exit(1)

    logging.info(f"Loading topology from {equil_pdb_path}")
    pdb = PDBFile(equil_pdb_path)
    topology = pdb.topology

    logging.info(f"Loading system from {equil_system_path}")
    with open(equil_system_path, "r") as f:
        system = XmlSerializer.deserialize(f.read())

    platform = select_platform(args.platform)
    dt_ps = args.timestep_fs / 1000.0

    meta = None
    cv_labels = []

    if args.metadynamics and args.metadynamics.get("enabled"):
        metad_config = dict(args.metadynamics)
        metad_config["_bias_dir"] = os.path.join(args.output_dir, metad_config.get("bias_dir", "metad_bias"))
        meta, cv_labels, _cv_atoms = build_metadynamics(system, topology, metad_config, 300 * kelvin, dt_ps)

    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, args.timestep_fs * femtoseconds)
    simulation = Simulation(topology, system, integrator, platform)

    if args.restart:
        restart_candidates = [p for p in [final_checkpoint_path, checkpoint_path] if os.path.exists(p)]

        if restart_candidates:
            latest_checkpoint = max(restart_candidates, key=os.path.getmtime)
            logging.info(f"Restart requested. Loading checkpoint from {latest_checkpoint}")
            simulation.loadCheckpoint(latest_checkpoint)
        else:
            logging.error("--restart was requested, but no checkpoint file was found.")
            sys.exit(1)

    else:
        if not os.path.exists(equil_state_path):
            logging.error("Cannot find equilibrated state. Please run --equilibrate first, or use --restart.")
            sys.exit(1)

        logging.info(f"Loading state from {equil_state_path}")
        simulation.loadState(equil_state_path)

    dcd_path = os.path.join(args.output_dir, "production.dcd")
    log_path = os.path.join(args.output_dir, "production.csv")

    report_interval_ps = 50
    report_interval = int(report_interval_ps / dt_ps)
    production_steps = int(args.prod_ns * 1000 / dt_ps)

    checkpoint_interval_ps = 100000.0
    checkpoint_interval = int(checkpoint_interval_ps / dt_ps)

    total_steps_for_reporter = simulation.currentStep + production_steps

    simulation.reporters.append(DCDReporter(dcd_path, report_interval))
    simulation.reporters.append(
        StateDataReporter(
            log_path,
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps_for_reporter,
            separator=",",
        )
    )
    simulation.reporters.append(CheckpointReporter(checkpoint_path, checkpoint_interval))
    simulation.reporters.append(LoggingReporter())

    if meta is not None:
        metad_cv_path = os.path.join(args.output_dir, "metad_cv.csv")
        simulation.reporters.append(MetadynamicsCVReporter(meta, cv_labels, metad_cv_path, report_interval))

    logging.info(f"Starting {args.prod_ns} ns production run ({production_steps} steps)...")

    if meta is not None:
        meta.step(simulation, production_steps)
    else:
        simulation.step(production_steps)

    logging.info(f"Saving final checkpoint to {final_checkpoint_path}")
    simulation.saveCheckpoint(final_checkpoint_path)

    if meta is not None:
        free_energy_path = os.path.join(args.output_dir, "metad_free_energy.npy")
        axes_path = os.path.join(args.output_dir, "metad_free_energy_axes.json")

        np.save(free_energy_path, meta.getFreeEnergy().value_in_unit(kilojoules_per_mole))

        with open(axes_path, "w") as f:
            json.dump(
                [
                    {
                        "label": label,
                        "min_degrees": math.degrees(variable.minValue),
                        "max_degrees": math.degrees(variable.maxValue),
                        "grid_width": variable.gridWidth,
                    }
                    for label, variable in zip(cv_labels, meta.variables)
                ],
                f,
                indent=2,
            )

        logging.info(f"Saved metadynamics free energy surface to {free_energy_path} (axes: {axes_path})")

    logging.info("--- PRODUCTION STAGE COMPLETE ---")


def main() -> None:
    conf_parser = argparse.ArgumentParser(
        description="A unified pipeline for OpenMM simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    conf_parser.add_argument("-c", "--config", help="Path to JSON config file.", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}
    config_dir = None

    if args.config:
        try:
            config_path = Path(args.config).expanduser()
            config_dir = config_path.resolve().parent

            with open(config_path) as f:
                defaults = json.load(f)
                print(f"Loaded configuration from {args.config}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load or parse config file: {e}")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description="A unified pipeline for OpenMM simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-r", "--receptor", help="PDB file for the protein receptor.")
    parser.add_argument("-l", "--ligand", help="SDF file for the ligand optional.")
    parser.add_argument("-o", "--output_dir", help="Directory to save all output files.")

    parser.add_argument(
        "--platform",
        default=None,
        help=(
            "OpenMM platform to use for equilibration/production (e.g. CUDA, OpenCL, CPU, "
            "Reference). Defaults to auto-detecting the fastest usable platform. Override this "
            "if auto-detection picks a platform that is compiled in but not actually usable on "
            "this machine (a known issue with some OpenCL driver setups)."
        ),
    )

    parser.add_argument("--prepare", action="store_true", help="Run the system preparation stage.")
    parser.add_argument("--equilibrate", action="store_true", help="Run the equilibration stage.")
    parser.add_argument("--produce", action="store_true", help="Run the production MD stage.")

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart production from the latest available checkpoint in the output directory.",
    )

    parser.add_argument("--prod_ns", type=float, default=1.0, help="Length of production run in nanoseconds.")

    parser.add_argument(
        "--metadynamics",
        type=json.loads,
        default=None,
        help=(
            "JSON object configuring well-tempered metadynamics torsion biasing during the "
            "production stage, e.g. "
            '\'{"enabled": true, "cvs": [{"type": "phi", "chain": "A", "residue": 45}]}\'. '
            "Normally supplied via --config."
        ),
    )

    parser.add_argument(
        "--bias_exchange",
        type=json.loads,
        default=None,
        help=(
            "JSON object configuring bias-exchange metadynamics during the production stage: "
            "multiple walkers, each biasing its own torsion CV(s), periodically swapping "
            "configurations. Mutually exclusive with --metadynamics. Example: "
            '\'{"enabled": true, "exchange_interval_ps": 2.0, "walkers": '
            '[{"cvs": [{"type": "phi", "chain": "A", "residue": 45}]}, '
            '{"cvs": [{"type": "psi", "chain": "A", "residue": 45}]}]}\'. '
            "Normally supplied via --config."
        ),
    )

    parser.add_argument(
        "--auto_cdr_bias_exchange",
        type=json.loads,
        default=None,
        help=(
            "JSON object enabling automated bias-exchange metadynamics for antibody Fabs: "
            "detects CDR loop residues with anarcii (pip install anarcii) and biases every "
            "backbone torsion of each loop, one walker per CDR loop. Mutually exclusive with "
            "--metadynamics and --bias_exchange. Example: "
            '\'{"enabled": true, "loops": ["H1","H2","H3","L1","L2","L3"]}\'. '
            "Normally supplied via --config."
        ),
    )

    parser.add_argument(
        "--lig_ff",
        type=str,
        choices=["espaloma", "smirnoff"],
        default="espaloma",
        help="Force field for ligand parameterization.",
    )
    parser.add_argument(
        "--ligand_resname",
        type=str,
        default=None,
        help="PDB residue name for the ligand. Defaults to the SDF filename stem, truncated to 3 characters.",
    )

    parser.add_argument("--timestep_fs", type=float, default=4.0, help="Integration timestep in femtoseconds.")

    parser.add_argument(
        "--hydrogen_mass",
        type=float,
        default=1.5,
        help="Mass for hydrogen atoms in amu. Use > 1.0 for H-mass repartitioning.",
    )

    parser.add_argument(
        "--water_model",
        default="tip3p",
        help="Water box model name passed to Modeller.addSolvent. Keep tip3p when using amber14/tip3pfb.xml.",
    )

    parser.add_argument(
        "--water_ff",
        default="amber14/tip3pfb.xml",
        help="OpenMM water force-field XML file.",
    )

    parser.add_argument(
        "--solvent_padding",
        type=float,
        default=1.0,
        help="Solvent box padding in nanometers for non-membrane systems.",
    )

    parser.add_argument(
        "--ionic_strength",
        type=float,
        default=0.15,
        help="Ionic strength in molar for solvent or membrane preparation.",
    )

    parser.add_argument(
        "--positive_ion",
        default="Na+",
        help="Positive ion type used for neutralization and salt.",
    )

    parser.add_argument(
        "--negative_ion",
        default="Cl-",
        help="Negative ion type used for neutralization and salt.",
    )

    parser.add_argument(
        "--membrane",
        action="store_true",
        help="Build a lipid membrane and water box around the prepared complex.",
    )

    parser.add_argument(
        "--lipid_type",
        default="POPC",
        help="Lipid type passed to Modeller.addMembrane.",
    )

    parser.add_argument(
        "--lipid_ff",
        default="amber14/lipid17.xml",
        help="OpenMM lipid force-field XML file used when --membrane is enabled.",
    )

    parser.add_argument(
        "--membrane_padding",
        type=float,
        default=1.0,
        help="Minimum membrane/water box padding in nanometers.",
    )

    parser.add_argument(
        "--membrane_center_z",
        type=float,
        default=None,
        help="Membrane center position along z in nanometers. Defaults to the prepared complex z-midpoint.",
    )

    parser.add_argument(
        "--membrane_center_chain_ids",
        nargs="*",
        default=[],
        help="Optional chain IDs used to auto-select membrane center z.",
    )

    parser.add_argument(
        "--membrane_center_residue_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Optional residue-id range used to auto-select membrane center z.",
    )

    parser.add_argument(
        "--membrane_auto_thickness",
        type=float,
        default=3.0,
        help="Hydrophobic slab thickness in nanometers for automatic membrane center detection.",
    )

    parser.add_argument("--ph", type=float, default=7.4, help="pH used for hydrogen addition.")

    parser.add_argument(
        "--strip_caps",
        action="store_true",
        help="Remove ACE/NME terminal caps before hydrogen addition and prepare ordinary termini.",
    )

    parser.add_argument(
        "--metal_aware_protonation",
        "--metal-aware-protonation",
        dest="metal_aware_protonation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force protonation variants from nearby metal contacts before hydrogen addition "
            "(currently histidine tautomers and cysteine thiolates). "
            "Use --no-metal-aware-protonation to disable."
        ),
    )

    parser.add_argument(
        "--metal_coordination_restraints",
        action="store_true",
        help="Add harmonic restraints between retained metal ions and nearby N/O/S donor atoms.",
    )

    parser.add_argument(
        "--metal_coordination_cutoff",
        type=float,
        default=3.0,
        help="Metal-donor detection cutoff in Angstrom.",
    )

    parser.add_argument(
        "--metal_coordination_k",
        type=float,
        default=50000.0,
        help="Metal coordination restraint force constant in kJ/mol/nm^2.",
    )

    parser.add_argument(
        "--ncaa",
        action="store_true",
        help="Enable noncanonical amino acid handling. Selected nonstandard residues are preserved.",
    )

    parser.add_argument(
        "--ncaa_resnames",
        nargs="*",
        default=[],
        help="Residue names to preserve as NCAAs, e.g. ALM PTR SEP.",
    )

    parser.add_argument(
        "--ncaa_ffxml",
        nargs="*",
        default=[],
        help="Custom OpenMM ffxml files containing NCAA residue templates.",
    )

    parser.add_argument(
        "--ncaa_mode",
        choices=["ffxml"],
        default="ffxml",
        help="NCAA parameterization mode. Current robust mode is ffxml.",
    )


    parser.add_argument(
        "--glycoprotein",
        action="store_true",
        help="Enable GLYCAM preparation for covalently glycosylated proteins during --prepare.",
    )

    parser.add_argument(
        "--glycam_xml",
        default="amber14/GLYCAM_06j-1.xml",
        help="OpenMM GLYCAM XML file used when --glycoprotein is enabled.",
    )

    parser.add_argument(
        "--glycam_hydrogens",
        default="glycam-hydrogens.xml",
        help="GLYCAM hydrogen definitions XML used by Modeller when --glycoprotein is enabled.",
    )

    parser.add_argument(
        "--glyco_debug",
        action="store_true",
        help="Write glycoprotein debug files before hydrogen addition.",
    )

    parser.add_argument(
        "--glyco_debug_residue_index",
        type=int,
        default=527,
        help="Topology residue index to dump when --glyco_debug is enabled.",
    )

    parser.add_argument(
        "--glyco_no_link_records",
        action="store_true",
        help="Ignore PDB LINK records during glycan detection and use distance detection only.",
    )

    parser.add_argument(
        "--glyco_no_distance_detection",
        action="store_true",
        help="Disable distance-based glycan bond detection and use LINK records only.",
    )

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    resolve_config_paths(args, config_dir)

    if not args.output_dir:
        print("ERROR: The --output_dir argument is required.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "pipeline.log")
    setup_logging(log_file)

    if args.config:
        logging.info(f"Loaded configuration from {args.config}")

    if args.ncaa and not args.ncaa_resnames:
        logging.error("--ncaa requires --ncaa_resnames, e.g. --ncaa_resnames ALM")
        sys.exit(1)

    if args.ncaa and not args.ncaa_ffxml:
        logging.error("--ncaa currently requires --ncaa_ffxml, e.g. --ncaa_ffxml ncaa/ALM.xml")
        sys.exit(1)

    if not (args.prepare or args.equilibrate or args.produce):
        logging.error("No action requested. Please specify at least one stage: --prepare, --equilibrate, or --produce.")
        sys.exit(1)

    if not args.receptor:
        logging.error("The --receptor argument is required.")
        sys.exit(1)

    validate_input_files(args)

    if args.prepare:
        run_prepare(args)

    if args.equilibrate:
        run_equilibrate(args)

    if args.produce:
        auto_cdr_enabled = args.auto_cdr_bias_exchange and args.auto_cdr_bias_exchange.get("enabled")
        bias_exchange_enabled = args.bias_exchange and args.bias_exchange.get("enabled")

        if auto_cdr_enabled or bias_exchange_enabled:
            run_production_bias_exchange(args)
        else:
            run_production(args)

    logging.info("All defined steps completed.")


if __name__ == "__main__":
    main()

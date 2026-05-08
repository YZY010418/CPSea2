import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
from tqdm import tqdm 
import argparse

def create_capping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("molecule cannot be generated from smiles")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol) 
    return mol

def set_pdb_info(mol, chain_id="A", residue_name="UNL", residue_number=1):
    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        atom_name = f"{symbol}{atom_counts[symbol]}"
        atom.SetProp("atom_name", atom_name)
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(
            atomName=atom_name,
            serialNumber=atom.GetIdx() + 1,
            residueName=residue_name,
            residueNumber=residue_number,
            chainId=chain_id,
            isHeteroAtom=True,
        ))


def atom_metadata(atom):
    monomer_info = atom.GetMonomerInfo()
    if monomer_info is not None:
        atom_name = monomer_info.GetName().strip()
        chain_id = monomer_info.GetChainId().strip() or "A"
        residue_name = monomer_info.GetResidueName().strip() or "UNL"
        residue_number = monomer_info.GetResidueNumber()
    else:
        atom_name = atom.GetProp("atom_name") if atom.HasProp("atom_name") else atom.GetSymbol()
        chain_id = "A"
        residue_name = "UNL"
        residue_number = 1

    return {
        "atom_index": atom.GetIdx(),
        "atom_serial": atom.GetIdx() + 1,
        "atom_name": atom_name,
        "element": atom.GetSymbol(),
        "chain_id": chain_id,
        "residue_name": residue_name,
        "residue_number": residue_number,
    }


def set_sdf_atom_name_property(mol):
    atom_names = [atom_metadata(atom)["atom_name"] for atom in mol.GetAtoms()]
    mol.SetProp("Atom_Names", ",".join(atom_names))


def save_atom_name_map(mol, output_dir, name):
    csv_path = os.path.join(output_dir, f"{name}_atom_names.csv")
    fieldnames = [
        "atom_index",
        "atom_serial",
        "atom_name",
        "element",
        "chain_id",
        "residue_name",
        "residue_number",
    ]
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for atom in mol.GetAtoms():
            writer.writerow(atom_metadata(atom))
    print(f"Atom name map is saved to: {csv_path}")


def generate_and_optimize_conformers(mol, num_conformers, seed=42):
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useExpTorsionAnglePrefs = True  
    params.useBasicKnowledge = True 
    
    print("Generating conformers...")
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if len(cids) == 0:
        raise RuntimeError("no conformer can be generated")
    
    print("Optimizing conformers...")
    results = []
    for conf_id in tqdm(cids, desc="Optimizing"):
        result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=200, confId=conf_id)
    
    return mol

def save_all_conformers(mol, output_path, output_dir, name):
    # Get all energies
    energies = []
    for conf_id in range(mol.GetNumConformers()):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        mmff_force_field = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        energy = mmff_force_field.CalcEnergy()
        energies.append((conf_id, energy))
    
    # Save all conformers regardless of energy
    all_conf_ids = [conf_id for conf_id, _ in energies]
    energy_by_conf_id = {conf_id: energy for conf_id, energy in energies}
    for conf_id in tqdm(all_conf_ids, desc="Saving SDF files"):
        sdf_path = f"{output_path}_{conf_id}.sdf"
        mol.SetProp("_Name", f"{name}_conformer_{conf_id}")
        set_sdf_atom_name_property(mol)
        mol.SetProp("Conf_ID", str(conf_id))
        mol.SetProp("Energy_kJ_per_mol", f"{energy_by_conf_id[conf_id]:.2f}")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol, confId=conf_id)
        writer.close()
    print(f"All conformations are saved to: {output_path}_*.sdf")

    # Save energy data
    csv_path = os.path.join(output_dir, f"{name}_energies.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Conf_ID', 'Energy (kJ/mol)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for conf_id, energy in energies:
            writer.writerow({'Conf_ID': conf_id, 'Energy (kJ/mol)': f"{energy:.2f}"})
    print(f"All conformation energies are saved to: {csv_path}")

def main(name, smiles, num_conformers, output_dir, seed=42):
    print(f"CappingGen with RDKit version {rdkit.__version__}")
    try:
        os.makedirs(output_dir, exist_ok=True)

        Capping = create_capping(smiles)
        print(f"Atom count: {Capping.GetNumAtoms()}")

        optimized_mol = generate_and_optimize_conformers(Capping, num_conformers, seed=seed)
        print(f"{optimized_mol.GetNumConformers()} conformers generated and optimized")

        set_pdb_info(optimized_mol)
        save_atom_name_map(optimized_mol, output_dir, name)
        output_all = os.path.join(output_dir, f"{name}_conformers")
        save_all_conformers(optimized_mol, output_all, output_dir, name)

    except Exception as e:
        print(f"error during CappingGen: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, default="Capping1")
    parser.add_argument("--smiles", type=str, required=True, default="CC1=NC=C(COP([O-])([O-])=O)C(C([H])=O)=C1O")
    parser.add_argument("--num", type=int, required=True, default=1000)
    parser.add_argument("--out", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ETKDG conformer generation.")
    args = parser.parse_args()
    main(args.name, args.smiles, args.num, args.out, seed=args.seed)

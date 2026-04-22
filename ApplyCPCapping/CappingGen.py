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
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(
            atomName=atom_name,
            serialNumber=atom.GetIdx() + 1,
            residueName=residue_name,
            residueNumber=residue_number,
            chainId=chain_id,
            isHeteroAtom=True,
        ))

def generate_and_optimize_conformers(mol, num_conformers):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  
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
    for conf_id in tqdm(all_conf_ids, desc="Saving PDB files"):
        pdb_path = f"{output_path}_{conf_id}.pdb"
        writer = Chem.PDBWriter(pdb_path)
        writer.write(mol, confId=conf_id)
        writer.close()
    print(f"All conformations are saved to: {output_path}_*.pdb")

    # Save energy data
    csv_path = os.path.join(output_dir, f"{name}_energies.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Conf_ID', 'Energy (kJ/mol)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for conf_id, energy in energies:
            writer.writerow({'Conf_ID': conf_id, 'Energy (kJ/mol)': f"{energy:.2f}"})
    print(f"All conformation energies are saved to: {csv_path}")

def main(name, smiles, num_conformers, output_dir):
    print(f"CappingGen with RDKit version {rdkit.__version__}")
    try:
        os.makedirs(output_dir, exist_ok=True)

        Capping = create_capping(smiles)
        print(f"Atom count: {Capping.GetNumAtoms()}")

        optimized_mol = generate_and_optimize_conformers(Capping, num_conformers)
        print(f"{optimized_mol.GetNumConformers()} conformers generated and optimized")

        set_pdb_info(optimized_mol)
        output_all = os.path.join(output_dir, f"{name}_conformers")
        save_all_conformers(optimized_mol, output_all, output_dir, name)

    except Exception as e:
        print(f"error during CappingGen: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, default="Capping1")
    parser.add_argument("--smiles", type=str, required=True, default="CC1=NC=C(COP([O-])([O-])=O)C(C([H])=O)=C1O")
    parser.add_argument("--num", type=int, required=True, default="1000")
    parser.add_argument("--out", type=str, default="./")
    args = parser.parse_args()
    main(args.name, args.smiles, args.num, args.out)

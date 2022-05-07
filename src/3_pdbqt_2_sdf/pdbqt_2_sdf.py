import os
from rdkit.Chem.rdmolfiles import MolFromMol2File
import rdkit
import re

l = rdkit.RDLogger.logger()
l.setLevel(0)

def transfer_bonds_charges(source_mol, source_mol_file, dest_mol_file, final_dest_file):

    # Get all atom lines from the destination molecule file
    with open(dest_mol_file, 'r') as f:
        dest_lines = f.readlines()
        dest_atom_lines = []
        iter_atom_lines = False
        for i, line in enumerate(dest_lines):
            if "@<TRIPOS>ATOM" in line:
                iter_atom_lines = True
            elif iter_atom_lines and "<" in line:
                iter_atom_lines = False
            elif iter_atom_lines and "<" not in line:
                dest_atom_lines.append(i)
            else:
                pass

    # Reorder atom lines from the destination molecule file according to atom_ids and ordering in _orig file.
    atom_counter = {}
    new_dest_atom_lines = []
    source_atoms = source_mol.GetAtoms()
    for i, source_atom in enumerate(source_atoms):
        sym = source_atom.GetSymbol().upper()
        if sym not in atom_counter.keys():
            atom_counter.update({sym:1})
        else: 
            atom_counter[sym] += 1

        for idx in dest_atom_lines:
            pattern = f'.*[0-9]+( +)({sym}{atom_counter[sym]}) (.*)'
            m = re.match(pattern, dest_lines[idx])
            if m:
                idx_part = str(i+1).rjust(7)
                modified_line = idx_part + m.group(1) + m.group(2) + ' ' + m.group(3) + '\n'
                new_dest_atom_lines.append(modified_line)

    with open(source_mol_file, 'r') as f:
        source_lines = f.readlines()

        # Get all bonds from source molecule
        source_bond_lines = []
        iter_bond_lines = False
        for line in source_lines:
            if "@<TRIPOS>BOND" in line:
                iter_bond_lines = True
                source_bond_lines.append(line)
            elif iter_bond_lines and "<" not in line:
                source_bond_lines.append(line)
            elif iter_bond_lines and "<" in line:
                iter_bond_lines = False
            else:
                pass
    
        # Get all other attributes (e.g. formal charge) from source molecule
        source_attr_lines = []
        iter_attr_lines = False
        for line in source_lines:
            if "@<TRIPOS>UNITY_ATOM_ATTR" in line:
                iter_attr_lines = True
                source_attr_lines.append(line)
            elif iter_attr_lines and "<" not in line:
                source_attr_lines.append(line)
            elif iter_attr_lines and "<" in line:
                iter_attr_lines = False
            else:
                pass
    
        # Get header from source molecule
        for i, line in enumerate(source_lines):
            if "@<TRIPOS>ATOM" in line:
                source_header_lines = source_lines[:i+1]
                break

    all_lines = source_header_lines + new_dest_atom_lines + source_attr_lines + source_bond_lines

    # Write modified destination file
    with open(final_dest_file, 'w') as f:
        f.writelines(all_lines)


orig_d = '1_orig_zinc'
pdbqt_d = '2_vina_output'
out_d = '3_cleaned_sdf'
bad_list = []
for file in os.listdir(pdbqt_d):

    if 'pdbqt' not in file:
        continue
    print(file)
    orig_basename = os.path.splitext(os.path.join(orig_d, file))[0]
    pdbqt_basename = os.path.splitext(os.path.join(pdbqt_d, file))[0]
    out_basename = os.path.splitext(os.path.join(out_d, file))[0]
    if os.path.exists(f'{out_basename}.sdf'):
        print(f'{out_basename}.sdf already exists.')
        continue

    try:
        # _orig.mol2 to removing hydrogens. This makes bond and formal charge indices 
        # line up with the indexing in the .pdbqt file
        cmd_0 = f"obabel {orig_basename}_orig.mol2 -O {pdbqt_basename}_bonds.mol2 -f 1 -l 1 -d"
        os.system(cmd_0)

        # pdbqt to mol2
        cmd_1 = f"obabel {pdbqt_basename}.pdbqt -O {pdbqt_basename}_coords.mol2 -f 1 -l 1"
        os.system(cmd_1)

        # Transfer bonds from original file
        mol2_source_file = pdbqt_basename + '_bonds.mol2'
        mol2_dest_file = pdbqt_basename + '_coords.mol2'
        mol2_final_file = pdbqt_basename + '_bonds_coords.mol2'
        mol2_source = MolFromMol2File(mol2_source_file, sanitize=False)
        transfer_bonds_charges(mol2_source, mol2_source_file, mol2_dest_file, mol2_final_file)

        # Add H's and convert to sdf
        cmd_2 = f"obabel {pdbqt_basename}_bonds_coords.mol2 -O {out_basename}.sdf -f 1 -l 1 -h"
        os.system(cmd_2)
        
    except Exception as e:
        if isinstance(e, KeyboardInterrupt) or isinstance(e, NameError):
            raise
        bad_list.append(file)

print("Bad list:\n", bad_list)


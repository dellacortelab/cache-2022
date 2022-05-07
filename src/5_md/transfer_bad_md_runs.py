##################
# Moves sdf's for bad ligand runs back into the notRun directory. Used for when MD runs are interrupted
##################


import os


md_output_dir = './5_md_runs'
ligands_dir = './3_cleaned_sdf_top_1500_notRun'
done_dir = './4_already_run'

needed_files = ['chkpnt_1.chk', 'Final_MD_3.pdb', 'NPT_2.pdb', 'trajectory_1.dcd', 'chkpnt_2.chk', 'minimized_1.pdb', 'NPT_3.pdb', 'trajectory_2.dcd', 'chkpnt_3.chk', 'minimized_2.pdb', 'performance_run1.txt', 'trajectory_3.dcd', 'Final_MD_1.pdb', 'minimized_3.pdb', 'performance_run2.txt', 'Final_MD_2.pdb', 'NPT_1.pdb', 'performance_run3.txt']

ligand_names = os.listdir(done_dir)
ligand_names = [os.path.splitext(name)[0] for name in ligand_names]

def is_bad_ligand(ligand_md_output_dir):

    if not os.path.exists(ligand_md_output_dir):
        return True

    for f in needed_files:
        if not os.path.exists(os.path.join(ligand_md_output_dir, f)):
            return True
    return False


bad_ligands = []
for ligand in ligand_names:
    ligand_md_output_dir = os.path.join(md_output_dir, ligand)
    if is_bad_ligand(ligand_md_output_dir):

        print('moving bad ligand', ligand)
        os.rename(os.path.join(done_dir, ligand + '.sdf'), os.path.join(ligands_dir, ligand + '.sdf'))

print(list(bad_ligands.keys()))


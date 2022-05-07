
import os
import argparse
parser = argparse.ArgumentParser(description='Start Small Molecule (.sdf) simulation')
parser.add_argument('--gpu', type=str, help='the gpu id')
args = parser.parse_args()

base_dir = '../../best_dock_poses/'
orig_dir = os.path.join(base_dir, '3_cleaned_sdf_ligand')
intermediate_dir = os.path.join(base_dir, '3_5_md_in_progress_ligand')
final_dir = os.path.join(base_dir, '4_already_run_ligand')
output_dir = os.path.join(base_dir, '5_md_runs_ligand')

os.makedirs(intermediate_dir, exist_ok=True)
while True:
    undone_files = os.listdir(orig_dir)
    if len(undone_files) == 0:
        break
    
    sdf = undone_files[0]
    orig_file = os.path.join(orig_dir, sdf)
    intermediate_file = os.path.join(intermediate_dir, sdf)
    final_file = os.path.join(final_dir, sdf)
    sdf_output_dir = os.path.join(output_dir, os.path.splitext(sdf)[0]) 
    os.makedirs(sdf_output_dir, exist_ok=True)

    os.rename(orig_file, intermediate_file)
    for taskID in range(1, 4):
        ret = os.system(f'python -u ./ligand_test.py -s {intermediate_file} --gpu {args.gpu} --length 50 --iter {taskID} --out {sdf_output_dir}')
    os.rename(intermediate_file, final_file)

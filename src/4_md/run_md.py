###############################
# Driver for MD runs
###############################

import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Start Small Molecule (.sdf) simulation')

parser.add_argument('--location', help='Where to run: on local GPUs or via sbatch.', required=True, choices=['local', 'sbatch'])
parser.add_argument('--mode', help='Whether to run on receptor, ligand, or the complex', choices=['receptor', 'ligand', 'complex'], default='complex')
parser.add_argument('--ligand_input_dir', default='../../data/3_cleaned_sdf')
parser.add_argument('--receptor_path', default='../../data/0_receptor/6DLO.pdb')
parser.add_argument('--intermediate_dir', default='../../data/3_5_md_in_progress')
parser.add_argument('--output_dir', default='../../data/4_md_runs')
parser.add_argument('--n_runs', type=int, default=3)
parser.add_argument('--reset', help='Remove files from intermediate_dir before running', action='store_true')
parser.add_argument('--gpu', help='The gpu id for running locally', type=int, default=0)
parser.add_argument('--email', help='The email address for sbatch reports')

args = parser.parse_args()

md_intermediate_root = os.path.join(args.intermediate_dir, args.mode)
md_output_root = os.path.join(args.output_dir, args.mode)

os.makedirs(md_intermediate_root, exist_ok=True)
os.makedirs(md_output_root, exist_ok=True)

if args.reset:
    for f in os.listdir(md_intermediate_root):
        os.remove(os.path.join(md_intermediate_root, f))

needed_files = [[f'chkpnt_{i}.chk', f'Final_MD_{i}.pdb', f'NPT_{i}.pdb', f'trajectory_{i}.dcd', f'minimized_{i}.pdb', f'performance_run{i}.txt'] for i in range(1, args.n_runs + 1)]
needed_files = [item for l in needed_files for item in l]

def is_ligand_finished(ligand_file):
    ligand_name = os.path.splitext(ligand_file)[0]
    sdf_output_dir = os.path.join(md_output_root, ligand_name)

    if not os.path.exists(sdf_output_dir):
        return False

    for f in needed_files:
        if not os.path.exists(os.path.join(sdf_output_dir, f)):
            return False

    return True

def get_ligand_files_queue():
    """Return the ligands to run MD on, removing the ligands that are finished and in progress"""
    ligand_files = os.listdir(args.ligand_input_dir)
    # Only include ligand files
    ligand_files = [l for l in ligand_files if os.path.splitext(l)[1] == '.sdf']
    finished_ligand_files = [ligand_file for ligand_file in ligand_files if is_ligand_finished(ligand_file)]
    in_progress_ligand_files = os.listdir(md_intermediate_root)
    ligand_files_queue = list(set(ligand_files) - set(finished_ligand_files) - set(in_progress_ligand_files))
    return ligand_files_queue

def launch_md_run(ligand_name=None, sbatch=True): 
    if ligand_name is not None:
        orig_file = os.path.join(args.ligand_input_dir, ligand_name + '.sdf')
        intermediate_file = os.path.join(md_intermediate_root, ligand_name + '.sdf')
        shutil.copy(orig_file, intermediate_file)
        output_dir = os.path.join(args.output_dir, ligand_name)
    else:
        receptor_name = os.path.splitext(os.path.basename(args.receptor_path))[0]
        output_dir = os.path.join(args.output_dir, receptor_name)

    os.makedirs(output_dir, exist_ok=True)
    
    print("launching")
    for taskID in range(1, args.n_runs + 1):
        cmd = f'python -u ./md.py --gpu {str(args.gpu)} --length 50 --iter {taskID} --mode {args.mode}'
        if args.mode != 'receptor':
            cmd += f' --ligand_name {ligand_name}'
        if args.mode != 'ligand':
            cmd += f' --receptor_path {args.receptor_path}'

        if sbatch:
            os.system(f'sbatch --output {output_dir} --mail-user {args.email} --job-name {args.mode} sbatch_run.sh "{cmd}"')
        else:
            os.system(cmd)
        print("launched")

# The difference between running locally and running using sbatch is that we handle queue management when running locally, and sbatch handles its own queue management
if args.location == 'local':
    if args.mode == 'receptor':
        launch_md_run(sbatch=False)
    else:
        while True:
            ligand_files_queue = get_ligand_files_queue()
            if len(ligand_files_queue) == 0:
                break
            ligand_name = os.path.splitext(ligand_files_queue[0])[0]
            launch_md_run(ligand_name=ligand_name, sbatch=False)
        
elif args.location == 'sbatch':
    if args.mode == 'receptor':
        launch_md_run(sbatch=False)
    else:
        ligand_files_queue = get_ligand_files_queue()
        for sdf in ligand_files_queue:
            ligand_name = os.path.splitext(sdf)[0]
            launch_md_run(ligand_name=ligand_name, sbatch=True)


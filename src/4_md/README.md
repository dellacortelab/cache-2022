# MD
Run molecular dynamics by running the `run_md.py` command. There are two required arguments to this script. `--mode` tells whether to run MD on the receptor, ligand, or the ligand-receptor complex. `--location` tells whether to run on local gpus or use sbatch to submit jobs to the supercomputer. If `--location` is "local", you should specify the id of the GPU to use.

For sbatch, you only need to run the script once. For running locally, you should start the command once on each GPU. There is a shared queue and each command will finish when the queue is empty.

Load dependencies:
```
conda create --file md_env.yml
conda activate md
```

Run on the complex locally. If running locally, the --gpu argument is required.
```python
python3 run_md.py --mode complex --location local --gpu 0
```
Run on just the ligand using sbatch. If using sbatch, the --email argument is required.
```python
python3 run_md.py --mode ligand --location sbatch --email example@gmail.com
```
To reset runs stuck in the _in_progress folder, add the --reset flag.

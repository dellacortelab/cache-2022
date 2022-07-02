# cache-2022

This is the codebase for the Della Corte lab submission to CACHE 2022. This is the general workflow:

1. Download ZINC/MCULE database
2. Run docking on ZINC/MCULE
3. Generate ML inputs for MILCDock from the docking runs
4. Generate MILC scores with MILCDock for each ligand
5. Run spectral clustering on the top 10k ligands to find 20 clusters, and select the top x% MILC scores in each cluster, such that we end up with 500 ligands
6. Run MD on the top 500 ligands
7. Run Free Energy and RMSF calculations on the top MD trajectories
8. Get final summed z-score for free energy, RMSF, and MILCDock score, and rank

####################
#
# Connor Morris
# Dennis Della Corte
# 2022-04-05
# Cache free energy template
# 
####################

import numpy as np
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

# Imports from the toolkit
import openff.toolkit
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
import time
from sys import stdout
from pathlib import Path
#from tempFile import NamedTemporaryFile
import mdtraj as mdt
import parmed as pmd
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import parmed
from simtk import openmm
import os
import argparse
from openff.toolkit.topology import Molecule

parser = argparse.ArgumentParser(description='Start Small Molecule (.sdf) simulation')
parser.add_argument('-s', dest='sdf',type=str, help='the input .sdf')
parser.add_argument('--gpu', type=str, help='the gpu id')
parser.add_argument('--length', dest='simLength',  type=int, default=100, 
                    help='Production simulation length in ns')
parser.add_argument('--out', dest='outPath',type=str, default='./',
                    help='the path for the file outputs')
parser.add_argument('--iter', dest='iter',type=int, default=0,
                    help='the iteration prefix')

args = parser.parse_args()
#initialize default parameters
simLength = args.simLength
numGPU = 1
outPath = args.outPath
iteration = args.iter
writeCheckpoint = True
molecule_path = args.sdf
receptor_path = '../code/6DLO_prepped.pdb' #TODO: Hardcode as always the same!
gpu_string = args.gpu

def save_pdb(sim, name):
    #Helper fct to quickly write out a pdb file in absence of a better option
    #Trying out a smarter way of saving PDBs
    position = simulation.context.getState(getPositions=True).getPositions()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    openmm.app.pdbfile.PDBFile.writeFile(simulation.topology, position, open(name+'.pdb','w'))
    print('Saved file: '+name+'.pdb')
    print(f'Energy: {energy._value*1.0/4.184} kcal/mol')

ligand = Molecule.from_file(molecule_path)
#molecule = Molecule.from_smiles('Nc1ccc(C(=O)OC[C@H](O)CO)cc1')

forcefield = openmm.app.ForceField('amber14-all.xml','amber14/tip3pfb.xml') #TODO: check if amber14sb works!

smirnoff = SMIRNOFFTemplateGenerator(
    forcefield="openff-2.0.0.offxml", molecules=[ligand]
)
print('Adding generator to forcefield...')
forcefield.registerTemplateGenerator(smirnoff.generator)

ligand_positions = ligand.conformers[0]
ligand_topology = ligand.to_topology()

# Create parmed Structure for ligand
# lig_system = forcefield.createSystem(ligand_topology, nonbondedMethod=openmm.app.PME,
#             nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
# structure = parmed.openmm.topsystem.load_topology( ligand_topology, lig_system,ligand_positions)
# # Write AMBER parameter/crd
# structure.save(os.path.join(outPath,'lig_system.prmtop'), overwrite=True)
# del lig_system
pdb = openmm.app.PDBFile(receptor_path)

#specify a box size
print("Made Modeller Object")
modeller = openmm.app.Modeller(pdb.topology, pdb.positions)

# Create parmed Structure for protein
# prot_system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
#             nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
# structure = parmed.openmm.topsystem.load_topology( pdb.topology, prot_system, pdb.positions)
# # Write AMBER parameter/crd
# structure.save(os.path.join(outPath,'prot_system.prmtop'), overwrite=True)
# del prot_system

modeller.add(ligand_topology.to_openmm(), ligand_positions)

# Create parmed Structure for protein-ligand complex
# complex_system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
#             nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
# structure = parmed.openmm.topsystem.load_topology( modeller.topology, complex_system, modeller.positions)
# # Write AMBER parameter/crd
# structure.save(os.path.join(outPath,'complex_system.prmtop'), overwrite=True)
# del complex_system

print("Solvating...")
modeller.addSolvent(forcefield, padding=0.9*openmm.unit.nanometers, positiveIon='Na+', negativeIon='Cl-',
                    ionicStrength=0.1*openmm.unit.molar) #padding like Feig, but physiological salt concentration

print("Solvation done. Creating system.")
system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)

# # Create parmed Structure for solvated protein-ligand complex
# structure = parmed.openmm.load_topology( modeller.getTopology(), system) #, modeller.positions)
# # Write AMBER parameter/crd
# structure.save(os.path.join(outPath,'complex_solvated_system.prmtop'), overwrite=True)

dt = 0.002 #ps

print(f"dt set to {dt*10**3:3.3f} fs")

#### ADD THE BAROSTAT FOR NPT EQUILIBRATION
pressure = 1 * openmm.unit.atmosphere  
temperature = 300 * openmm.unit.kelvin
barostat_frequency = 1  
barostat = openmm.openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
system.addForce(barostat)

integrator = openmm.openmm.LangevinIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, dt*openmm.unit.picoseconds)
    
#Putting multiple GPUs in place 

#gpu_string = ''
#for i in range(numGPU):
#    gpu_string += str(i)+','
#gpu_string = gpu_string[:-1]

platform = openmm.openmm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': gpu_string, 'Precision': 'mixed'}

simulation = openmm.app.simulation.Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

print('Start energy minimization')
simulation.minimizeEnergy()
save_pdb(simulation, os.path.join(outPath,"minimized_"+str(iteration)))
print('End energy minimization')

print("Start NPT Equilibration")
simulation.step(250*1000) #.5ns
save_pdb(simulation, os.path.join(outPath,"NPT_"+str(iteration)))
print("NPT Equilibration successful")

print("Generating new simulation object.")
#create new context w/out barostat
#get values from previous context
oldState = simulation.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
integrator = openmm.openmm.LangevinIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, dt*openmm.unit.picoseconds)
platform = openmm.openmm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': gpu_string, 'Precision': 'mixed'}

#generate new simulation object
simulation = openmm.app.simulation.Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setState(oldState)

#print("Start simulation with restraint", str(restraints_on))

start = time.time()

#report the # of steps taken each ns
print(f"Divide step by {int(1/(dt*10**-3))} to get time in ns")
simulation.reporters.append(openmm.app.StateDataReporter(stdout, int(1/(dt*10**-3)), step=True))
if writeCheckpoint:
    simulation.reporters.append(openmm.app.CheckpointReporter(os.path.join(outPath, f'chkpnt_{iteration}.chk'), int(5/(dt*10**-3))))#write a checkpoint every 5 ns
    print("Will save a checkpoint every 5 ns")

#Write out frames every 5 ps TODO: update after dG try runs
simulation.reporters.append(openmm.app.DCDReporter(os.path.join(outPath,'trajectory_'+str(iteration)+'.dcd'), int(0.05/(dt*10**-3)))) #write out DCD every 50 ps.
    
print('Starting full simulation.')
simulation.step(int(simLength/(dt*10**-3))) #length ns
simulation.reporters.pop()
print("Simulation done!")

save_pdb(simulation, os.path.join(outPath,"Final_MD_"+str(iteration)))
end = time.time()

with open(os.path.join(outPath,'performance_run'+str(iteration)+'.txt'),'w') as f:
    f.write(str(start)+'\n')
    f.write(str(end)+'\n')

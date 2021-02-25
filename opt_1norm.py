#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:37:15 2020

@author: emielkoridon
"""

import os
import numpy as np
from scipy.optimize import minimize
import scipy 
from random import random
from openfermion.functionals import JW1norm_spatfast
from pyscf import gto, scf, lo, tools
import module as md
import time


#%%======================================================|
# Set molecule and optimization parameters
#========================================================|

# Set molecule parameters.
basis = ['sto-3g', 'cc-pvdz','def2svp'][0]
multiplicity = 1

# Set calculation parameters.
optimize_occ = 0 # Do you want to also optimize the external occupied orbitals?
consider_cas = 0 # Do we consider an active space or the full space?
# Set size of active space
n_orbitals = 4
n_electrons = 4

# Import .xyz geometry. This is kind of an involved code because I have many
# different geometries available...
# If you want to consider a Hydrogen chain or ring, set the following to 1
H_chain = 0
H_ring = 0
if H_chain:
    N_atoms = 12
    r = 1.8
    geometry = [('H',( 0., 0., z*r)) for z in range(N_atoms)]
    
elif H_ring:
    theta = 88 * np.pi/180
    r = 1.3
    # Introduction of the molecular structure (.txt file)
    geometry = [
                ('H', (r*np.cos(theta/2.),   r*np.sin(theta/2.),  0.)),
                ('H', (r*np.cos(theta/2.),   -r*np.sin(theta/2.), 0.)),
                ('H', (-r*np.cos(theta/2.),  r*np.sin(theta/2.),  0.)),
                ('H', (-r*np.cos(theta/2.),  -r*np.sin(theta/2.), 0.))
                ]
else:
    fname = ['xyz_files/H2nosym.txt','xyz_files/H2COnosym.txt','xyz_files/H10.txt',\
             'xyz_files/C2.txt', 'xyz_files/LiH.txt', 'xyz_files/HLiO.txt', \
             'xyz_files/H2Onosym.txt', 'xyz_files/H14.txt', \
             'xyz_files/hnch2_s0min_dzp.txt', 'xyz_files/hnc3h6_s0min_dzp.txt',\
             'xyz_files/hnc5h10_s0min_dzp.txt', 'xyz_files/hnc7h14_s0min_dzp.txt',\
             'xyz_files/benzene.txt','xyz_files/PCy3.txt','xyz_files/PCy3Cl2Ru.txt',\
             'xyz_files/femoco.txt', 'xyz_files/M06-L.txt','xyz_files/butane.txt',\
             'xyz_files/pentane.txt','xyz_files/hexane.txt','xyz_files/heptane.txt',\
             'xyz_files/butene.txt','xyz_files/pentene.txt','xyz_files/hexene.txt',\
             'xyz_files/heptene.txt'][17]
    geometry = md.xyz_to_geom(fname)
     
# Determine number of AOs and electrons
nmo = md.count_ao(geometry,basis,spin=multiplicity-1)
nelec = md.count_elec(geometry,basis,spin=multiplicity-1)
print('Number of AOs:',nmo,'\nNumber of electrons:', nelec)
threshold = 1e-10

# Choose whether you want to start from an CMO or LMO reference
localize = 0
# Choose whether you want to start with a random initial rotation 
randomize = 0

# Set active indices
if consider_cas:
    ncore = (nelec - n_electrons) // 2
    ntot = ncore + n_orbitals # Number of core orbitals + number of active orbitals
    active_indices = list(range(ncore,ntot))
    occupied_indices = list(range(ncore))
else:
    ncore = 0
    ntot = nmo
    active_indices = list(range(nmo))
    occupied_indices = []

print(active_indices,occupied_indices)

# Build molecule and run RHF
mol = gto.Mole()
mol.atom = geometry
mol.basis = basis
mol.spin = multiplicity - 1
mol.symmetry = 0
mol.build()

if multiplicity == 1:
    myhf = scf.RHF(mol)
else:
    myhf = scf.ROHF(mol)

myhf.run()

# Extract MO_coeff and integrals. Then determine initial qubit 1-norm.
C_nonloc = np.copy(myhf.mo_coeff)

constant = float(mol.energy_nuc())


print('---------CANONICAL_ORBITALS---------')

C_copy = np.copy(C_nonloc)
t8 = time.time()
one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_copy[:,:ntot], threshold)
print("calculating integrals took",time.time()-t8)

if consider_cas:
    t10 = time.time()
    CASconstant, one_body_integrals, two_body_integrals =\
    md.get_active_space_integrals(one_body_integrals,
                                  two_body_integrals,
                                  occupied_indices,
                                  active_indices)
    print("transforming integrals to active space ints took", time.time()-t10)
else:
    CASconstant = 0
t4 = time.time()
qub1norm = JW1norm_spatfast(constant+CASconstant,
                            one_body_integrals,
                            two_body_integrals)
print('\ncalculating norm of qubit hamiltonian took', time.time()-t4)

print('\n')

print('---------LOCALIZED_ORBITALS---------')
localizemethod = ['Pipek-Mezey','Boys','ibo','ER'][0]

C = np.copy(C_nonloc)
if localizemethod == 'Pipek-Mezey':
    orb = lo.PipekMezey(mol).kernel(C[:,ncore:ntot])
elif localizemethod == 'Boys':
    orb = lo.Boys(mol).kernel(C[:,ncore:ntot])
elif localizemethod == 'ibo':
    orb = lo.ibo.ibo(mol, C[:,ncore:ntot])
elif localizemethod == 'ER':
    orb = lo.EdmistonRuedenberg(mol).kernel(C[:,ncore:ntot])
else:
    raise ValueError('Localization method not recognized')

C_locPM = np.hstack((C_nonloc[:,:ncore],orb,C_nonloc[:,ntot:nmo]))
one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_locPM[:,:ntot], threshold)
if consider_cas:
    CASconstant_PM, one_body_integrals, two_body_integrals =\
    md.get_active_space_integrals(one_body_integrals,
                                  two_body_integrals,
                                  occupied_indices,
                                  active_indices)
else:
    CASconstant_PM = 0
t4 = time.time()
qub1norm_loc = JW1norm_spatfast(constant+CASconstant,
                            one_body_integrals,
                            two_body_integrals)
print('calculating norm of qubit hamiltonian took', time.time()-t4)

if consider_cas:
    print("Considering 1-norm of active space Hamiltonian")
else:
    print("Considering 1-norm of full space Hamiltonian")
    
print("1-norm for CMOs is:",qub1norm)
print("1norm for LMOs is:",qub1norm_loc)


OPT_OO_MAX_ITER  = 5000 # Maximum number of steps for the OO

# Build the initial parameters
Rot_param_values = []
bounds = (-1.,1.)

bounds = []

# Determine starting parameters
for q in range(nmo-1):
    for p in range(q+1,nmo):
        if ( p in active_indices and q in active_indices ) or\
        ( (p in occupied_indices and q in occupied_indices) and optimize_occ ):
            Rot_param_values += [1e-1 * (random()-0.5) if randomize else 0.] # <== Set starting Rot_param_values here.
            bounds += [ [-1., 1.] ] 

print("amount of parameters:", len(Rot_param_values))
# Building the bounds for the rotational parameter amplitudes in the form of constraints
# (simply beacause cobyla doesn't have intrinsic bound options)
cons = md.constraints(bounds)

#%%======================================================|
# Run minimization
#========================================================|
# Define the cost function
def Cost_function_OO_OneNorm(Rot_param_values, verbose=False):
    """
    Cost function to minimize the One-Norm using MO rotations.
    """
    t6 = time.time()
    # Define K matrix
    t5 = time.time()
    K = md.K_matr(Rot_param_values, nmo, active_indices,
                  occupied_indices, optimize_occ)
    if verbose: print("\nInitializing K-matrix took:",time.time()-t5)
    # Exponentiate K matrix
    t1 = time.time()
    U_OO   = scipy.linalg.expm( - K )
    if verbose: print("exponentiating matrix took:",time.time()-t1)
    # Extract MO_coeff
    t3 = time.time()
    if localize:
        C_MO   = C_locPM @ U_OO
    else:
        C_MO   = C_nonloc @ U_OO
    if verbose: print("Matrix multiplication took:",time.time()-t3)
    # Extract integrals
    t2 = time.time()
    one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_MO[:,:ntot], threshold)
    if verbose: print("extracting integrals took:",time.time()-t2)
    t4 = time.time()
    if consider_cas:
        CASconstant, one_body_integrals, two_body_integrals =\
        md.get_active_space_integrals(one_body_integrals,
                                      two_body_integrals,
                                      occupied_indices,
                                      active_indices)
    else:
        CASconstant = 0
    # Compute 1-norm
    t4 = time.time()
    OneNorm = JW1norm_spatfast(constant+CASconstant,
                               one_body_integrals,
                               two_body_integrals)
    if verbose: print('calculating norm of qubit hamiltonian took', time.time()-t4)
    print('1-norm is:',OneNorm)
    if verbose: print("total time for 1 cost function evaluation is", time.time()-t6)
    return OneNorm

print("starting 1-norm nonloc is:",qub1norm)
print("1norm locPM is:",qub1norm_loc)

verbose = 1
t7 = time.time()
f_min_OO = minimize( Cost_function_OO_OneNorm,
                      x0      = Rot_param_values,
                      args    = verbose,
                      constraints=cons,
                      method  = 'cobyla',
                      options = {'maxiter': OPT_OO_MAX_ITER,
                                'tol'    : 1e-5,
                                'disp': True}  )
print("total time for minimization with",OPT_OO_MAX_ITER,"max iterations was:", time.time()-t7)
print("message:",f_min_OO.message,"number of function evaluations:",f_min_OO.nfev)

#%%======================================================|
# Summarize Results
#========================================================|
# Summarize results
K = md.K_matr(f_min_OO.x, nmo, active_indices,
              occupied_indices, optimize_occ) 
U_OO   = scipy.linalg.expm( - K )
if localize:
    C_OO = C_locPM @ U_OO
else:
    C_OO = C_nonloc @ U_OO

one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_OO[:,:ntot], threshold)
if consider_cas:
        CASconstant, one_body_integrals, two_body_integrals =\
        md.get_active_space_integrals(one_body_integrals,
                                      two_body_integrals,
                                      occupied_indices,
                                      active_indices)
else:
        CASconstant = 0
OneNormorbOO = JW1norm_spatfast(constant+CASconstant,
                            one_body_integrals,
                            two_body_integrals)

one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_nonloc[:,:ntot], threshold)
if consider_cas:
        CASconstant, one_body_integrals, two_body_integrals =\
        md.get_active_space_integrals(one_body_integrals,
                                      two_body_integrals,
                                      occupied_indices,
                                      active_indices)
else:
        CASconstant = 0
OneNormorbnonloc = JW1norm_spatfast(constant+CASconstant,
                            one_body_integrals,
                            two_body_integrals)

one_body_integrals, two_body_integrals = md.compute_integrals(
        mol, myhf, C_locPM[:,:ntot], threshold)
if consider_cas:
        CASconstant, one_body_integrals, two_body_integrals =\
        md.get_active_space_integrals(one_body_integrals,
                                      two_body_integrals,
                                      occupied_indices,
                                      active_indices)
else:
        CASconstant = 0
OneNormorblocPM = JW1norm_spatfast(constant+CASconstant,
                            one_body_integrals,
                            two_body_integrals)
if randomize:
    print("Starting from random orbital rotation...")
if consider_cas:
    print("Considering CAS(" + str(n_electrons) + "," + str(n_orbitals) +\
          "), \n1norm of CMOs is", OneNormorbnonloc, "\n1norm of LMOs is",
          OneNormorblocPM, "\nFinal 1norm of optimizer is", OneNormorbOO)
else:
    print("Considering full space, \n1norm of CMOs is", OneNormorbnonloc,
          "\n1norm of LMOs is", OneNormorblocPM,"\nFinal 1norm of optimizer is",
          OneNormorbOO)
print("number of function evaluations:",f_min_OO.nfev)

#%%======================================================|
# Optional: Extract Cube files for visualization of MOs
#========================================================|
"""
# Find description of molecule
if H_chain:
    if consider_cas:
        description = 'H' + str(N_atoms) + str(basis) + 'ne' + str(n_electrons) +\
            'no' + str(n_orbitals)
    else:
        description = 'H' + str(N_atoms) + str(basis)
    
elif H_ring:
    if consider_cas:
        description = 'H4' + str(basis) + 'ne' + str(n_electrons) + 'no' + str(n_orbitals)
    else:
        description = 'H4' + str(basis)

else:
    if consider_cas:
        description = fname.replace('xyz_files/','').replace('.txt','') + str(basis)\
            + 'ne' + str(n_electrons) + 'no' + str(n_orbitals)
    else:
        description = fname.replace('xyz_files/','').replace('.txt','') + str(basis)

# Code to extract Orbital optimized CUBE files.
# The file looks like: cwd/CUBE_FILES/pyscfcube{description}{localizemethod}
# {localized}{randomized}{consider_cas}{MO index}, to differentiate different MOs.

# Code for orbital optimized MOs:
Cost_function_OO_OneNorm(f_min_OO.x)
K = md.K_matr(f_min_OO.x, nmo, active_indices,
              occupied_indices, optimize_occ) 
U_OO   = scipy.linalg.expm( - K )
if localize:
    C_OO = C_locPM @ U_OO
else:
    C_OO = C_nonloc @ U_OO
t13 = time.time()
for i in range(ncore,ntot):
    tools.cubegen.orbital(mol, os.getcwd() + '/CUBE_FILES/pyscfcube'\
                          + description + 'onenorm_orb' + localizemethod + str(localize)\
                          + str(randomize) + str(consider_cas) + str(i) , C_OO[:,i])
print('Cube files of molecule', description,'created in', os.getcwd() + '/CUBE_FILES/')
print('extracting cube files took', time.time()-t13) 


# If you want to extract CMOs or LMOs:
t13 = time.time()
for i in range(ncore,ntot):
    tools.cubegen.orbital(mol, os.getcwd() + '/CUBE_FILES/pyscfcube'\
                          + description + localizemethod + str(0)\
                          + str(randomize) + str(consider_cas) + str(i) , C_nonloc[:,i])
print('Cube files of molecule', description,'created in', os.getcwd() + '/CUBE_FILES/')
print('extracting cube files took', time.time()-t13)

t13 = time.time()
for i in range(ncore,ntot):
    tools.cubegen.orbital(mol, os.getcwd() + '/CUBE_FILES/pyscfcube'\
                          + description + localizemethod + str(1)\
                          + str(randomize) + str(consider_cas) + str(i) , C_locPM[:,i])
print('Cube files of molecule', description,'created in', os.getcwd() + '/CUBE_FILES/')
print('extracting cube files took', time.time()-t13)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:58:26 2020

@author: emielkoridon
"""
# from __future__ import absolute_import

from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo, ci, cc, fci, mp, lo, tools
import os
import time

# from openfermion import MolecularData
# from openfermionpyscf import run_pyscf, PyscfMolecularData

# from openfermion.transforms import get_fermion_operator, get_sparse_operator, \
#     jordan_wigner, bravyi_kitaev

def Visualize_MO(filename, XYZ_geo, Natom):
    '''
    Returns the ISO surface plot of the molecular orbital contained in a “.cube” file.
    '''
    import py3Dmol
    import os
    file = open(os.getcwd() + '/CUBE_FILES/' + filename).read()
    p = py3Dmol.view(width=600,height=600)
    p.addModel("{} \n\n ".format(Natom) + XYZ_geo,'xyz')
    p.addVolumetricData(file , "cube", {'isoval': +0.1, 'color': "red", 'opacity': 0.9})
    p.addVolumetricData(file , "cube", {'isoval': -0.1, 'color': "blue", 'opacity': 0.9})
    #p.addVolumetricData(file , “cube”, {‘isoval’: -0.1, ‘color’: “blue”, ‘opacity’: 0.9})
    p.setStyle({'sphere':{'scale':'0.15'}, 'stick':{'radius':'0.05'}})
    p.zoom(5)
    p.show()
    del p
    
def xyz_to_geom(fname):
    geometry = []
    #Natom = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            gotx = False
            goty = False 
            gotz = False
            gotatom = False
            for j in range(len(l)):
                gotone = False
                char = l[j]
                if char != ' ' and char.isalpha() and not gotatom: 
                    for n in range(j, len(l)):
                        if l[n]==' ' and gotatom == False:
                            atom = str(l[j:n])
                            gotatom = True
                if char != ' ' and (char.isdigit() or char=='-') and not gotx:
                    for k in range(j, len(l)):
                        if l[k]==' ' and gotx == False:
                            x = float(l[j:k])
                            gotx = True
                            endx = k
                            gotone = True
                if gotx and (char.isdigit() or char=='-') and not goty and not gotone and j>endx:
                    for m in range(j, len(l)):
                        if l[m]==' ' and goty == False:
                            #print('x,endx, j, l[j]',x, endx, j, l[j])
                            y = float(l[j:m])
                            goty = True
                            endy = m
                            gotone = True
                if gotx and goty and (char.isdigit() or char=='-') and not gotz and not gotone and j>endy:
                    #print('x,y,endx,endy,j,l[j]',x,y, endx, endy, j, l[j])
                    z = float(l[j:-1])
                    gotz == True
                    #print('z',z, 'j', j, 'i',i)
                    break
            geometry.append(tuple((atom,(x,y,z))))
    return geometry
    #     print(line)
    

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals

def count_terms_ferm(fermion_hamiltonian, thresholdferm):
    """
    Count terms in a FermionOperator object
    """
    ntermsferm = 0 
    for i, j in enumerate(fermion_hamiltonian):
        norm = j.induced_norm()
        #print('i=',i,'j=',j)#,'inducednorm =', norm)
        if abs(norm) > thresholdferm:
            ntermsferm += 1
            #print('norm is', norm, 'so we count the term!')
        #else:
            #print('norm is', norm, 'so we do not count the term.')
    return ntermsferm

def count_terms_qub(qubit_hamiltonian, thresholdqub):
    """
    Count terms in a QubitOperator object
    """
    ntermsqub = 0
    for i, j in enumerate(qubit_hamiltonian):
        norm = j.induced_norm()
        #print('i=',i,'j=',j)#,'inducednorm =', norm)
        if abs(norm) > thresholdqub:
            ntermsqub += 1
            #print('norm is', norm, 'so we count the term!')
        #else:
            #print('norm is', norm, 'so we do not count the term.')
    
    return ntermsqub

def compute_1norm(hamiltonian):
    """
    
    Parameters
    ----------
    hamiltonian : OpenFermion Hamiltonian object

    Returns
    -------
    One norm of Hamiltonian

    """
    norm = 0.
    for i, j in enumerate(hamiltonian):
        norm += abs(j.induced_norm())
    return norm

def count_ao(geometry, basis, spin=0):
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.spin = spin
    mol.symmetry = False
    mol.build()
    
    return mol.nao

def count_elec(geometry, basis, spin=0):
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.spin = spin
    mol.symmetry = False
    mol.build()
    
    return mol.nelectron

def plot_th(thresholds, energies, nterms, ntermsHq, ntermsHf, 
            energies_loc, nterms_loc, ntermsHq_loc, ntermsHf_loc,
            energies_locvirt, nterms_locvirt, ntermsHq_locvirt, ntermsHf_locvirt,
            HF_energy, FCI_energy, ccsdenergy, ylims, title, zoom=False):
    """
    Plot thresholds vs energies    
    """
        
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4,ncols=1, figsize=(15,11))
    fig.suptitle(title)
    ax1.plot(thresholds,energies, label='Non-localized')
    ax1.plot(thresholds,energies_loc, label='Localized')
    ax1.plot(thresholds,energies_locvirt, label='Localized with virt')
    
    if not zoom:
        ax1.plot(thresholds, HF_energy, 'm-', label='HF-energy')
    else:    
        ax1.plot(thresholds, ccsdenergy, 'c-', label='CCSD-energy')
    
    ax1.plot(thresholds, FCI_energy, 'r-', label='FCI-energy')
    ax1.set_xscale('symlog',linthreshx=1e-16)
    ax1.grid()
    ax1.set_ylim(ylims)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('FCI Energy (a.u.)')
    ax1.legend()
    
    ax2.plot(thresholds, nterms, label='Non-localized')
    ax2.plot(thresholds, nterms_loc, label='Localized')
    ax2.plot(thresholds, nterms_locvirt, label='Localized with virt')
    #    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Terms in 2-body tensor')
    ax2.set_xscale('symlog',linthreshx=1e-16)
    ax2.grid()
    ax2.legend()
    
    ax3.plot(thresholds, ntermsHq, label='Non-localized')
    ax3.plot(thresholds, ntermsHq_loc, label='Localized')
    ax3.plot(thresholds, ntermsHq_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax3.set_ylabel('Number of Terms in H_qub')
    ax3.set_xlabel('Threshold')
    ax3.set_xscale('symlog',linthreshx=1e-16)
    ax3.grid()
    ax3.legend()
    
    ax4.plot(thresholds, ntermsHf, label='Non-localized')
    ax4.plot(thresholds, ntermsHf_loc, label='Localized')
    ax4.plot(thresholds, ntermsHf_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax4.set_ylabel('Number of Terms in H_ferm')
    ax4.set_xlabel('Threshold')
    ax4.set_xscale('symlog',linthreshx=1e-16)
    ax4.grid()
    ax4.legend()
    plt.show()
    
def plot_eri(oneD_mo_ints, oneD_mo_ints_loc, nmo, alpha):
    x = np.arange(nmo**4)

    ax = plt.subplot(111)
    ax.plot(x, oneD_mo_ints, color='r', alpha=alpha)
    ax.plot(x, oneD_mo_ints_loc, color='b', alpha=alpha)
    
    plt.show()

def plot_H4(angles, hf_energies, fci_energies, ccsd_energies, 
            nterms, ntermsHf, ntermsHq,title, ylims=None):
    
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4,ncols=1, figsize=(15,11))
    fig.suptitle(title)
    #ax1.plot(angles,energies, label='Non-localized')
    #ax1.plot(angles,energies_loc, label='Localized')
    #ax1.plot(angles,energies_locvirt, label='Localized with virt')
    ax1.plot(angles, hf_energies, 'm-', label='HF-energy')
    ax1.plot(angles, fci_energies, 'r-', label='FCI-energy')
    ax1.plot(angles, ccsd_energies, 'c-', label='CCSD-energy')
    #ax1.set_xscale('symlog',linthreshx=1e-16)
    ax1.grid()
    if not ylims is None:
        ax1.set_ylim(ylims)
    ax1.set_xlabel('angle')
    ax1.set_ylabel('FCI Energy (a.u.)')
    ax1.legend()
    
    ax2.plot(angles, nterms)#, label='Non-localized')
    #ax2.plot(angles, nterms_loc, label='Localized')
    #ax2.plot(angles, nterms_locvirt, label='Localized with virt')
    #    ax2.set_xlabel('angle')
    ax2.set_ylabel('Terms in 2-body tensor')
    #ax2.set_xscale('symlog',linthreshx=1e-16)
    ax2.grid()
    #ax2.legend()
    
    ax3.plot(angles, ntermsHq)#, label='Non-localized')
    #ax3.plot(angles, ntermsHq_loc, label='Localized')
    #ax3.plot(angles, ntermsHq_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('angle log')
    ax3.set_ylabel('Number of Terms in H_qub')
    ax3.set_xlabel('angle')
    #ax3.set_xscale('symlog',linthreshx=1e-16)
    ax3.grid()
    #ax3.legend()
    
    ax4.plot(angles, ntermsHf)#, label='Non-localized')
    #ax4.plot(angles, ntermsHf_loc, label='Localized')
    #ax4.plot(angles, ntermsHf_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('angle log')
    ax4.set_ylabel('Number of Terms in H_ferm')
    ax4.set_xlabel('angle')
    #ax4.set_xscale('symlog',linthreshx=1e-16)
    ax4.grid()
    #ax4.legend()
    plt.show()
    

def plot_terms(thresholds, nterms, ntermsHq, ntermsHf, nterms_loc, ntermsHq_loc,
               ntermsHf_loc, nterms_locvirt, ntermsHq_locvirt, ntermsHf_locvirt,title):
    """
    Plot thresholds vs terms   
    """
        
    fig, [ax2, ax4, ax3] = plt.subplots(nrows=3,ncols=1, figsize=(15,11))
    fig.suptitle(title)
    
    ax2.plot(thresholds, nterms, label='Non-localized')
    ax2.plot(thresholds, nterms_loc, label='Localized')
    ax2.plot(thresholds, nterms_locvirt, label='Localized with virt')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Terms in 2-body tensor')
    ax2.set_xscale('symlog',linthreshx=1e-16)
    ax2.grid()
    ax2.legend()
    
    ax3.plot(thresholds, ntermsHq, label='Non-localized')
    ax3.plot(thresholds, ntermsHq_loc, label='Localized')
    ax3.plot(thresholds, ntermsHq_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax3.set_ylabel('Number of Terms in H_qub')
    ax3.set_xlabel('Threshold')
    ax3.set_xscale('symlog',linthreshx=1e-16)
    ax3.grid()
    ax3.legend()
    
    ax4.plot(thresholds, ntermsHf, label='Non-localized')
    ax4.plot(thresholds, ntermsHf_loc, label='Localized')
    ax4.plot(thresholds, ntermsHf_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax4.set_ylabel('Number of Terms in H_ferm')
    ax4.set_xlabel('Threshold')
    ax4.set_xscale('symlog',linthreshx=1e-16)
    ax4.grid()
    ax4.legend()
    plt.show()

def plot_cas(thresholds, energies, nterms, ntermsHq, ntermsHf, 
            energies_loc, nterms_loc, ntermsHq_loc, ntermsHf_loc,
            energies_locvirt, nterms_locvirt, ntermsHq_locvirt, ntermsHf_locvirt,
            HF_energy, FCI_energy, ccsdenergy, ylims, title, zoom=False):
    """
    Plot thresholds vs energies    
    """
        
    fig, [ax1, ax2, ax4, ax3] = plt.subplots(nrows=4,ncols=1, figsize=(15,11))
    fig.suptitle(title)
    ax1.plot(thresholds,energies, label='Non-localized')
    ax1.plot(thresholds,energies_loc, label='Localized')
    ax1.plot(thresholds,energies_locvirt, label='Localized with virt')
    
    if not zoom:
        ax1.plot(thresholds, HF_energy, 'm-', label='HF-energy')
    else:    
        ax1.plot(thresholds, ccsdenergy, 'c-', label='CCSD-energy')
    
    ax1.plot(thresholds, FCI_energy, 'r-', label='FCI-energy')
    ax1.set_xscale('symlog',linthreshx=1e-16)
    ax1.grid()
    ax1.set_ylim(ylims)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('CASCI energy (a.u.)')
    ax1.legend()
    
    ax2.plot(thresholds, nterms, label='Non-localized')
    ax2.plot(thresholds, nterms_loc, label='Localized')
    ax2.plot(thresholds, nterms_locvirt, label='Localized with virt')
    #    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Terms in full 2-body tensor')
    ax2.set_xscale('symlog',linthreshx=1e-16)
    ax2.grid()
    ax2.legend()
    
    ax3.plot(thresholds, ntermsHq, label='Non-localized')
    ax3.plot(thresholds, ntermsHq_loc, label='Localized')
    ax3.plot(thresholds, ntermsHq_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax3.set_ylabel('Number of Terms in H_qub')
    ax3.set_xlabel('Threshold')
    ax3.set_xscale('symlog',linthreshx=1e-16)
    ax3.grid()
    ax3.legend()
    
    ax4.plot(thresholds, ntermsHf, label='Non-localized')
    ax4.plot(thresholds, ntermsHf_loc, label='Localized')
    ax4.plot(thresholds, ntermsHf_locvirt, label='Localized with virt')
    #    ax3.set_xlabel('Threshold log')
    ax4.set_ylabel('Number of Terms in H_ferm')
    ax4.set_xlabel('Threshold')
    ax4.set_xscale('symlog',linthreshx=1e-16)
    ax4.grid()
    ax4.legend()
    plt.show()
    

def prod_out(M1, M2):
    n=[M1.shape[0],M1.shape[1],M2.shape[0],M2.shape[1]]
    
    R=np.zeros(n)
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                for l in range(n[3]):
                    R[i,j,k,l]=M1[i,j]*M2[k,l]
    return R

# def multiprocess_func:
def cholesky(G, nao, threshold, realMO=True, verbose=1):
    """
    
    Parameters
    ----------
    G : numpy.array
        two-electron integrals.
    nao : numpy.array
        number of AOs/MOs.
    threshold : int
        convergence threshold.
    realMO : bool, optional
        Real (True) or complex (False) MOs. The default is True.
    verbose : int, optional
        verbosity (0, 1 or 2). The default is 1.

    Returns
    -------
    Cholesky matrix L

    """
    #starting with cholesky decomposition
    D=np.zeros((nao,nao))
    for i in range(nao):
        for j in range(nao):
            D[i,j]=G[i,j,i,j]
            # print(G[i,j,i,j])
    i_max=np.zeros((2),dtype=np.int)
    i_help=np.where(np.max(abs(D))==abs(D))
    try:
        i_max[0]=i_help[0][0][0]
        i_max[1]=i_help[0][0][1]
    except:
        i_max[0]=i_help[0][0]
        i_max[1]=i_help[1][0]
    D_max=D[i_max[0],i_max[1]]
    if verbose == 1:
        print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
    if verbose == 2:
        print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
        print(D) 
    
    m=0
    Res=G
    while abs(D_max)>threshold:
        if verbose == 1 or verbose == 2:
            print('-------- iteration m = '+str(m)+'-------')
        
        #t1 = time.time()
        L_h=Res[:,:,i_max[0],i_max[1]]/np.sqrt(np.abs(D_max))
        L_sign=np.sign(D_max)
        #print('Computing L_h took', time.time()-t1)
        t2 = time.time()
        G_h=np.zeros((nao,nao,nao,nao)) 
        if m==0:
            L=L_h
            Ls=L_sign
            G_h=G_h+Ls*np.einsum('ij,kl->ijkl', L[ :, :], L[ :, :])
        else:
            L=np.dstack((L,L_h))  
            Ls=np.hstack((Ls,L_sign))
            for i in range(m+1):
                if realMO:
                    G_h=G_h+np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
                    #G_h=G_h+prod_out(L[ :, :, i], L[ :, :, i])
                    # print("DIFFERENCE PROD_OUT AND EINSUM:", \
                    #       np.max(prod_out(L[ :, :, i], L[ :, :, i])-\
                    #       np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])))
                else:
                    G_h=G_h+Ls[i]*np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
                    # G_h=G_h+Ls[i]*prod_out(L[ :, :, i], L[ :, :, i])
                    
        if verbose == 1 or verbose == 2:
            print('computing G_h took', time.time() - t2)
        #t3 = time.time()
        Res=G-G_h
        err_h=np.sum(np.diag(np.abs(Res.reshape(Res.shape[0]**2,Res.shape[1]**2))))
        # err_h = np.sum(np.abs(Res))
        #print('computing residue and error took', time.time()-t3)
        #t4 = time.time()
        for i in range(nao):
            for j in range(nao):
                D[i,j]=Res[i,j,i,j]
                
        m=m+1
        i_help=np.where(np.max(abs(D))==abs(D))
        try:
            i_max[0]=i_help[0][0][0]
            i_max[1]=i_help[0][0][1]
        except:
            i_max[0]=i_help[0][0]
            i_max[1]=i_help[1][0]
        D_max=D[i_max[0],i_max[1]]
        if verbose == 2:
            print(D)
        if verbose == 1 or verbose == 2:
            print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
            print('current error = '+str(err_h))
        #print('computing new D and D_max took', time.time() - t4)
        if m>500: 
            break
    if verbose == 1 or verbose == 2:   
        print('_________________ decomposition done ___________________________')
    return(L)
# def cholesky(G, nao, threshold, realMO=True, verbose=1):
#     """
    
#     Parameters
#     ----------
#     G : numpy.array
#         two-electron integrals.
#     nao : numpy.array
#         number of AOs/MOs.
#     threshold : int
#         convergence threshold.
#     realMO : bool, optional
#         Real (True) or complex (False) MOs. The default is True.
#     verbose : int, optional
#         verbosity (0, 1 or 2). The default is 1.

#     Returns
#     -------
#     Cholesky matrix L

#     """
#     #starting with cholesky decomposition
#     D=np.zeros((nao,nao))
#     for i in range(nao):
#         for j in range(nao):
#             D[i,j]=G[i,j,i,j]
            
#     i_max=np.zeros((2),dtype=np.int)
#     i_help=np.where(np.max(abs(D))==abs(D))
#     try:
#         i_max[0]=i_help[0][0][0]
#         i_max[1]=i_help[0][0][1]
#     except:
#         i_max[0]=i_help[0][0]
#         i_max[1]=i_help[1][0]
#     D_max=D[i_max[0],i_max[1]]
#     if verbose == 1:
#         print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
#     if verbose == 2:
#         print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
#         print(D) 
    
#     m=0
#     Res=G
#     while abs(D_max)>threshold:
#         if verbose == 1 or verbose == 2:
#             print('-------- iteration m = '+str(m)+'-------')
        
#         #t1 = time.time()
#         L_h=Res[:,:,i_max[0],i_max[1]]/np.sqrt(np.abs(D_max))
#         L_sign=np.sign(D_max)
#         #print('Computing L_h took', time.time()-t1)
#         t2 = time.time()
#         G_h=np.zeros((nao,nao,nao,nao)) 
#         if m==0:
#             L=L_h
#             Ls=L_sign
#             G_h=G_h+Ls*prod_out(L[ :, :], L[ :, :])
#         else:
#             L=np.dstack((L,L_h))  
#             Ls=np.hstack((Ls,L_sign))
#             for i in range(m+1):
#                 if realMO:
#                     G_h=G_h+np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
#                     #G_h=G_h+prod_out(L[ :, :, i], L[ :, :, i])
#                     # print("DIFFERENCE PROD_OUT AND EINSUM:", \
#                     #       np.max(prod_out(L[ :, :, i], L[ :, :, i])-\
#                     #       np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])))
#                 else:
#                     G_h=G_h+Ls[i]*np.einsum('ij,kl->ijkl', L[ :, :, i], L[ :, :, i])
#                     # G_h=G_h+Ls[i]*prod_out(L[ :, :, i], L[ :, :, i])
                    
#         if verbose == 1 or verbose == 2:
#             print('computing G_h took', time.time() - t2)
#         #t3 = time.time()
#         Res=G-G_h
#         err_h=np.sum(np.diag(np.abs(Res.reshape(Res.shape[0]**2,Res.shape[1]**2))))
#         # err_h = np.sum(np.abs(Res))
#         #print('computing residue and error took', time.time()-t3)
#         #t4 = time.time()
#         for i in range(nao):
#             for j in range(nao):
#                 D[i,j]=Res[i,j,i,j]
                
#         m=m+1
#         i_help=np.where(np.max(abs(D))==abs(D))
#         try:
#             i_max[0]=i_help[0][0][0]
#             i_max[1]=i_help[0][0][1]
#         except:
#             i_max[0]=i_help[0][0]
#             i_max[1]=i_help[1][0]
#         D_max=D[i_max[0],i_max[1]]
#         if verbose == 2:
#             print(D)
#         if verbose == 1 or verbose == 2:
#             print('ind = '+str(i_max[0])+' '+str(i_max[1])+'  D_max='+str(D_max))
#             print('current error = '+str(err_h))
#         #print('computing new D and D_max took', time.time() - t4)
#         if m>500: 
#             break
#     if verbose == 1 or verbose == 2:   
#         print('_________________ decomposition done ___________________________')
#     return(L)

def save_data(description, thresholds, energies, nterms, ntermsHf, normsHf, ntermsHq, normsHq,
             energies_loc, nterms_loc, ntermsHf_loc, normsHf_loc, ntermsHq_loc,
             normsHq_loc, energies_locvirt, nterms_locvirt, ntermsHf_locvirt,
             normsHf_locvirt, ntermsHq_locvirt, normsHq_locvirt, references):
    datadir = os.getcwd() + '/Saved_data/NOofterms/data' + description
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    with open(datadir + '/energies.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in energies)
    
    with open(datadir + '/energies_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in energies_loc)
        
    with open(datadir + '/energies_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in energies_locvirt)
        
    with open(datadir + '/nterms.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in nterms)
          
    with open(datadir + '/nterms_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in nterms_loc)
          
    with open(datadir + '/nterms_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in nterms_locvirt)
        
    with open(datadir + '/ntermsHq.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHq)
        
    with open(datadir + '/ntermsHq_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHq_loc)
        
    with open(datadir + '/ntermsHq_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHq_locvirt)
    
    with open(datadir + '/ntermsHf.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHf)
    
    with open(datadir + '/ntermsHf_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHf_loc)
        
    with open(datadir + '/ntermsHf_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in ntermsHf_locvirt)
    
    with open(datadir + '/normsHf.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHf)
    
    with open(datadir + '/normsHf_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHf_loc)
        
    with open(datadir + '/normsHf_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHf_locvirt)
        
    with open(datadir + '/normsHq.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHq)
    
    with open(datadir + '/normsHq_loc.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHq_loc)
       
    with open(datadir + '/normsHq_locvirt.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in normsHq_locvirt)
                   
    with open(datadir + '/thresholds.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in thresholds)

    with open(datadir + '/references.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % i for i in references)

def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)

def loc(myhf,
        mol,
        localize_virt,
        localizemethod,
        Orth_AO,
        localize_cas,
        n_orbitals,
        n_electrons,
        localize_sep,
        verbose):
    t0 = time.time()
    C_nonloc = np.copy(myhf.mo_coeff)
    nmo = C_nonloc.shape[1] # Number of MOs
    
    
    if Orth_AO:
        t5 = time.time()
        S = myhf.get_ovlp()
        S_eigval, S_eigvec = np.linalg.eigh(S)
        S_sqrt_inv = S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ np.linalg.inv(S_eigvec)
        C = S_sqrt_inv
        if verbose: print("Computing inverse overlap took", time.time()-t5)
    else:

        ndocc = np.count_nonzero(myhf.mo_occ) # Number of (doubly) occupied MOs
        if localize_cas:
            # Localize MOs in active space
            n_core_orbitals = (mol.nelectron - n_electrons) // 2
            if localize_virt:
                ntot = n_core_orbitals + n_orbitals
            else:
                ntot = ndocc
                localize_sep = 1
        else:
            # Localize every MO
            n_core_orbitals = 0
            if localize_virt:
                ntot = len(myhf.mo_occ)
            else:
                localize_sep = 1
                ntot = ndocc

        C = C_nonloc

        if localize_sep:
            print ('LOCALIZING SEPERATELY')
            if localizemethod == 'Pipek-Mezey':
                orb = lo.PipekMezey(mol).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.PipekMezey(mol).kernel(C[:,ndocc:ntot])
            elif localizemethod == 'Boys':
                orb = lo.Boys(mol).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.Boys(mol).kernel(C[:,ndocc:ntot])
            elif localizemethod == 'ibo':
                orb = lo.ibo.ibo(mol, C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.ibo.ibo(mol, C[:,ndocc:ntot])
            elif localizemethod == 'ER':
                orb = lo.EdmistonRuedenberg(mol).kernel(C[:,n_core_orbitals:ndocc])
                if localize_virt:
                    orbvirt = lo.EdmistonRuedenberg(mol).kernel(C[:,ndocc:ntot])
            else:
                print('Localization method not recognized')
                raise
        else:
            print('LOCALIZING TOGETHER','n_core',n_core_orbitals,'ntot',ntot)
            if localizemethod == 'Pipek-Mezey':
                orb = lo.PipekMezey(mol).kernel(C[:,n_core_orbitals:ntot])
            elif localizemethod == 'Boys':
                orb = lo.Boys(mol).kernel(C[:,n_core_orbitals:ntot])
            elif localizemethod == 'ibo':
                orb = lo.ibo.ibo(mol, C[:,n_core_orbitals:ntot])
            elif localizemethod == 'ER':
                orb = lo.EdmistonRuedenberg(mol).kernel(C[:,n_core_orbitals:ntot])
            else:
                print('Localization method not recognized')
                raise


        if localize_virt:
            if localize_sep:
                #print('putting C together separately')
                if localize_cas:
                    C = np.hstack((C_nonloc[:,:n_core_orbitals],orb,orbvirt,C_nonloc[:,ntot:nmo]))
                else:
                    C = np.hstack((orb,orbvirt))
            else:
                #print('putting C together mixed')
                if localize_cas:
                    C = np.hstack((C_nonloc[:,:n_core_orbitals], orb, C_nonloc[:,ntot:nmo]))
                else:
                    C = orb
        else:
            #print('putting C together separately')
            if localize_cas:
                C = np.hstack((C_nonloc[:,:n_core_orbitals], orb, C_nonloc[:,ndocc:]))
            else:                
                C = np.hstack((orb,C_nonloc[:,ndocc:]))
        #print('CSHAPE =',C.shape)
        
        if verbose: print('core',n_core_orbitals,'ndocc',ndocc,'ntot',ntot,'nmo',nmo)
    if verbose: print('localizing took', time.time()-t0)
    
    return C

def gen_cube(myhf, mol, C, localize, localizemethod, description, verbose):
    t13 = time.time()
    ndocc = np.count_nonzero(myhf.mo_occ)
    nmo = len(myhf.mo_occ)
    print('ndocc', ndocc, 'nmo', nmo)
    for i in range(nmo):    
        tools.cubegen.orbital(mol, os.getcwd() + '/CUBE_FILES/pyscfcube'\
                              + description + localizemethod + str(localize)\
                              + str(i) , myhf.mo_coeff[:,i])
    print('Cube files of molecule', description,'created in', os.getcwd() + '/CUBE_FILES/')
    if verbose: print('extracting cube files took', time.time()-t13)
    
def get_active_space_integrals(one_body_integrals,
                               two_body_integrals,
                               occupied_indices=None,
                               active_indices=None):
        """Restricts a molecule at a spatial orbital level to an active space

        This active space may be defined by a list of active indices and
            doubly occupied indices. Note that one_body_integrals and
            two_body_integrals must be defined
            n an orthonormal basis set.

        Args:
            occupied_indices: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            active_indices: A list of spatial orbital indices indicating
                which orbitals should be considered active.

        Returns:
            tuple: Tuple with the following entries:

            **core_constant**: Adjustment to constant shift in Hamiltonian
            from integrating out core orbitals

            **one_body_integrals_new**: one-electron integrals over active
            space.

            **two_body_integrals_new**: two-electron integrals over active
            space.
        """
        # Fix data type for a few edge cases
        occupied_indices = [] if occupied_indices is None else occupied_indices
        if (len(active_indices) < 1):
            raise ValueError('Some active indices required for reduction.')

    

        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices:
            core_constant += 2 * one_body_integrals[i, i]
            for j in occupied_indices:
                core_constant += (2 * two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])

        # Modified one electron integrals
        one_body_integrals_new = np.copy(one_body_integrals)
        for u in active_indices:
            for v in active_indices:
                for i in occupied_indices:
                    one_body_integrals_new[u, v] += (
                        2 * two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])

        # Restrict integral ranges and change M appropriately
        return (core_constant,
                one_body_integrals_new[np.ix_(active_indices,
                                                 active_indices)],
                two_body_integrals[np.ix_(active_indices, active_indices,
                                             active_indices, active_indices)])

def spin_coeff(one_body_integrals, two_body_integrals, threshold=1e-10):
    n_qubits = 2 * one_body_integrals.shape[0]
    
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
    
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
    
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (
                                              two_body_integrals[p, q, r, s]
                                              / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (
                                              two_body_integrals[p, q, r, s]
                                              / 2.)
    
                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s + 1] = (
                                              two_body_integrals[p, q, r, s]
                                              / 2.)
    
    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < threshold] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < threshold] = 0.
    return one_body_coefficients, two_body_coefficients
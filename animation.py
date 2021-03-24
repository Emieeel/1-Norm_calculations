#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:29:29 2021

@author: emielkoridon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
n_mol = -1
basisname = ['sto-3g', 'cc-pvdz', 'def2svp', 'minbasis', 'fullbasis','def2tzvp.nw'][1]
fname = ['xyz_files/H2nosym.txt','xyz_files/H2COnosym.txt','xyz_files/H10.txt',\
         'xyz_files/C2.txt', 'xyz_files/LiH.txt', 'xyz_files/HLiO.txt',\
         'xyz_files/H2Onosym.txt', 'xyz_files/H14.txt',\
         'xyz_files/hnch2_s0min_dzp.txt', 'xyz_files/hnc3h6_s0min_dzp.txt',\
         'xyz_files/hnc5h10_s0min_dzp.txt', 'xyz_files/hnc7h14_s0min_dzp.txt',\
         'xyz_files/benzene.txt','xyz_files/PCy3.txt','xyz_files/PCy3Cl2Ru.txt',\
         'xyz_files/femoco.txt','xyz_files/M06-L.txt','xyz_files/Ru_SI_II.txt',
         'xyz_files/Ru_TS_II-III.txt','xyz_files/Ru_SI_V.txt','xyz_files/Ru_SI_VIII.txt',\
         'xyz_files/Ru_TS_VIII-IX.txt','xyz_files/Ru_SI_IX.txt','xyz_files/Ru_SI_XVIII.txt',
         'xyz_files/formaldimine_rot.txt'][n_mol]

description = fname.replace('xyz_files/','').replace('.txt','') + str(basisname)

fig = plt.figure(figsize=(30,20),constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
# ax1.get_xaxis().set_visible(False)
# ax1.get_yaxis().set_visible(False)
# ax2.get_xaxis().set_visible(False)
# ax2.get_yaxis().set_visible(False)
ax2.axis('off')
ax3.axis('off')
datasize = 32

datadir = os.getcwd() + '/Saved_data/jacobi_scan/' + description

with open(datadir + '/thetas.txt', 'r') as filehandle:
    thetas = np.array([float(i.rstrip()) for i in filehandle.readlines()])

with open(datadir + '/norms.txt', 'r') as filehandle:
    norms = np.array([float(i.rstrip()) for i in filehandle.readlines()])

# ax1.title = "$\\theta$ rotation between orbital " + str(orb1) + " and orbital " + str(orb2)
# ax1.plot(thetas,norms)
ax1.set_xlabel('$\\theta$',size=40)
ax1.set_ylabel('$\lambda_Q$ (a.u.)',size=40)
# ax1.set_title(title, fontsize=24)

# ax.legend()

# ax.legend(fontsize=20)
ax1.tick_params(axis='x',labelsize=20)
ax1.tick_params(axis='y',labelsize=20)


imgsl = []
imgsr = []
for i in range(1,datasize+2):
    imgsl.append(mpimg.imread(datadir + '/MO_plots/hnch2_orb1_' + str(i) + '.png'))
    imgsr.append(mpimg.imread(datadir + '/MO_plots/hnch2_orb2_' + str(i) + '.png'))
ims = []

for i in range(datasize+1):
    iml, = ax1.plot(thetas[:i],norms[:i], 'red')
    imm = ax2.imshow(imgsl[i])
    imr = ax3.imshow(imgsr[i])
    ims.append([iml, imm, imr])

ani = animation.ArtistAnimation(fig, ims, interval=200)#, blit=False)

# ani.save(description +"_gridspec.avi")
    



# x = np.array(list(range(5)))

# fig = plt.figure()
# p1 = fig.add_subplot(211)
# p2 = fig.add_subplot(212)

# p1.set_xlim([0,15])
# p1.set_ylim([0,100])

# # p2.set_xlim([0,15])
# # p2.set_ylim([0,100])

# # set up empty lines to be updates later on
# l1, = p1.plot([],[],'b')
# # l2, = p2.plot([],[],'r')

# def gen1():
#     i = 0.5
#     while(True):
#         yield i
#         i += 0.1

# # def gen2():
# #     j = 0
# #     while(True):
# #         yield j
# #         j += 1

# def run1(c):
#     y = c*x
#     l1.set_data(x,y)

# # def run2(c):
# #     y = c*x
# #     l2.set_data(x,y)

# ani1 = animation.FuncAnimation(fig,run1,gen1,interval=100)
# # ani2 = animation.FuncAnimation(fig,run2,gen2,interval=100)
plt.show()

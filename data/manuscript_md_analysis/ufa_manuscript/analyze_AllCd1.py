import numpy as np
import matplotlib.pyplot as pl
import mdtraj as md
import matplotlib as mpl
#from nglview.player import TrajectoryPlayer
import os
from Bio import pairwise2
import pandas
import itertools
from matplotlib import rc
from matplotlib import rcParams

#### THINGS YOU MIGHT REASONABLY NEED TO CHANGE ####
# If you have some odd trajectories, change numFrames
frames=200
# So I only had maxdyn set to 1 
max_dyn = 30
# Load in a key to pull out the distances
save_pair = pandas.read_csv('datfiles/cd1_pairLocate.csv')
# Pick out more resid pairs if wanted/needed
key_cd1_pairs = [[76,153],[73,154],[70,161],[69,161]]

# You can edit these at some point if you want to change the formatting of your figures
font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 20}
COLOR = 'black'
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = COLOR
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

rc('font', **font)

for Simdir in ['cd1d_apo_37_1','cd1d_apo_37_2','cd1d_apo_37_3','cd1d_apo_27_1','cd1d_apo_27_2','cd1d_apo_27_3']:
    topfile = Simdir+'/namd/step3_input.pdb'
    struct = md.load_pdb(topfile)
    table,bonds = struct.topology.to_dataframe()

    cd1_resids = '12 14 16 26 28 30 38 40 47 58 63 66 69 70 73 76 77 80 81 88 90 96 98 100 114 116 118 123 124 131 140 141 144 148 151 153 154 157 158 161 162 165 166 169 173'
    cd1_resname = 'CQSTGAHWVFWLIFYSFDVLLLVAFVFILWWVALDWTTVLLTCFL'

    # select out only the platform domain
    mdsel_full = struct.topology.select("chainid == 0 and resid 1 to 180 and name CA ")
    # select out only the Caitlin identified residues
    mdsel = struct.topology.select("chainid == 0 and sidechain and not name H and residue "+cd1_resids)

    first_frame = md.load_frame(topfile,0)
    rmsdpre = []
    rmsdFullpre = []
    pocket_dists_pre = []
    pocket_pairs = list(itertools.product(mdsel, mdsel))
    for dyn in np.arange(max_dyn):
        chunk=md.load(Simdir+'/namd/dyn'+str(dyn+1)+'.dcd',top=topfile)
        rmsdpre.append(md.rmsd(chunk, first_frame,atom_indices=mdsel))
        rmsdFullpre.append(md.rmsd(chunk, first_frame,atom_indices=mdsel_full))
        
        pocket_dists_pre.append(md.compute_distances(chunk, atom_pairs=pocket_pairs, periodic=False))
        
    cryst_dists = md.compute_distances(first_frame, atom_pairs=pocket_pairs, periodic=False)
    rmsd=np.reshape(rmsdpre,(max_dyn)*frames)
    rmsdFull=np.reshape(rmsdFullpre,(max_dyn)*frames)
    pocket_dists = np.reshape(pocket_dists_pre,[(max_dyn)*frames,np.shape(pocket_dists_pre)[2]])

    fig, ax = pl.subplots(1, 1,squeeze=False,figsize=(10,8))
    pl.plot(rmsd,linewidth=2.5)
    pl.plot(rmsdFull,linewidth=2.5)
    pl.xlabel('Steps')
    pl.ylabel('RMSD')
    pl.legend(['Pocket','Full'])
    pl.savefig(Simdir+'_rmsd.pdf',format='pdf')

    final_distance_df = []
    for i in np.arange(len(key_cd1_pairs)):
        pre1 = save_pair[save_pair['Resid1'] == key_cd1_pairs[i][0]]
        pre2 = pre1[pre1['Resid2'] == key_cd1_pairs[i][1]]
        fin_location = pre2['Dist_loc']
        first = True
        for x in fin_location.values:
            # Have to do some crazy processing when loading back in the saved dataframe:
            pain = x[1:-1]+','
            pain2 = pain.split()
            fin = [int(a[:-1]) for a in pain2]
            pocket_sub = pocket_dists[:,fin]
            cryst_sub = cryst_dists[:,fin]
            if first:
                pocket_f = pocket_sub
                cryst_f = cryst_sub
                first = False
            else:
                pocket_f = np.hstack((pocket_f,pocket_sub))
                cryst_f = np.hstack((cryst_f,cryst_sub))
                
        fin_dist = [key_cd1_pairs[i][0],key_cd1_pairs[i][1],np.average(cryst_f),np.average(pocket_f,axis=1)]
        if len(final_distance_df) == 0:
            final_distance_df = np.transpose(pandas.DataFrame(fin_dist))
        else:
            final_distance_df = pandas.concat([final_distance_df, np.transpose(pandas.DataFrame(fin_dist))])

    fig, ax = pl.subplots(2, 2,squeeze=False,figsize=(16,12))
    ax[0,0].plot(final_distance_df.values[0][3],color='Gray')
    ax[0,0].set_title('Ser76-Trp153'); ax[0,0].set_ylim([0,1.2])
    ax[0,1].plot(final_distance_df.values[1][3],color='Gray')
    ax[0,1].set_title('Tyr73-Thr154'); ax[0,1].set_ylim([0,1.2])
    ax[1,0].plot(final_distance_df.values[2][3],color='Gray')
    ax[1,0].set_title('Phe70-Leu161'); ax[1,0].set_ylim([0,1.2])
    ax[1,1].plot(final_distance_df.values[3][3],color='Gray')
    ax[1,1].set_title('Ile69-Leu161'); ax[1,1].set_ylim([0,1.2])
    traj_len = len(final_distance_df.values[0][3])
    ax[0,0].plot(np.arange(traj_len),np.ones(traj_len)*final_distance_df.values[0][2],color='black',linewidth=2.5)
    ax[0,1].plot(np.arange(traj_len),np.ones(traj_len)*final_distance_df.values[1][2],color='black',linewidth=2.5)
    ax[1,0].plot(np.arange(traj_len),np.ones(traj_len)*final_distance_df.values[2][2],color='black',linewidth=2.5)
    ax[1,1].plot(np.arange(traj_len),np.ones(traj_len)*final_distance_df.values[3][2],color='black',linewidth=2.5)

    pl.savefig(Simdir+'_keyPairdist.pdf',format='pdf')

    # Lastly, double check that you were actually looking at the correct residues:
    def convert_3Let(inp):
        first = True
        three_let = ['ALA','GLY','ARG','LYS','ASP','GLU','ASN','GLN','MET','CYS','PHE','TYR','THR','TRP','PRO','SER','LEU','VAL','HIS','ILE']
        sin_let = [  'A',  'G',  'R',  'K',  'D',  'E',  'N',  'Q',  'M',  'C',  'F',  'Y',  'T',  'W',  'P',  'S',  'L',  'V',  'H',  'I']
        sin_final = []
        for i in inp:
            hold = []
            for scan in np.arange(len(three_let)):
                if i.lower() == three_let[scan].lower():
                    hold = sin_let[scan]
                    break
            # In these pdbs especially, there will occasionally be some
            # weird residues (artifical AAs or otherwise)
            if len(hold) == 0:
                continue
            if first:
                sin_final = hold
                first = False
            else:
                sin_final = np.hstack((sin_final,hold))
        if len(sin_final) == 0:
            return()
        return(sin_final)

    first = True
    for atom_index in mdsel:
        atom_ID = table[table['serial'] == atom_index][['resName','resSeq','name']].values[0]
        atom_df = np.transpose(pandas.DataFrame(atom_ID))
        if first:
            atom_cat = atom_df
            first = False
        else:
            atom_cat = pandas.concat([atom_cat,atom_df],axis=0)
            
    seq_check = ''.join(convert_3Let(atom_cat.drop_duplicates(1)[0].values))
    print('Do the sequences match what we expect?:')
    print(cd1_resname == seq_check)
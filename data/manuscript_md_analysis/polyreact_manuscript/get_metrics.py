import numpy as np
import mdtraj as md
import os
import ctb_md_analysis as ctb

for ab in ['CH65','1F02','CR9114','F16','2g02','2g02mut','338e6','43g10','4c05','4b03']:
    # navigate to your specific directory using  
    os.chdir('/hpcdata/cbg/cbg_data/boughter/simulation/fab_sims/'+ab+'_charmmed/amber')

    # Need to change all of these depending on if you're looking at 
    # GPU or NAMD simulations
    if ab == 'CH65' or ab == '1F02' or ab == 'CR9114' or ab == 'F16':
        dyns = 75
    else:
        dyns = 250
    
    cdr1l,cdr2l,cdr3l,cdr1h,cdr2h,cdr3h = ctb.get_cdr_loc(ab)

    # PBCsetup is the naming convention for some of the NAMD simulations
    #topfile='../step3_pbcsetup.pdb'
    topfile='step3_charmm2amber.pdb'
    first_frame = md.load_frame(topfile,0)

    rmsd1lpre = []; rmsd2lpre = []; rmsd3lpre = []; rmsd1hpre = []; rmsd2hpre = []; rmsd3hpre =[]
    frames=200
    for dyn in np.arange(dyns):
        chunk=md.load('dyn'+str(dyn+1)+'.dcd',top=topfile)
        rmsd1lpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr1l))
        rmsd2lpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr2l))
        rmsd3lpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr3l))
        rmsd1hpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr1h))
        rmsd2hpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr2h))
        rmsd3hpre.append(md.rmsd(chunk, first_frame,atom_indices=cdr3h))
        #print(chunk, '\n', chunk.time)
    rmsd1l=np.reshape(rmsd1lpre,(dyn+1)*frames);rmsd2l=np.reshape(rmsd2lpre,(dyn+1)*frames);rmsd3l=np.reshape(rmsd3lpre,(dyn+1)*frames)
    rmsd1h=np.reshape(rmsd1hpre,(dyn+1)*frames);rmsd2h=np.reshape(rmsd2hpre,(dyn+1)*frames);rmsd3h=np.reshape(rmsd3hpre,(dyn+1)*frames)

    np.savetxt(ab+'rmsd1l.dat',rmsd1l); np.savetxt(ab+'rmsd2l.dat',rmsd2l); np.savetxt(ab+'rmsd3l.dat',rmsd3l)
    np.savetxt(ab+'rmsd1h.dat',rmsd1h); np.savetxt(ab+'rmsd2h.dat',rmsd2h); np.savetxt(ab+'rmsd3h.dat',rmsd3h)

    # NOTE ABOUT THE RMSF: For some reason the .image_molecules() function used on chunk_pre2 did NOT work
    # We'd still get massive RMSF jumps from periodic boundary crossing. Instead had to use the VMD
    # function pbc wrap. Worked much better and was applied to all systems, so the below
    # code assume NO periodic boundary crossings throughout the whole trajectory.

    rmsfpre=[]
    xxx=first_frame.topology.select("protein")
    first_new = first_frame.remove_solvent()
    for dyn in np.arange(dyns):
        chunk_pre=md.load('dyn'+str(dyn+1)+'.dcd',top=topfile)
        chunk_pre2 = chunk_pre.remove_solvent()
        chunk = chunk_pre2.superpose(first_new,0)

        rmsfpre.append(md.rmsf(chunk, first_frame,0,atom_indices=xxx))

    np.savetxt(ab+'rmsfFULL.dat',rmsfpre)

    # Chunk up the RMSF into individual loops.
    # Can't use these for bootstrapping, unfortunately.
    rmsf1l = []; rmsf2l = []; rmsf3l = []; rmsf1h = []; rmsf2h = []; rmsf3h =[]
    for j in np.arange(dyns):
        for i in cdr3h:
            rmsf3h.append(rmsfpre[j][i])
        for i in cdr2h:
            rmsf2h.append(rmsfpre[j][i])
        for i in cdr1h:
            rmsf1h.append(rmsfpre[j][i])
        for i in cdr3l:
            rmsf3l.append(rmsfpre[j][i])
        for i in cdr2l:
            rmsf2l.append(rmsfpre[j][i])
        for i in cdr1l:
            rmsf1l.append(rmsfpre[j][i])

    np.savetxt(ab+'rmsf1l.dat',rmsf1l); np.savetxt(ab+'rmsf2l.dat',rmsf2l); np.savetxt(ab+'rmsf3l.dat',rmsf3l)
    np.savetxt(ab+'rmsf1h.dat',rmsf1h); np.savetxt(ab+'rmsf2h.dat',rmsf2h); np.savetxt(ab+'rmsf3h.dat',rmsf3h)

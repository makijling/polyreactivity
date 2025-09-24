import pandas
import numpy as np
from sklearn.utils import resample

def get_cdr_loc(ab):
    if ab.lower() == 'ch65':
        cdr1l = [3660, 3667, 3681, 3693, 3712, 3719, 3743, 3765]; cdr2l = [4032, 4043, 4064, 4076, 4087, 4099, 4125]; cdr3l = [4629, 4653, 4665, 4676, 4687, 4698, 4710, 4727]
        cdr3h = [1472, 1496, 1503, 1510, 1529, 1546, 1558, 1582, 1593, 1609, 1621, 1642, 1663, 1684, 1705, 1712, 1729, 1741]
        cdr2h = [788, 807, 819, 833, 844, 851]; cdr1h = [368, 375, 396, 410, 430, 444, 456, 477]
        #################################################################

    elif ab.lower() == '1f02':
        cdr1h = [329, 340, 347, 354, 373, 393, 417, 428]; cdr2h = [733, 752, 762, 778, 798, 805]; cdr3h = [1418, 1442, 1451, 1463, 1484, 1505, 1526, 1533, 1547, 1558, 1575, 1594, 1606]
        cdr1l = [3608, 3619, 3636, 3647, 3663, 3674, 3685, 3699]; cdr2l = [3973, 3992, 4013, 4020, 4030, 4041, 4055, 4079]; cdr3l = [4586, 4603, 4624, 4631, 4645, 4658, 4670, 4694]
        #################################################################

    elif ab.lower() == 'cr9114':
        cdr1l=[3565, 3577, 3588, 3602, 3621, 3628, 3652]; cdr2l = [3935, 3954, 3975, 3986, 4000, 4012, 4029]; cdr3l = [4547, 4571, 4583, 4595, 4606, 4625, 4647, 4654]
        cdr1h=[355, 362, 369, 383, 394, 408, 422]; cdr2h=[724, 737, 749, 768, 788, 795]; cdr3h=[1431, 1448, 1455, 1469, 1490, 1511, 1532, 1553, 1564, 1571]
        #################################################################

    elif ab.lower() == 'f16':
        cdr1l=[3782, 3793, 3810, 3821, 3837, 3851, 3871, 3885, 3906]; cdr2l=[4238, 4259, 4283, 4293, 4304]; cdr3l=[4833, 4850, 4867, 4888, 4912, 4928, 4942, 4954]
        cdr3h=[1542, 1559, 1578, 1602, 1613, 1632, 1651, 1672, 1692, 1699, 1723, 1742, 1753]
        cdr2h=[748, 767, 778, 799, 811, 821, 835]; cdr1h=[345, 356, 363, 383, 397, 417, 428]

    elif ab.lower() == '43g10':
        cdr1h = [387, 398, 417, 431, 442, 449, 469, 490, 514]; cdr2h = [822, 843, 853, 860, 874, 888, 900, 921, 937];cdr3h = [1511, 1535, 1542, 1556, 1563, 1587, 1606, 1621, 1642, 1666, 1673]
        cdr1l = [3687, 3698, 3717, 3736, 3757, 3768, 3789, 3803, 3820, 3842, 3856, 3877];cdr2l = [4182, 4192, 4203];cdr3l = [4747, 4764, 4785, 4802, 4813, 4832, 4844, 4863, 4877, 4897, 4904]
        #################################################################

    elif ab.lower() == '2g02':
        cdr1h = [384, 405, 419, 439, 450, 464, 485, 492]; cdr2h = [786, 797, 807, 828, 842, 849, 866, 880]; cdr3h = [1457, 1467, 1491, 1503, 1527, 1551, 1563, 1582, 1601, 1615, 1622, 1633, 1652, 1659, 1671, 1692, 1716]
        cdr1l = [3696, 3703, 3722, 3741, 3762, 3781, 3793, 3800, 3814, 3828, 3849]; cdr2l = [4164, 4180, 4191]; cdr3l = [4730, 4747, 4764, 4771, 4785, 4806, 4832, 4844, 4864, 4878, 4898]
        #################################################################

    elif ab.lower() == '338e6':
        cdr1h = [352, 359, 379, 393, 413, 424, 435]; cdr2h = [756, 775, 786, 797, 804, 811, 822]; cdr3h = [1466, 1476, 1500, 1524, 1531, 1551, 1572, 1592, 1604]
        cdr1l = [3639, 3656, 3670, 3686, 3693]; cdr2l = [4002, 4013]; cdr3l = [4564, 4581, 4598, 4619, 4633, 4644, 4667, 4679, 4698]
        #################################################################

    elif ab.lower() == '2g02mut':
        cdr1h = [384, 405, 419, 439, 450, 464, 485, 492]; cdr2h = [786, 797, 807, 828, 842, 849, 866, 880]; cdr3h = [1457, 1467, 1491, 1503, 1525, 1547, 1559, 1578, 1597, 1611, 1618, 1629, 1648, 1655, 1667, 1688, 1712]
        cdr1l = [3682, 3689, 3708, 3727, 3748, 3767, 3779, 3786, 3800, 3814, 3835]; cdr2l = [4150, 4166, 4177]; cdr3l = [4716, 4733, 4750, 4757, 4771, 4792, 4818, 4830, 4850, 4864, 4884]

    elif ab.lower() == '3b03':
        cdr1h=[348, 368, 382, 402, 413, 427, 448, 458]; cdr2h=[744, 755, 762, 776, 783, 790, 804, 818, 839, 860]; cdr3h=[1422, 1438, 1460, 1467, 1484, 1504, 1523, 1538, 1562, 1583, 1597, 1621, 1641, 1655, 1667, 1691]
        cdr1l=[3660, 3671, 3687, 3698, 3709, 3723]; cdr2l=[4024, 4034, 4045]; cdr3l=[4564, 4581, 4598, 4609, 4621, 4635, 4661, 4675, 4687, 4701, 4721]

    elif ab.lower() == '4c05':
        cdr1l = [326, 340, 354, 373, 380, 391, 413, 424, 440]; cdr2l = [991, 998, 1012];cdr3l = [1286, 1303, 1319, 1343, 1355, 1379, 1393, 1412, 1426, 1438, 1455]
        cdr1h = [3504, 3511, 3518, 3529, 3546, 3570, 3581];cdr2h = [3924, 3945, 3966, 3977, 3984];cdr3h = [4593, 4617, 4624, 4640, 4651, 4661, 4680, 4696, 4707, 4723, 4735, 4756, 4777, 4798, 4819, 4840, 4857]

    return(cdr1l,cdr2l,cdr3l,cdr1h,cdr2h,cdr3h)


def get_rmsf(dat_input,ab,dyns=50):

    cdr1l,cdr2l,cdr3l,cdr1h,cdr2h,cdr3h = get_cdr_loc(ab)

    rmsf_1l_pre = dat_input.values[:dyns,cdr1l]
    rmsf_2l_pre = dat_input.values[:dyns,cdr2l]
    rmsf_3l_pre = dat_input.values[:dyns,cdr3l]
    rmsf_1h_pre = dat_input.values[:dyns,cdr1h]
    rmsf_2h_pre = dat_input.values[:dyns,cdr2h]
    rmsf_3h_pre = dat_input.values[:dyns,cdr3h]

    return(rmsf_1l_pre,rmsf_2l_pre,rmsf_3l_pre,rmsf_1h_pre,rmsf_2h_pre,rmsf_3h_pre)

# Keep the data in a single file. Don't alter that
# Coming from the get_rmsf command
def reshape_rmsf(data,N_nanosecond=500,dyns=250):
    # Nframe is now number of nanoseconds per dyn
    # here we can now auto-encode time dependence, instead
    # of doing some kind of weird weighting here.
    Nframe = N_nanosecond/dyns
    rmsf_1l_pre = data[0]
    rmsf_2l_pre = data[1]
    rmsf_3l_pre = data[2]
    rmsf_1h_pre = data[3]
    rmsf_2h_pre = data[4]
    rmsf_3h_pre = data[5]

    for i in np.arange(dyns):
        xx2=(rmsf_1l_pre[i]**2)*Nframe
        xx1=(rmsf_2l_pre[i]**2)*Nframe
        xx0=(rmsf_3l_pre[i]**2)*Nframe
        yy2=(rmsf_1h_pre[i]**2)*Nframe
        yy1=(rmsf_2h_pre[i]**2)*Nframe
        yy0=(rmsf_3h_pre[i]**2)*Nframe
        if i == 0:
            rmsf1LFull=xx2 ; rmsf2LFull=xx1 ; rmsf3LFull=xx0
            rmsf1HFull=yy2 ; rmsf2HFull=yy1 ; rmsf3HFull=yy0
        else:
            rmsf1LFull=rmsf1LFull+xx2; rmsf2LFull=rmsf2LFull+xx1; rmsf3LFull=rmsf3LFull+xx0
            rmsf1HFull=rmsf1HFull+yy2; rmsf2HFull=rmsf2HFull+yy1; rmsf3HFull=rmsf3HFull+yy0
            
    rmsf3hfull=np.sqrt(rmsf3HFull/(Nframe*(i+1))); rmsf3lfull=np.sqrt(rmsf3LFull/(Nframe*(i+1))); 
    rmsf2hfull=np.sqrt(rmsf2HFull/(Nframe*(i+1))); rmsf2lfull=np.sqrt(rmsf2LFull/(Nframe*(i+1)))
    rmsf1hfull=np.sqrt(rmsf1HFull/(Nframe*(i+1))); rmsf1lfull=np.sqrt(rmsf1LFull/(Nframe*(i+1)))

    return(rmsf1lfull,rmsf2lfull,rmsf3lfull,rmsf1hfull,rmsf2hfull,rmsf3hfull)

# For now, we are only comparing triplicates, and I'm just going to hard code it
# Could probably be smarter about this... but I don't want to
def rmsf_compile(data1,N1,dyn1,data2,N2,dyn2,data3,N3,dyn3):
    # Nframe is now number of nanoseconds per dyn
    # here we can now auto-encode time dependence, instead
    # of doing some kind of weird weighting here.
    Nframe1 = N1/dyn1; Nframe2 = N2/dyn2; Nframe3 = N3/dyn3
    totTime = N1+N2+N3
    maxDyn = np.max([dyn1,dyn2,dyn3])

    # break all of our data into the individual chains:
    rmsf_1l_pre1 = data1[0]; rmsf_1l_pre2 = data2[0]; rmsf_1l_pre3 = data3[0]
    rmsf_2l_pre1 = data1[1]; rmsf_2l_pre2 = data2[1]; rmsf_2l_pre3 = data3[1]
    rmsf_3l_pre1 = data1[2]; rmsf_3l_pre2 = data2[2]; rmsf_3l_pre3 = data3[2]
    rmsf_1h_pre1 = data1[3]; rmsf_1h_pre2 = data2[3]; rmsf_1h_pre3 = data3[3]
    rmsf_2h_pre1 = data1[4]; rmsf_2h_pre2 = data2[4]; rmsf_2h_pre3 = data3[4]
    rmsf_3h_pre1 = data1[5]; rmsf_3h_pre2 = data2[5]; rmsf_3h_pre3 = data3[5]

    for i in np.arange(maxDyn):
        if i < dyn1:
            pre_1l=(rmsf_1l_pre1[i]**2)*Nframe1
            pre_2l=(rmsf_2l_pre1[i]**2)*Nframe1
            pre_3l=(rmsf_3l_pre1[i]**2)*Nframe1
            pre_1h=(rmsf_1h_pre1[i]**2)*Nframe1
            pre_2h=(rmsf_2h_pre1[i]**2)*Nframe1
            pre_3h=(rmsf_3h_pre1[i]**2)*Nframe1
            has1 = True
        if i < dyn2:
            if has1:
                pre_1l=np.vstack((pre_1l,(rmsf_1l_pre2[i]**2)*Nframe2))
                pre_2l=np.vstack((pre_2l,(rmsf_2l_pre2[i]**2)*Nframe2))
                pre_3l=np.vstack((pre_3l,(rmsf_3l_pre2[i]**2)*Nframe2))
                pre_1h=np.vstack((pre_1h,(rmsf_1h_pre2[i]**2)*Nframe2))
                pre_2h=np.vstack((pre_2h,(rmsf_2h_pre2[i]**2)*Nframe2))
                pre_3h=np.vstack((pre_3h,(rmsf_3h_pre2[i]**2)*Nframe2))
            else:
                pre_1l=(rmsf_1l_pre2[i]**2)*Nframe2
                pre_2l=(rmsf_2l_pre2[i]**2)*Nframe2
                pre_3l=(rmsf_3l_pre2[i]**2)*Nframe2
                pre_1h=(rmsf_1h_pre2[i]**2)*Nframe2
                pre_2h=(rmsf_2h_pre2[i]**2)*Nframe2
                pre_3h=(rmsf_3h_pre2[i]**2)*Nframe2
            has2 = True
        if i < dyn3:
            if has1 or has2:
                pre_1l=np.vstack((pre_1l,(rmsf_1l_pre3[i]**2)*Nframe3))
                pre_2l=np.vstack((pre_2l,(rmsf_2l_pre3[i]**2)*Nframe3))
                pre_3l=np.vstack((pre_3l,(rmsf_3l_pre3[i]**2)*Nframe3))
                pre_1h=np.vstack((pre_1h,(rmsf_1h_pre3[i]**2)*Nframe3))
                pre_2h=np.vstack((pre_2h,(rmsf_2h_pre3[i]**2)*Nframe3))
                pre_3h=np.vstack((pre_3h,(rmsf_3h_pre3[i]**2)*Nframe3))
            else:
                pre_1l=(rmsf_1l_pre3[i]**2)*Nframe3
                pre_2l=(rmsf_2l_pre3[i]**2)*Nframe3
                pre_3l=(rmsf_3l_pre3[i]**2)*Nframe3
                pre_1h=(rmsf_1h_pre3[i]**2)*Nframe3
                pre_2h=(rmsf_2h_pre3[i]**2)*Nframe3
                pre_3h=(rmsf_3h_pre3[i]**2)*Nframe3
            has3 = True

        # Alright so everything above was about "pulling the data out" of the time average
        # So, in the calculation of RMSF, we should just have a sum of deviations over a trajectory chunk
        # We've been compiling these into a matrix, independent of which triplicate run we're using
        # so in this next step we're compiling these further...

        if i == 0:
            rmsf1Lcat=pre_1l ; rmsf2Lcat=pre_2l ; rmsf3Lcat=pre_3l
            rmsf1Hcat=pre_1h ; rmsf2Hcat=pre_2h ; rmsf3Hcat=pre_3h
        else:
            rmsf1Lcat=np.vstack((rmsf1Lcat,pre_1l)) ; rmsf2Lcat=np.vstack((rmsf2Lcat,pre_2l)) ; rmsf3Lcat=np.vstack((rmsf3Lcat,pre_3l))
            rmsf1Hcat=np.vstack((rmsf1Hcat,pre_1h)) ; rmsf2Hcat=np.vstack((rmsf2Hcat,pre_2h)) ; rmsf3Hcat=np.vstack((rmsf3Hcat,pre_3h))

        # Reset this because at some point we have more dyns in one sample or another.
        has1 = False; has2 = False; has3 = False

    return(rmsf1Lcat,rmsf2Lcat,rmsf3Lcat,rmsf1Hcat,rmsf2Hcat,rmsf3Hcat)

# assume that we're taking the ouputs of rmsf_compile
# I don't think (?) that we can permute these all at once,
# so let's just try to pick out individual loops at a time
def rmsf_stats(compiled1,compiled2,totTime1=2000,totTime2=2000,num_reps = 1000,loop=0,loc=[]):

    if loop == -1:
        dset1 = compiled1
        dset2 = compiled2
    else:
        if len(loc) != 0:
            dset1 = compiled1[loop][:,loc]
            dset2 = compiled2[loop][:,loc]
        else:
            dset1 = compiled1[loop]
            dset2 = compiled2[loop]

    datlen1 = len(dset1); datlen2 = len(dset2)
    z0 = np.sqrt(np.sum(dset1,axis=0)/(totTime1)) - np.sqrt(np.sum(dset2,axis=0)/(totTime2))

    fullDat = np.vstack((dset1,dset2))

    for rep in np.arange(num_reps):
        permute_dat = resample(fullDat,replace=False)

        z = np.sqrt(np.sum(permute_dat[:datlen1],axis=0)/(totTime1)) - np.sqrt(np.sum(permute_dat[datlen1:],axis=0)/(totTime2))

        if type(z0) == np.float64:
            if z**2 >= z0**2:
                num_sig += 1
        else:
            # It turns out this works pretty nicely, doing
            # element-wise comparisons of the data
            z_temp = z**2 >= z0**2
            z_check = z_temp.astype(int)

            if rep == 0:
                num_sig = z_check
            else:
                num_sig += z_check

    p = (num_sig+1)/(num_reps+1)
    return(p)

def rmsf_boot(compiled1,totTime1=2000,num_reps = 1000,loop=0):

    if loop == -1:
        dset1 = compiled1
    else:
        dset1 = compiled1[loop]

    boot_list = []
    for rep in np.arange(num_reps):
        dset_boot = resample(dset1,replace=True)
        tempDat = np.sqrt(np.sum(dset_boot,axis=0)/(totTime1))
        boot_list.append(tempDat)

    avg = np.average(boot_list,axis=0)
    
    # For some reason, the numpy standard deviation ALWAYS underestimates the error
    std = np.sqrt(np.sum((boot_list-avg)**2,axis=0)/(len(boot_list)-1))
    # But... on implementation, there doesn't seem to be a meaningful difference...
    #std = np.std(boot_list,axis=0)

    return(avg,std)
import os
from shutil import copy2, copytree, rmtree, move
import pandas as pd
import numpy as np

def calculate_aTL_stats(Func_DIR, ROI_DIR):
    """Expects to be called in a for loop iterating over sub_list, in which ROI_DIR and Func_DIR have been defined"""
    
    calc_source = f"{Func_DIR}/tfMRI_LANGUAGE_hp200_s4.gfeat/cope1.feat/thresh_zstat1.nii.gz"
    lh_ROI = f"{ROI_DIR}/lh_aTL_mask.nii.gz"
    rh_ROI = f"{ROI_DIR}/rh_aTL_mask.nii.gz"
    
    # calculate stats for lh_TL
    os.system(f"mri_segstats --in {calc_source} --seg {lh_ROI} --id 1 --sum ant_lh_table.txt --sumwf ant_lh_table2.txt")
    # calculate stats for rh_TL
    os.system(f"mri_segstats --in {calc_source} --seg {rh_ROI} --id 1 --sum ant_rh_table.txt --sumwf ant_rh_table2.txt")
    
    
def calculate_ant_LI(subject):
    """Expects stats to be run first, calculates LI within the aTL"""
    
    lh_count = 0
    rh_count = 0
    with open('ant_lh_table2.txt', 'r') as lh:
        c_l = lh.read()
        count_l = float(c_l.strip())
        lh_count += count_l
    with open('ant_rh_table2.txt', 'r') as rh:
        c_r = rh.read()
        count_r = float(c_r.strip())
        rh_count += count_r
        
    ant_LI = (lh_count - rh_count)/(lh_count + rh_count)
    # Create csv with subs and LI
    
    with open('/media/felix/HCP_HDD/ant_LI.csv', 'a') as index:
        index.write(f"{subject}, {ant_LI}\n")
    print(ant_LI)
    
    
for i in sub_list:
    path = os.path.join(SUB_DIR, i)
    ROI_DIR = f"{path}/Structural/MNINonLinear/ROIs/"
    Func_DIR = f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE/"
    
    os.chdir(Func_DIR)
    
    calculate_aTL_stats(Func_DIR, ROI_DIR)
    calculate_ant_LI(i)

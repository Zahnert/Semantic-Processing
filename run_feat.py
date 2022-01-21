import os
import glob
from shutil import copy2, copytree, rmtree
from multiprocessing import Pool
from time import process_time

def run_lvl1(PE_dir):
    """Expects Phase Encoding direction as input. Runs customized fsf file for Feat"""
    os.system(f"feat /home/felix/neuro-bin/language/lvl1_{PE_dir}.fsf")

def replace_old_lvl2_fsf():
    """Expects to be in tfMRI_LANGUAGE dir. renames original fsf and copies custom fsf from neuro-bin"""
    old_fsf = './tfMRI_LANGUAGE_hp200_s4_level2.fsf'
    renamed_fsf = './old.fsf'
    os.rename(old_fsf, renamed_fsf)
    copy2('/home/felix/neuro-bin/language/tfMRI_LANGUAGE_hp200_s4_level2.fsf', './')

def cp_prep_script(path_to_subject):
    """Not all sub_dirs contain prep script. This is solved here."""
    
    prep_script = 'prepare_level2_feat_analysis.sh'
    lvl2_dir = f"{path_to_subject}/Func/MNINonLinear/tfMRI_LANGUAGE"
    
    filelist=[]
    for i in os.listdir(lvl2_dir):
        filename = str(i)
        filelist.append(filename)
    
    if prep_script not in filelist:
        copy2('/path/to/subjects_dir/100206/Func/MNINonLinear/tfMRI_LANGUAGE/prepare_level2_feat_analysis.sh', lvl2_dir)
        
 def run_feat_lvl_1(i):
    """Expects to be run in a for loop iterating over a subjects list, e.g. by pool"""
    
    path = os.path.join(SUB_DIR, i)
    
    # replace the original lvl2.fsf file
    os.chdir(f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE")
    replace_old_lvl2_fsf()

    # add in prepare_lvl2_script where it is missing
    cp_prep_script(path)

    # first lvl analysis
    for j in 'LR', 'RL':
        os.chdir(f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE_{j}")  # cd into LR_dir / RL_dir
        run_lvl1(j)
        
        
def run_feat_lvl_2(i): 
    """Expects to be run in a for loop iterating over a subjects list, due to abspath to tmp dir not eligible for parallel processing"""
    path = os.path.join(SUB_DIR, i)
    
    # run prepare lvl2 script
    os.chdir(f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE")
    os.system("./prepare_level2_feat_analysis.sh tfMRI_LANGUAGE_hp200_s4_level2.fsf")

    # create tmp dir so that we can generate abspath for lvl1 feat dirs for our lvl2 analysis.
    os.mkdir(f'/tmp/Feat')
    for k in 'LR', 'RL':
        os.chdir(f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE_{k}")  # cd into LR_dir / RL_dir
        dir_to_copy = f'./tfMRI_LANGUAGE_{k}_hp200_s4.feat'
        copytree(dir_to_copy, f'/tmp/Feat/tfMRI_LANGUAGE_{k}_hp200_s4.feat')

    # run lvl 2 on tmp-dirs && rm them afterwards
    os.chdir(f"{path}/Func/MNINonLinear/tfMRI_LANGUAGE")
    os.system("feat tfMRI_LANGUAGE_hp200_s4_level2.fsf")

    # Cleaning up
    rmtree('/tmp/Feat')
    

## Here we run the first lvl analysis with multiprocessing
pool = Pool(6, maxtasksperchild=2) 

try:
    pool.map(run_feat_lvl_1, sub_list)
finally:
    pool.close()
    pool.join()
    
    
## Second lvl analysis
t1_start = process_time()

for i in sub_list:
    run_feat_lvl_2(i)
    
t1_stop = process_time()

print(t1_stop-t1_start)

import os
from shutil import copy2, copytree, rmtree, move
from multiprocessing import Pool
from time import process_time


def gen_ant_aseg(path):
    """generates anterior aparc+aseg (anterior to MNI -22) for later warp to str space"""
    
    os.chdir(f'{path}/Structural/MNINonLinear/')
    os.system(f'fslmaths aparc+aseg.nii.gz -roi 0 -1 149 161 0 -1 0 -1 ant_aseg.nii.gz')
    

    
def reg_ant2str(path):
    """register anterior aparc+aseg (anterior MNI Y-22) to str = diffusion space"""
    
    os.chdir(f'{path}/Structural/MNINonLinear/')
    os.system(f'applywarp --ref={path}/Structural/T1w/aparc+aseg.nii.gz --in=ant_aseg.nii.gz --warp=./xfms/standard2acpc_dc.nii.gz \
    --out={path}/Structural/T1w/ant_aseg_str.nii.gz')
    
    
    
def resample_str2diff(path):
    """Extracts nodif and resamples parcellation from 0.7 mm to 1.25 mm (as diffusion)"""
    """Does this for the anterior and normal aseg"""
    
    os.chdir(f"{path}/Diffusion/")
    os.system(f"fslroi data.nii.gz nodif.nii.gz 0 1")
    nodif = f"{path}/Diffusion/nodif.nii.gz"
    aseg = f"{path}/Structural/T1w/aparc+aseg.nii.gz"
    aseg_ant = f"{path}/Structural/T1w/ant_aseg_str.nii.gz"
    os.system(f"mri_vol2vol --mov {nodif} --targ {aseg} --inv --interp nearest --o aseg2diff.nii.gz --regheader")
    os.system(f"mri_vol2vol --mov {nodif} --targ {aseg_ant} --inv --interp nearest --o aseg_ant2diff.nii.gz --regheader")
    
    
def binarize_ant_aseg(path):
    os.chdir(f"{path}/Diffusion/")
    os.system(f"fslmaths aseg_ant2diff.nii.gz -bin aseg_ant2diff_bin.nii.gz")
    
    
def gen_seeds(path):
    """runs tcsh script for Desikan or Destrieux. Expects seed_list.txt in neuro-bin and aseg2diff in Diffusion dir"""
    """Generates seeds and seeds.txt, which contains paths to seeds"""
    """This script, too, can be found in this repository"""
    
    os.mkdir(f"{path}/Diffusion/seeds")
    os.system(f"/home/felix/neuro-bin/language/generate_seeds.sh {path}/Diffusion/")

    

def mod_seeds(path):
    """Splits TL seeds in anterior and posterior"""
    
    os.chdir(f"{path}/Diffusion/seeds")
    split_ctx = ['superiortemporal', 'inferiortemporal', 'middletemporal', 'parahippocampal', 'transversetemporal', 'fusiform']
    
    for hemi in 'rh', 'lh':
        for roi in split_ctx:
            os.system(f'fslmaths ctx-{hemi}-{roi}.nii.gz -mul {path}/Diffusion/aseg_ant2diff_bin.nii.gz ctx-{hemi}-ant-{roi}.nii.gz')
            os.system(f'fslmaths ctx-{hemi}-{roi}.nii.gz -sub ctx-{hemi}-ant-{roi}.nii.gz ctx-{hemi}-post-{roi}.nii.gz')
            
    for hemi in 'Right', 'Left':
        os.system(f'fslmaths {hemi}-Hippocampus.nii.gz -mul {path}/Diffusion/aseg_ant2diff_bin.nii.gz {hemi}-ant-Hippocampus.nii.gz')
        os.system(f'fslmaths {hemi}-Hippocampus.nii.gz -sub {hemi}-ant-Hippocampus.nii.gz {hemi}-post-Hippocampus.nii.gz')
        
        
        
        
def new_seeds_text(path):
    """creates a new seeds text to run probtrackX for custom parcellation from"""
    
    seed_dir = f"{path}/Diffusion/seeds"
    os.chdir(seed_dir)
    move(f'{seed_dir}/seeds.txt', f"{path}/Diffusion/")
    split_ctx = ['superiortemporal', 'inferiortemporal', 'middletemporal', 'parahippocampal', 'transversetemporal', 'fusiform']

    seed_list = []
    for i in os.listdir(seed_dir):
        seed = str(i)
        seed_list.append(seed)
    
    for hemi in 'rh', 'lh':
        for i in split_ctx:
            seed = f"ctx-{hemi}-{i}.nii.gz"
            if seed in seed_list:
                seed_list.remove(seed)
                
    seed_list.remove('Right-Hippocampus.nii.gz')
    seed_list.remove('Left-Hippocampus.nii.gz')
    
    with open('seeds2.txt', 'a') as s:
        for i in seed_list:
            s.write(f"{seed_dir}/{i}\n")
    move(f"{path}/Diffusion/seeds.txt", seed_dir)
    
    
    
def seeds3(path):
    """creates seeds w/o cerebellum"""
    
    os.chdir(f'{path}/Diffusion/seeds/')
    with open('seeds2.txt', 'r') as s:
        lines = s.readlines()
    with open("seeds3.txt", "w") as s:
        for line in lines:
            if line.strip("\n") != f"{path}/Diffusion/seeds/Left-Cerebellum-Cortex.nii.gz" and line.strip("\n") != f"{path}/Diffusion/seeds/Right-Cerebellum-Cortex.nii.gz":
                s.write(line)
                

                
def to_interface(path):
    """creates gm/wm interface, multiplies seeds with this and replaces them"""
    
    os.chdir(f"{path}/Diffusion/")
    os.system(f'fslmaths white.nii.gz -dilD white_dil.nii.gz')
    seed_dir = f"{path}/Diffusion/seeds"
    os.chdir(seed_dir)              
    cortical_seeds = []
    
    for i in os.listdir(seed_dir):
        seed = str(i)
        cortical_seeds.append(seed)
        
    matching = [i for i in cortical_seeds if "ctx" not in i]
    new_seeds = [i for i in cortical_seeds if i not in matching]
    for seed in new_seeds:
        os.system(f'fslmaths {seed} -mul {path}/Diffusion/white_dil.nii.gz int_{seed}')
        os.remove(f'./{seed}')
        os.rename(f'int_{seed}', seed)
           

            
  def ribbon_exclusion(path):
    """Subtracts white_dil from ribbon for exclusion from tracking"""
    
    # ribbon to diff
    os.chdir(f"{path}/Diffusion/")
    nodif = f"{path}/Diffusion/nodif.nii.gz"
    ribbon = f"{path}/Structural/T1w/ribbon.nii.gz"
    seeds = f"{path}/Diffusion/seeds/"
    os.system(f"mri_vol2vol --mov {nodif} --targ {ribbon} --inv --interp nearest --o ribbon2diff_full.nii.gz --regheader")
    
    # binarize ribbon
    os.system(f'mri_binarize --i ribbon2diff_full.nii.gz --match 3 --o left_ribbon_diff.nii.gz')
    os.system(f'mri_binarize --i ribbon2diff_full.nii.gz --match 42 --o right_ribbon_diff.nii.gz')
    
    # add l+r
    os.system(f'fslmaths left_ribbon_diff.nii.gz -add right_ribbon_diff.nii.gz ribbon2diff.nii.gz')
    
    # create exclusion
    os.system(f'fslmaths ribbon2diff.nii.gz -sub white_dil.nii.gz ribbon_subtr.nii.gz')
    os.system(f'mri_binarize --i ribbon_subtr.nii.gz --match 1 --o ribbon_bin.nii.gz')
    
    # remove Archicortex from this
    os.system(f'fslmaths ribbon_bin.nii.gz -sub {seeds}/Right-Hippocampus.nii.gz -sub {seeds}/Left-Hippocampus.nii.gz \
    -sub {seeds}/Left-Amygdala.nii.gz -sub {seeds}/Right-Amygdala.nii.gz ribbon_preprefinal.nii.gz')
    
    # binarize again
    os.system(f'mri_binarize --i ribbon_preprefinal.nii.gz --match 1 --o ribbon_exclusion_prefinal.nii.gz')
    
    # add ventricles
    os.system(f'fslmaths ribbon_exclusion_prefinal.nii.gz -add exclusion_mask.nii.gz ribbon_exclusion_nbin.nii.gz')
    os.system(f'mri_binarize --i ribbon_exclusion_nbin.nii.gz --match 1 --o ribbon_exclusion.nii.gz')
    
    # Clean up
    os.remove(f'{path}/Diffusion/ribbon2diff_full.nii.gz')
    os.remove(f'{path}/Diffusion/left_ribbon_diff.nii.gz')
    os.remove(f'{path}/Diffusion/right_ribbon_diff.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon2diff.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon_subtr.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon_bin.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon_preprefinal.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon_exclusion_prefinal.nii.gz')
    os.remove(f'{path}/Diffusion/ribbon_exclusion_nbin.nii.gz')          
    
    
def mask_no_cerebellum(path):
    """expects custom aseg nodif mask in bedpostx dir"""
    
    os.chdir(f"{path}/Diffusion/")
    nodif_m = f"{path}/Diffusion/Diffusion.bedpostX/nodif_brain_mask.nii.gz"
    exclude_list = ['8', '47', '7', '46', '16']
    for i in exclude_list:
        os.system(f'mri_binarize --i aseg2diff.nii.gz --match {i} --o {i}.nii.gz')
    os.system(f'fslmaths {nodif_m} -sub 8.nii.gz -sub 47.nii.gz -sub 7.nii.gz -sub 46.nii.gz -sub 16.nii.gz nodif_brain_mask_no_cerebellum.nii.gz')
    for i in exclude_list:
        os.remove(f'{i}.nii.gz')
    move(f"{path}/Diffusion/nodif_brain_mask_no_cerebellum.nii.gz", f"{path}/Diffusion/Diffusion.bedpostX/")
              
      
  def cleanup(path):
    os.remove(f'{path}/Structural/MNINonLinear/ant_aseg.nii.gz')
    os.remove(f'{path}/Structural/T1w/ant_aseg_str.nii.gz')
    os.remove(f'{path}/Diffusion/aseg_ant2diff.nii.gz')
    
    
  def parcellate(i):
    """Expects to be run with multiprocessing iterating over subjects list"""
    
    path = os.path.join(SUB_DIR, i)
    
    gen_ant_aseg(path)
    reg_ant2str(path)
    resample_str2diff(path)
    binarize_ant_aseg(path)
    cleanup(path)
    
    gen_seeds(path)
    mod_seeds(path)
    new_seeds_text(path)
    to_interface(path)
    ribbon_exclusion(path)
    mask_no_cerebellum(path)
    seeds3(path)

    
    
# Here, these functions are called.  
t1_start = process_time()

pool = Pool(4, maxtasksperchild=2) 

try:
    pool.map(parcellate, sub_list) # if this does not work consider maxtaskperchild = 1
finally:
    pool.close()
    pool.join()
    
t1_stop = process_time()

print(t1_stop-t1_start)

import os
from shutil import move, rmtree, copytree, copy2
import numpy as np
import pandas as pd

def nodal_asymmetry(mat, sub):
    """Sums up nodal communication efficiency in sending and receiving, calculates difference"""
    """Concatenates with LI"""
    """Input should be an asymmetric communication efficiency matrix"""
    
    df = pd.read_csv(f"{SUB_DIR}/ant_LI.csv", names=["Subject", "LI"], index_col="Subject")
    LI = df.loc[[int(sub)], ['LI']]
    LI_number = LI.iloc[0, 0]
    
    v_out = np.zeros(96,)
    v_in = np.zeros(96,)
    delta = np.zeros(96,)
    
    # rows are outgoing!! connections, columns are incoming!! connections
    for i in range(0,96):
        v_out[i] += (np.sum(mat[i,:]) / 96)
        v_in[i] += (np.sum(mat[:,i]) / 96)
        
        delta[i] += (v_out[i] - v_in[i])    
        
    fullvec = np.append(delta, LI)
    return fullvec
  
  
  
  def nan_converter(mat):
    """converts nans from si files to 0"""
    
    newmat = np.nan_to_num(mat, copy=True, nan=0.0, posinf=0, neginf=0)
    return newmat
  
  
  
  def nav_to_enav(navmat):
    """Converts weighed nav matrices to navigation efficiency matrices"""
    
    m = np.zeros((96,96))
    for i in range(0,96):
        for j in range(0,96):
            m[i,j] = 1/navmat[i,j]       
    return m
  
  
def gen_nodal_asymmetry(walker, thr):
  """runs nodal asymmetry"""

  #loads seed for stacking
  if walker == 'diffusion':
      seedm = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/100206_Ediff_{thr}.txt", delimiter=',', usecols=range(96))

  elif walker == 'navigation':
      seedm_1 = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/100206_navigation_wei_{thr}.txt", delimiter=',', usecols=range(96))
      seedm = nav_to_enav(seedm_1) # convert to navigation efficiency

  elif walker == 'searchinformation':
      seedm_1 = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/100206_SI_{thr}.txt", delimiter=',', usecols=range(96))
      seedm = seedm_1 *-1 # convert to SI Efficiency

  # get rid of nans and infinities in matrices
  s = nan_converter(seedm)

  #get nodal asymmetry of our seed, which already appends the LI!
  seed = nodal_asymmetry(s, '100206')

  new_subs = [i for i in sub_list if i != '100206']

  #loads individual matrix, performs nodal asymmetry and stacks onto seed
  for sub in new_subs:
      if walker == 'diffusion':
          m = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/{sub}_Ediff_{thr}.txt", delimiter=',', usecols=range(96))
      elif walker == 'navigation':
          m_1 = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/{sub}_navigation_wei_{thr}.txt", delimiter=',', usecols=range(96))
          m = nav_to_enav(m_1)

      elif walker == 'searchinformation':
          m_1 = np.loadtxt(f"{SUB_DIR}/Communication/{walker}/{sub}_SI_{thr}.txt", delimiter=',', usecols=range(96))
          m = m_1 *-1 # because large SI == less efficiency

      # get rid of nans and infinities in matrices
      mat = nan_converter(m)

      new_asymmetry_vec = nodal_asymmetry(mat, sub)
      seed = np.vstack((seed, new_asymmetry_vec))

  #transforms into dataframe
  with open(f"{SUB_DIR}/Columns", 'r') as l:
      col = l.readlines()
      cols = [i.strip("\n") for i in col]

  df = pd.DataFrame(seed, columns=cols)

  df["Cluster"] = df["LI"] * 1
  df["Cluster"].where(df["Cluster"] <= -0.064318, 1, inplace=True)
  df["Cluster"].where(df["Cluster"] > -0.064318, 0, inplace=True)

  df.to_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asymmetry_{thr}.csv", index=False)
    
#Example function call
for thr in ['0', '80']:
    for walker in ['diffusion', 'navigation', 'searchinformation']:
        gen_nodal_asymmetry(walker, thr)
        
        
#####################################################################################################
        
#Test these nodal asymmetry matrices for differences from 0

from mne.stats import permutation_t_test as permute
from scipy import stats
from statsmodels.stats.multitest import multipletests as m

# Typical and atypical subjects are tested seperately for difference from 0

def load_and_split_nodal_asymmetry(walker, thr):
    """Loads nodal asy df, splits it in right and left, removes LI and Cluster and returns right, left"""
    
    df = pd.read_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asymmetry_{thr}.csv")
    right = df.loc[df["Cluster"] == 0]
    left = df.loc[df["Cluster"] == 1]
    right = right.drop(["Cluster", "LI"], axis=1)
    left = left.drop(["Cluster", "LI"], axis=1)
    return left, right
  
# Here we test left and right vs 0 and remove non-language regions
def run_permutation(walker, thr):
    """runs permutation against 0 for left and right nodal asymmetry separately"""
    """Removes non-language Nodes"""
    """Valid Expressions for walker are <'diffusion'> <'navigation'> <'searchinformation'>"""
    """Valid thr are '0', '80'."""

    #splits the nodal asy
    left, right = load_and_split_nodal_asymmetry(walker, thr)
    
    #perform permutation testing - which delta in send-receive is significantly different from 0?
    T_obs_l, p_vals_l, H0_l = permute(left, n_permutations=10000, tail=0, n_jobs=2)
    T_obs_r, p_vals_r, H0_r = permute(right, n_permutations=10000, tail=0, n_jobs=2)
    
    # Here, we are only interested in language ROI.
    non_language_indices = [3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                            30,31,32,36,40,42,43,45,46,47,48,49,50,55,57,59,60,61,62,63,64,65,
                            66,67,68,69,70,72,73,74,78,82,84,85,87,88,89,90,91,92,95]
    
    #delete non language
    left_np = left.to_numpy()
    l_del_1 = np.delete(left_np, non_language_indices, 1)
    right_np = right.to_numpy()
    r_del_1 = np.delete(right_np, non_language_indices, 1)
    
    p_l = np.delete(p_vals_l, non_language_indices)
    p_r = np.delete(p_vals_r, non_language_indices)
    T_obs_l = np.delete(T_obs_l.to_numpy(), non_language_indices)
    T_obs_r = np.delete(T_obs_r.to_numpy(), non_language_indices)
    
    # our arrays back to dataframes
    with open(f"{SUB_DIR}/language_regions", 'r') as l: #txt file containing language ROI names in the correct order.
        col = l.readlines()
        cols = [i.strip("\n") for i in col]    
    left_subjects = pd.DataFrame(l_del_1, columns = cols)
    right_subjects = pd.DataFrame(r_del_1, columns = cols)
    
    return p_l, T_obs_l, p_r, T_obs_r, left_subjects, right_subjects
  
# Example Function call
for thresh in ['0', '80']:
    for walker in ['diffusion', 'navigation', 'searchinformation']:
        p_vals_l, T_obs_l, p_vals_r, T_obs_r, left_masked, right_masked = run_permutation(walker, thresh)
        
        feature_cols = left_masked.columns
        np.set_printoptions(suppress=True)
        test_dict = dict(zip(feature_cols, zip(p_vals_l[:], T_obs_l[:], p_vals_r[:], T_obs_r[:])))
        test_df = pd.DataFrame.from_dict(test_dict, orient='index')
        test_df.rename(columns = {0:'p_left', 1:'T_left', 2:'p_right', 3:'T_right'}, inplace = True)
        
        test_df.to_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asymmetry_sign_ROI_{thresh}.csv")
        left_masked.to_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asy_left_masked_{walker}_{thresh}.csv", index=False)
        right_masked.to_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asy_right_masked_{walker}_{thresh}.csv", index=False)

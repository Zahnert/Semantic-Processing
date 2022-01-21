import os
import pandas as pd
import numpy as np


def fetch_brainvol(sub):
    """Gets brainvol of the individual subject. Make sure {sub} is int in this instance"""
    
    # read df of unrestricted HCP data
    df = pd.read_csv(f"/path/to/unrestricted_data.csv", index_col='Subject')
    df = df['FS_BrainSeg_Vol_No_Vent']
    brainvol = df.loc[[int(sub)]]
    vol = brainvol.iloc[0]
    
    return vol
  
  
def calculate_tractstats(df, brainvol, sub):
    """Takes individual xtract_stats df and brain volume as input, calculates desired measures and returns an array of them as well as a column list"""

    # first calculate LI
    af_vol_LI = (df.iloc[0,1] - df.iloc[1,1])/ (df.iloc[0,1] + df.iloc[1,1])
    ilf_vol_LI = (df.iloc[2,1] - df.iloc[3,1])/ (df.iloc[2,1] + df.iloc[3,1])
    uf_vol_LI = (df.iloc[4,1] - df.iloc[5,1])/ (df.iloc[4,1] + df.iloc[5,1])
    
    # now single volume measures corrected by brain vol
    af_l_vol = (df.iloc[0,1] / brainvol)
    af_r_vol = (df.iloc[1,1] / brainvol)
    
    ilf_l_vol = (df.iloc[2,1] / brainvol)
    ilf_r_vol = (df.iloc[3,1] / brainvol)
    
    uf_l_vol = (df.iloc[4,1] / brainvol)
    uf_r_vol = (df.iloc[5,1] / brainvol)
    
    FA = df.iloc[:,9].values.tolist()
    
    measures = [af_vol_LI, ilf_vol_LI, uf_vol_LI, af_l_vol, af_r_vol, ilf_l_vol, ilf_r_vol, uf_l_vol, uf_r_vol]
    measures.extend(FA)
    m = np.asarray(measures)
    
    LI_df = pd.read_csv("/path/to/ant_LI.csv", names=['Subject', 'LI'], index_col='Subject')
    LI = LI_df.loc[[int(sub)], ['LI']]
    LI_number = LI.iloc[0,0]
    
    m = np.append(m, LI_number)
    
    columns = ['af_vol_LI', 'ilf_vol_LI', 'uf_vol_LI', 'af_l_vol', 'af_r_vol', 'ilf_l_vol', 'ilf_r_vol', 'uf_l_vol', 'uf_r_vol', 'af_l_FA', 'af_r_FA', 'ilf_l_FA', 'ilf_r_FA', 'uf_l_FA', 'uf_r_FA', 'LI']
    
    return m, columns
  
  
  
  
  def tractstats_df(sub_list):
    """Creates a summary-df of our tract stats"""
    
    seed = pd.read_csv(f"{SUB_DIR}/100206/Diffusion/XTRACT/stats.csv")
    s_vol = fetch_brainvol(100206)
    seed_vec, cols = calculate_tractstats(seed, s_vol, 100206)
    
    # in 155231 and 307127, the left uf was not extracted. One is clear right lateralized, the other left
    new_subs = [i for i in sub_list if i not in ['100206', '155231', '307127']]
    
    for sub in new_subs:
        
        raw_stats = pd.read_csv(f"{SUB_DIR}/{sub}/Diffusion/XTRACT/stats.csv")
        vol = fetch_brainvol(sub)
        stats_vec, columns = calculate_tractstats(raw_stats, vol, sub)
        
        seed_vec = np.vstack((seed_vec, stats_vec))
        
    df = pd.DataFrame(seed_vec, columns=cols)
    df['Cluster'] = df['LI'] * 1
    df['Cluster'].where(df['Cluster'] <= -0.064318, 1, inplace=True)
    df['Cluster'].where(df['Cluster'] >= -0.064318, 0, inplace=True)
    
    df.to_csv('/path/to/xtract_stats.csv', index=False)
    

    
# Function call
tractstats_df(sub_list)

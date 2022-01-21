import os
import numpy as np
import pandas as pd
import bct


SUB_DIR = "/path/to/matrices/"

def normalize_by_waytotal(matrix, waytotal):
    """Divides each subjects matrix row wise by the respective waytotal"""
    
    m = (matrix / waytotal.reshape(-1,1)) * 100
    return m
  

def gen_rows(matrix, row):
    """adds ROI a -> all + all -> a and generates thus a new symmetricized row for a."""
    # if either of the added values is 0, we do not consider the connection valid and set it to 0.
    new_row = []
    for i in range(0, 96): # 96 because this statement is exclusive of the last value.
        row_val = matrix[row, i]
        col_val = matrix[i, row]
        if row_val and col_val != 0:
            new_val = (row_val + col_val)/2
        else:
            new_val = row_val * col_val
        new_row.append(new_val)
    new_row_arr = np.array(new_row)
    return(new_row_arr)

  
def add_rows(matrix, i, nr):
    """Takes in the empty matrix, its number of row to choose from and the newly generated row"""
    matrix[i, :] += nr
    return(matrix)
  
  
def threshold(matrix, thr):
    """assigns each edge a rank and deletes it if it's rank is below the given threshold"""
    threshold = np.percentile(matrix, int(thr))
    for i in range(0, 96):
        for j in range(0, 96):
            if matrix[i, j] < threshold:
                matrix[i, j] *= 0
    return(matrix)
  
  
def binarize(matrix):
    """takes matrix, returns binarized copy"""
    m = matrix.copy()
    for i in range(0,96):
        for j in range(0,96):
            if m[i, j]>0:
                m[i,j]=1
    return m
  

  
def get_metric_mats(metric='degree', l_thr=60, u_thr=81):
    """creates normalized matrices between 20-40% density"""
    """calculates degree/bc for each density and saves an array of these degrees per subject"""
    """default are 60th and 80th percentile = 20-40% density"""
    
    df = pd.read_csv(f"{SUB_DIR}/ant_LI.csv", names=["Subject", "LI"], index_col="Subject")
    
    # normalize and symmetricize subject wise matrices
    
    for sub in sub_list:
        
        x = np.loadtxt(f"{SUB_DIR}/raw_mats/{sub}_mat") # die raw-mat
        y = np.loadtxt(f"{SUB_DIR}/raw_mats/{sub}_waytotal") # waytotal to divide by
        z = normalize_by_waytotal(x, y)
        
        m = np.zeros([96, 96]) # generate empty matrix
        
        for i in range(0,96):
            nr = gen_rows(z, i) # newly generated row number i
            add_rows(m, i, nr)
        
        thresh_list = [i for i in range(l_thr, u_thr)]
        
        sub_vec = np.zeros([0,97])
        
        if metric == 'degree':
        
            for thr in thresh_list:

                LI = df.loc[[int(sub)], ['LI']]
                LI_number = LI.iloc[0, 0]

                #threshold matrix
                m_thresh = threshold(m, thr)

                # binarize
                m_bin = binarize(m_thresh)

                degree = bct.degrees_und(m_bin)
                degree = np.append(degree, LI)

                sub_vec = np.vstack((sub_vec, degree))

            np.savetxt(f"{SUB_DIR}/degree_auc/{sub}_degrees.txt", sub_vec)
            
        elif metric == 'bc':
            
            for thr in thresh_list:
            
                LI = df.loc[[int(sub)], ['LI']]
                LI_number = LI.iloc[0, 0]

                #threshold matrix
                m_thresh = threshold(m, thr)

                # binarize
                m_bin = binarize(m_thresh)

                bc = bct.betweenness_bin(m_bin)
                bc = np.append(bc, LI)

                sub_vec = np.vstack((sub_vec, bc))
            
            np.savetxt(f"{SUB_DIR}/bc_auc/{sub}_bcs.txt", sub_vec)
            
        else:
            print('parse metric to be calculated (either degree or bc)')
            

            
def integrate_metrics(metric='degree'):
    """Each column of the degree/bc-matrix represents a node, each row represents a network density
       Here we integrate the columns to receive an AUC.
    """

    from scipy.integrate import simps
    
    df = pd.read_csv(f"{SUB_DIR}/ant_LI.csv", names=["Subject", "LI"], index_col="Subject")

    for sub in sub_list:
        
        d = np.loadtxt(f"{SUB_DIR}/{metric}_auc/{sub}_{metric}s.txt")
        
        LI = df.loc[[int(sub)], ['LI']]
        LI_number = LI.iloc[0, 0]
        
        l = []
        
        for i in range(0,96):
            
            node = d[:,i]
            
            node_auc = simps(node, dx=1)
            
            l.append(node_auc)
            
        l.append(LI_number)
            
        np.savetxt(f"{SUB_DIR}/{metric}_auc/aucs/{sub}_aucs.txt", l)
        
        
        
        
def stack_aucs(metric='degree'):
    """Stacks vectors to n x (metric+LI) matrix and appends categorical LI"""

        
    seedvec = np.loadtxt(f"{SUB_DIR}/{metric}_auc/aucs/100206_aucs.txt")
    new_subs = [i for i in sub_list if i != '100206']

    for sub in new_subs:
        new_vec = np.loadtxt(f"{SUB_DIR}/{metric}_auc/aucs/{sub}_aucs.txt")
        seedvec = np.vstack((seedvec, new_vec))

    with open(f"{SUB_DIR}/Columns", 'r') as l:
        col = l.readlines()
        cols = [i.strip("\n") for i in col]

    df = pd.DataFrame(seedvec, columns=cols)

    df["Cluster"] = df["LI"] * 1
    df["Cluster"].where(df["Cluster"] <= -0.064318, 1, inplace=True)
    df["Cluster"].where(df["Cluster"] > -0.064318, 0, inplace=True)

    df.to_csv(f"{SUB_DIR}/{metric}_auc/aucs/auc_df.csv", index=False)

    
    
    
######## Functions for matrix preprocessing for computation of communication models ##############

def normalize_for_inversion(matrix):
    """circumvents turning largest weight to 0 upon inversion by adding +1 to the denominator of bct.norm"""
    m = matrix.copy()
    indices = np.where(m == m.max())
    maxval = m[indices][0]
    new_mat = m / (maxval + 1)
    return new_mat
  
# create connectivity lengths matrix

m_norm = normalize_for_inversion(matrix)

m_invert= -np.log10(m_norm)  
m_invert[~np.isfinite(m_invert)] = 0

import pandas as pd
from scipy import stats
import numpy as np
import os
from statsmodels.stats.multitest import multipletests as m
from numpy import mean
from numpy import var
from math import sqrt


# function to calculate Cohen's d for independent samples
# This code is adopted from machinelearningmastery.com
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s
  
## Example for node degree
df = pd.read_csv(f"{SUB_DIR}/XY/degree.csv")

# generate groups for testing
typical = df.loc[df['Cluster'] == 1]
atypical = df.loc[df['Cluster'] == 0]
typical = typical.drop(['LI', 'Cluster'], axis=1)
atypical = atypical.drop(['LI', 'Cluster'], axis=1)

# generate results k for permutation t-test with 100k permutations. Implemented as a loop, as this is expensive in terms of RAM if loaded all at once.
k = np.zeros(2,)
for i in range(0,96):
    result = stats.ttest_ind(typical.iloc[:,i], atypical.iloc[:,i], equal_var = False, permutations=100000)
    res = np.array(result)
    k = np.vstack((k, res))
k = np.delete(k, 0, 0)

# Compute Cohen's d and create a dataframe with all our stats
cohend_vec = np.zeros(96,)
for i in range(0,96):    
    d1 = typical.iloc[:,i].values
    d2 = atypical.iloc[:,i].values
    cohen = cohend(d1, d2)
    cohend_vec[i] += cohen
    
feature_cols = typical.columns
np.set_printoptions(suppress=True)

ttest_dict = dict(zip(feature_cols, zip(k[:, 0], k[:, 1], cohend_vec)))
ttest_df = pd.DataFrame.from_dict(ttest_dict, orient='index')
ttest_df.rename(columns = {0:'t_stat', 1:'p_val', 2:'cohen_d'}, inplace = True)
ttest_df.sort_values(by=['p_val'], ascending=True).head(30)

# Correction for multiple comparisons, fdr-benjamini-hochberg
m(ttest_df['p_val'], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=True)



________________________________________________________________________________________________________

### Here we load and test nodal asymmetry matrices, which is the same procedure

def load_masked_asy(walker, thr):
    """Loads and returns left and right lateralized subs seperately after removing non-language nodes.
    'walker' just describes the communication model here."""
    
    l = pd.read_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asy_left_masked_{walker}_{thr}.csv")
    r= pd.read_csv(f"{SUB_DIR}/Communication/{walker}/nodal_asy_right_masked_{walker}_{thr}.csv")
    
    return l, r
  
# load example matrix
l, r = load_masked_asy('diffusion', '0')

# identify ROI that are not different from 0 in both groups and thus to be excluded
df = pd.read_csv(f"{SUB_DIR}/Communication/diffusion/nodal_asymmetry_sign_ROI_0.csv")
df.sort_values(by=['p_left'], ascending=False).head(10)

# drop roi, that are not different from 0
l.drop(['ctx-rh-parstriangularis'], axis=1, inplace=True)
r.drop(['ctx-rh-parstriangularis'], axis=1, inplace=True)

# same permutation test as for node degree
k = np.zeros(2,)
for i in range(0, l.shape[1]):
    result = stats.ttest_ind(l.iloc[:,i], r.iloc[:,i], equal_var = False, permutations=100000)
    res = np.array(result)
    k = np.vstack((k, res))
k = np.delete(k, 0, 0)

cohend_vec = np.zeros(l.shape[1],)
for i in range(0,l.shape[1]):
    d1 = l.iloc[:,i].values
    d2 = r.iloc[:,i].values
    cohen = cohend(d1, d2)
    cohend_vec[i] += cohen
    
feature_cols = l.columns
np.set_printoptions(suppress=True)
ttest_dict = dict(zip(feature_cols, zip(k[:, 0], k[:, 1], cohend_vec)))
perm_test_df = pd.DataFrame.from_dict(ttest_dict, orient='index')
perm_test_df.rename(columns = {0:'t_stat', 1:'p_val', 2:'cohen_d'}, inplace = True)
perm_test_df.sort_values(by=['p_val'], ascending=True).head(50)

# Correction for multiple comparisons, fdr-benjamini-hochberg
m(perm_test_df['p_val'], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=True)

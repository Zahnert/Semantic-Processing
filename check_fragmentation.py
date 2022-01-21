import os
import numpy as np
import pandas as pd
import bct


def get_fragmentation_thr(thr):
    """Expects a threshold as input, output is a) number of subjects with >1 components, b) list of these subjects."""
    
    components = []
    larger_1 = []
    frag = []
    for sub in sub_list:

        os.chdir(f"{SUB_DIR}/thr{thr}/")

        m = np.loadtxt(f'{SUB_DIR}/thr{thr}/{sub}_m_thresh{thr}.txt')
        v_comp, n_comp = bct.get_components(m)
        
        components.append(len(n_comp))
        if len(n_comp) > 1:
            larger_1.append(len(n_comp))
            frag.append(sub)

    print(f"In {len(larger_1)} Subjects, the network fragments at a threshold of {thr}.")
    return larger_1, 

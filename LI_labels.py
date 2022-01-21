import os
import numpy as np
import pandas as pd
from shutil import copy2, copytree, rmtree
from sklearn.cluster import KMeans
from jenkspy import JenksNaturalBreaks as jnb


SUB_DIR = '/home/felix/language/all_matrices/'

ant = pd.read_csv(f"{SUB_DIR}/ant_LI.csv", names=["Subject", "LI"], index_col="Subject")
X = np.array(ant["LI"])

kmeans = KMeans(n_clusters=3, random_state=42).fit(X.reshape(-1, 1))
labels = kmeans.labels_

# Create column with labels per subject according to kmeans
ant["Cluster"] = labels
#Indices just for plotting
ant["Ind"] = [i for i in range(0,1038)]


# Here we just change the cluster labels from 0,1,2 to -1,0,1 for adequate plotting
ant["Cluster"].where(ant["Cluster"] < 2, 3, inplace=True) ## set 2, i.e. bilateral, to 3
ant["Cluster"].where(ant["Cluster"] > 0, 2, inplace=True) ## set 0 (left) to 2
ant["Cluster"].where(ant["Cluster"] != 1, -1, inplace=True) ## set 1 (right) to -1
ant["Cluster"].where(ant["Cluster"] != 3, 0, inplace=True) ## bilateral to 0
ant["Cluster"].where(ant["Cluster"] != 2, 1, inplace=True) ## left to 1

# Here we generate Fig. 1
from matplotlib.pyplot import plotting as plt
%matplotlib inline
x = ant.plot(kind="scatter", x="Ind", y="LI", alpha=0.6, c="Cluster", cmap="jet", colorbar=True, figsize =(10,7))

# locate max value of right lateralized subjects.
j = ant.loc[li['Cluster'] == -1]
j.max()

# Jenks natural breaks for comparison
j = jnb()
j.fit(X)
print(j.inner_breaks_)

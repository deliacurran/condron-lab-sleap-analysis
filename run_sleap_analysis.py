# This code uses the SLEAP AI library for multi-animal pose tracking.
# Please cite the following work when using SLEAP:
# Talmo D. Pereira et al., "SLEAP: A deep learning system for multi-animal pose tracking," Nature Methods, 2022. 
# DOI: https://doi.org/10.1038/s41592-021-01210-2

import h5py
import numpy as np

## LOAD DATA ## 
filename = '/Users/delia/Desktop/CONDRON LAB/1 Node Prediction/Analysis/labels.onenode.final.004_s8-movie copy.analysis.h5'

with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        num_frames, num_nodes, _, num_instances = locations.shape
        node_names = [n.decode() for n in f["node_names"][:]]
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:]

print("===filename===")
print(filename)
print("===HDF5 datasets===")
print(dset_names)
print("===locations data shape===")
print(locations.shape)
print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print(occupancy_matrix.shape)
print(tracks_matrix.shape)
    
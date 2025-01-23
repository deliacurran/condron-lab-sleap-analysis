# This code uses the SLEAP AI library for multi-animal pose tracking.
# Please cite the following work when using SLEAP:
# Talmo D. Pereira et al., "SLEAP: A deep learning system for multi-animal pose tracking," Nature Methods, 2022. 
# DOI: https://doi.org/10.1038/s41592-021-01210-2

import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

## LOAD DATA ## 
filename = '/Users/delia/Desktop/UVA/CONDRON LAB/1 Node Prediction/Analysis/labels.onenode.final.004_s8-movie copy.analysis.h5'
threshold = 50

with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        num_frames, num_nodes, _, num_instances = locations.shape
        node_names = [n.decode() for n in f["node_names"][:]]
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:]

# print("===filename===")
# print(filename)
# print("===HDF5 datasets===")
# print(dset_names)
# print("===locations data shape===")
# print(locations.shape)
# print("===nodes===")
# for i, name in enumerate(node_names):
#     print(f"{i}: {name}")
# print(occupancy_matrix.shape)
# print(tracks_matrix.shape)

## INTERPOLATE MISSING VALUES ##
## added this from: https://sleap.ai/notebooks/Analysis_examples.html
def fill_missing(array, kind="linear"):
    def interpolate_1d(y):
        x = np.where(~np.isnan(y))[0]
        if len(x) == 0:
            return np.zeros_like(y)  # Default to 0 if all values are NaN
        if len(x) == 1:
            return np.full_like(y, y[x[0]])  # Fill with the single known value
        interp_func = interp1d(
            x, y[x], kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        return interp_func(np.arange(len(y)))
    return np.apply_along_axis(interpolate_1d, -1, array)

print(f"Initial NaNs: {np.isnan(locations).sum()}")
locations = fill_missing(locations)
print(f"NaNs after fill_missing: {np.isnan(locations).sum()}")

## idea:
    # get (x, y) coordinates of each larvae
    # for each given frame, find the larvae instance closest to the current larvae (save distance)
    # another function returns average of distances to closest larvae
    # smaller distance on average = more likely clustering (close to others)
    # larger distance on average = not clustering
    # assign a categorical group based on average distance

def get_coordinates(loc, frame, node, instance):
        """
        Returns the (x, y) coordinates for a larva's node in the given frame.
        """
        x, y = loc[frame, node, :, instance]
        return (x, y)

def get_closest_instance_in_frame(frame, node, instance):
    """
    Returns the distance to the closest other larva.
    This is the minimum distance to any other larva instance in the given frame.
    """
    (x, y) = get_coordinates(locations, frame, node, instance)
    min_distance = float('inf')
    for other_instance in range(num_instances):
        if other_instance != instance:
            (other_x, other_y) = get_coordinates(locations, frame, node, other_instance)
            distance = np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def get_average_distance_to_larvae(instance, node):
    """
    Computes the average distance between an individual larvae and the closest other larvae across all video frames.
    Uses the spiracle node to compute distances.
    """
    distance_sum = 0
    for frame in range(num_frames):
        distance_sum += get_closest_instance_in_frame(frame, node, instance)
    return (distance_sum / num_frames)

def run_clustering_analysis(threshold):
    """
    Classify all larvae into clustering or not clustering.
    """
    ## CHANGE: need to change to keep track of which classification goes with which larva
    # classifications = []
    # for instance in range(num_instances):
    #     average_distance = get_average_distance_to_larvae(instance, 0)
    #     print(f"Instance {instance}: Average Distance = {average_distance}")
    #     if average_distance < threshold:
    #          classifications.append(1)     # represents clustering
    #     else:
    #          classifications.append(0)     # represents not clustering
    # return classifications
    average_distances = []
    for instance in range(num_instances):
        average_distance = get_average_distance_to_larvae(instance, 0)
        average_distances.append(average_distance)
    return average_distances

# larvae_classifications = run_clustering_analysis(threshold)
# print(larvae_classifications)

# create a graph
def generate_graphs(average_distances):
    # Define thresholds to test
    # Compute clustering amounts as inversely proportional to average distances
    clustering_amounts = [max(threshold - avg, 0) for avg in average_distances]

    # Plot results
    instances = np.arange(len(average_distances))  # Instance indices
    plt.figure(figsize=(10, 6))

    # Bar chart of clustering amounts
    bars = plt.bar(instances, clustering_amounts, color='skyblue', edgecolor='black')

    # Highlight instances that are clustering
    for i, bar in enumerate(bars):
        if average_distances[i] < threshold:
            bar.set_color('green')
            bar.set_edgecolor('darkgreen')

    # Add labels, title, and threshold line
    plt.axhline(y=0, color='gray', linestyle='--', label=f"Threshold: {threshold}")
    plt.title("Clustering Amounts by Instance", fontsize=14)
    plt.xlabel("Instance Index", fontsize=12)
    plt.ylabel("Clustering Amount (Inverse of Distance)", fontsize=12)
    plt.xticks(instances, [f"Instance {i}" for i in instances])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()

generate_graphs(run_clustering_analysis(threshold))

# save the results to a file
# with open("classifications.txt", "w") as f:
#     for classification in larvae_classifications:
#         f.write(f"{classification}\n")



# def get_coordinates(loc):
#     for frame in range(num_frames):
#         for instance in range(num_instances):
#             for node in range(num_nodes):
#                 x, y = loc[frame, node, :, instance]  # Get the (x, y) coordinates
#                 print(f"Frame {frame}, Instance {instance}, Node {node}: (x, y) = ({x}, {y})")

# print(get_coordinates(locations))

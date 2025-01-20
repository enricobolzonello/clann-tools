import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import h5py
from tqdm import tqdm
from argparse import ArgumentParser
from utils.distance import compute_distance
import os

from joblib import Memory
from utils.utils import is_valid_file

def greedy_distances(hdf5_filepath, distance_name):
    distances = defaultdict(int)

    with h5py.File(hdf5_filepath, "r") as file:
        data = file["train"]
        vectors = data[:]

    num_vectors = vectors.shape[0]
    for i in tqdm(range(num_vectors)):
        for j in range(i + 1, num_vectors):
            distance = compute_distance(distance_name, vectors[i], vectors[j])
            distances[distance] += 1
    
    return distances

def main():
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    # Setup cache
    location = os.path.dirname(os.path.abspath(__file__)) + '/cachedir'
    memory = Memory(location, verbose=0)

    # Parse arguments
    parser = ArgumentParser(
        prog="visualizer.py",
        description="Visualizes the relationship between hyperparameters and collision statistics from a CSV file, generating corresponding plots.",
    )
    parser.add_argument("-input", dest="csv_filename", required=True,
                        help="Path to the CSV file containing the hyperparameters and collision data (e.g., k, l, bucket width, distance metric, number of collisions, unique pairs).", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-dataset", dest="dataset_filename", required=True,
                        help="Path to the HDF5 dataset used for the ground truth calculation", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-distance", dest="distance_type", required=True,
                        help="Type of distance metric as specified by the dataset", choices=['jaccard', 'euclidean', 'angular'])
    parser.add_argument("-output", dest="base_name", required=False,
                        help="Output base_name", metavar="BASE_NAME", 
                        default="", type=str)
    args = parser.parse_args()

    # Compute ground truth and cache it
    ground_truth_cached = memory.cache(greedy_distances)
    ground_truth_distances = ground_truth_cached(args.dataset_filename, args.distance_type)
    
    # Read the CSV data
    data = defaultdict(lambda: defaultdict(int))
    with open(args.csv_filename, "r") as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader) 
        
        for row in csv_reader:
            k, l, bucket_width, num_centers, distance, num_collisions, unique_pairs = int(row[0]), int(row[1]), float(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[5])
            key = (k, l, bucket_width, num_centers)
            data[key][distance] += unique_pairs 
    
    # Histogram plots for each combination of hyperparameters
    print("generating histograms")
    for (k, l, bucket_width, num_centers), distance_counts in tqdm(data.items()):
        distances = sorted(distance_counts.keys())
        counts = [(distance_counts[dist] / ground_truth_distances[dist]) * 100 for dist in distances]
        
        # Ensure a reasonable range for single-entry histograms
        if len(distances) == 1:
            single_dist = distances[0]
            distances = [single_dist - 1, single_dist, single_dist + 1]
            counts = [0, counts[0], 0]
        
        # Plot histogram for this combination of hyperparameters
        plt.figure(figsize=(10, 6))
        bar_width = max(0.4, 0.8 / len(distances))  # Dynamic bar width
        plt.bar(distances, counts, width=bar_width, alpha=0.6, label=f'k={k}, l={l}, bucket_w={bucket_width}, centers={num_centers}')
        plt.xlabel("Distance")
        plt.ylabel("Percentage of Points Mapped")
        plt.ylim(0, max(counts) * 1.1)
        plt.title(f"Percentage of Points Mapped by Distance\n(k={k}, l={l}, bucket_width={bucket_width}, centers={num_centers})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}{args.base_name}histogram_k{k}_l{l}_bw{bucket_width}_c{num_centers}.png")
        plt.close()
        
    print("All histograms generated and saved.")

if __name__ == "__main__":
    main()

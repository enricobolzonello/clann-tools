import numpy as np
import h5py
import math
from tqdm import tqdm
import argparse

from utils.utils import is_valid_file

def normalize_rows(array):
    """ Normalizes each row of the array to unit length. """
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return array / norms

def euclidean_to_angular(euclidean_distances, train, queries):
    """
    Converts Euclidean distances to angular distances for each query.
    Euclidean distances are based on non-normalized vectors, so we 
    normalize the vectors first before calculating angular distances.
    """
    num_queries = queries.shape[0]
    num_neighbors = euclidean_distances.shape[1]

    angular_distances = np.zeros((num_queries, num_neighbors))

    for i in tqdm(range(num_queries)):
        query = queries[i]
        for j in range(num_neighbors):
            neighbor_idx = j
            neighbor = train[neighbor_idx]
            
            # Normalize both vectors
            query_normalized = query / np.linalg.norm(query)
            neighbor_normalized = neighbor / np.linalg.norm(neighbor)
            
            # Compute cosine similarity
            cosine_similarity = np.dot(query_normalized, neighbor_normalized)
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            
            # Convert to angular distance
            angular_distances[i, j] = math.acos(cosine_similarity)

    return angular_distances

def main(input_hdf5, output_hdf5):
    """ Converts Euclidean ground truth distances to angular distances and saves new HDF5. """
    with h5py.File(input_hdf5, 'r') as infile:
        train = infile['train'][:]
        queries = infile['test'][:]
        euclidean_distances = infile['distances'][:]
        neighbors = infile['neighbors'][:]

        train_normalized = normalize_rows(train)
        queries_normalized = normalize_rows(queries)

        angular_distances = euclidean_to_angular(euclidean_distances, train_normalized, queries_normalized)

    # Save the new HDF5 file with angular distances
    with h5py.File(output_hdf5, 'w') as outfile:
        outfile.create_dataset('train', data=train)
        outfile.create_dataset('test', data=queries)
        outfile.create_dataset('distances', data=angular_distances)
        outfile.create_dataset('neighbors', data=neighbors)

    print(f"Successfully saved new HDF5 file with angular distances to '{output_hdf5}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-input", dest="input_hdf5", required=True, help="", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-output", dest="output_hdf5", required=True, help="")
    args = parser.parse_args()

    main(args.input_hdf5, args.output_hdf5)

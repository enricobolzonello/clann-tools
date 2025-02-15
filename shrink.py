from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from utils.utils import is_valid_file
import re

def shrink_dataset(filepath, d):
    filepath = Path(filepath)
    filename = filepath.name

    # Check if filename contains a number surrounded by "-"
    if re.search(r"-(\d+)-", filename):
        new_filename = re.sub(r"-(\d+)-", f"-{d}-", filename)
    else:
        new_filename = re.sub(r"\.h5$", f"-{d}.h5", filename)

    opath = filepath.parent / new_filename

    if not opath.is_file():
        with h5py.File(filepath) as hfp:
            data = hfp["train"][:,:d]
            queries = hfp["test"][:,:d]

        data = data / np.linalg.norm(data, axis=1)[:,np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=1)[:,np.newaxis]

        distances = []
        for i in tqdm(range(10000)):
            ds = np.sort(np.linalg.norm(data - queries[i], axis=1))
            distances.append(np.array(ds[:100]))
            ds = None

        distances = np.array(distances)
        with h5py.File(opath, "w") as hfp:
            hfp["train"] = data
            hfp["test"] = queries
            hfp["distances"] = distances

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="shrink.py",
        description="",
    )
    parser.add_argument("-f", dest="filepath", required=True,
                        help="", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-dim", dest="d", required=True,
                        help="", type=int)
    args = parser.parse_args()

    shrink_dataset(args.filepath, args.d)
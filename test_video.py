import os
import pickle
import numpy as np

length_dir = "./data/pkummd/Length"

max_len = 0
max_file = None

for fname in sorted(os.listdir(length_dir)):
    if not fname.endswith(".pkl"):
        continue

    path = os.path.join(length_dir, fname)

    # always load pickle in binary mode
    with open(path, "rb") as f:
        length_arr = pickle.load(f)   # e.g. array([182, 182, ..., 174])

    # get max from this file
    local_max = np.max(length_arr)

    # update global max
    if local_max > max_len:
        max_len = local_max
        max_file = fname

print("Global maximum length:", max_len)
print("Found in file:", max_file)

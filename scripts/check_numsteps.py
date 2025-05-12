import os
import fnmatch
import h5py

HDF5_DIR = "/home/ivan/Kuavo_RDT/data/Kuavo"  
file_paths = []
for root, _, files in os.walk(HDF5_DIR):
    for filename in fnmatch.filter(files, '*.hdf5'):
        file_path = os.path.join(root, filename)
        file_paths.append(file_path)

print(f"Found {len(file_paths)} HDF5 files.\n")

for path in file_paths:
    try:
        with h5py.File(path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            print(f"{os.path.basename(path)} → num_steps = {num_steps}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

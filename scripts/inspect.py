import json
import numpy as np

with open("grid_dataset_neurips_small.json") as f:
    records = json.load(f)

print(f"Total records: {len(records)}")
print(f"Label distribution: {dict(zip(*np.unique([r['label'] for r in records], return_counts=True)))}")
print(f"rho shape: {len(records[0]['rho'])}")         # expect 59
print(f"load_p shape: {len(records[0]['load_p'])}")   # expect 37
print(f"gen_p shape: {len(records[0]['gen_p'])}")     # expect 22
print(f"rho_max across dataset: {max(max(r['rho']) for r in records):.3f}")
print(f"fault_loc None count: {sum(1 for r in records if r['fault_loc'] is None)}")
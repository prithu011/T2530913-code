import json
import numpy as np
import sys
from collections import Counter

def main():
    # Use command line argument or default filename
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/grid_dataset_neurips2020.jsonl"

    labels = []
    rho_lens = []
    load_p_lens = []
    gen_p_lens = []
    rho_max = 0.0
    fault_loc_none = 0
    total_records = 0

    print(f"Inspecting {filename}...")
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            total_records += 1
            labels.append(r['label'])
            rho_lens.append(len(r['rho']))
            load_p_lens.append(len(r['load_p']))
            gen_p_lens.append(len(r['gen_p']))
            rho_max = max(rho_max, max(r['rho']))
            if r.get('fault_loc') is None:
                fault_loc_none += 1
            
            if i == 0:
                print(f"Example record keys: {list(r.keys())}")

    print(f"\nTotal records: {total_records}")
    print(f"Label distribution: {dict(Counter(labels))}")
    print(f"Unique rho shapes: {set(rho_lens)}")
    print(f"Unique load_p shapes: {set(load_p_lens)}")
    print(f"Unique gen_p shapes: {set(gen_p_lens)}")
    print(f"rho_max across dataset: {rho_max:.3f}")
    print(f"fault_loc None count: {fault_loc_none}")

if __name__ == "__main__":
    main()

import pickle
import torch

# Load the dataset
data = pickle.load(open('../data/extended_remove_h_centered_v2.pkl', 'rb'))
sample = data[0]

print(f"--- Sample 0 Inspection ---")
print(f"Query Atom Types:\n{sample.atom_type}")
print(f"\nQuery Coordinates (pos):\n{sample.pos}")
print(f"\nReference Coordinates (pos_r):\n{sample.pos_r}")
print(f"\nQuery Edge Index:\n{sample.edge_index}")

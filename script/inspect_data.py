import pickle
import torch

file_path = '../data/extended_remove_h_centered_v2.pkl'

def inspect_pickle(path):
    print(f"Loading {path}...")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded.")
        print(f"Type of data: {type(data)}")
        
        # If it's a list or dict, get length and inspect the first element
        if isinstance(data, list):
            print(f"Number of items: {len(data)}")
            if len(data) > 0:
                first = data[0]
                print(f"Type of first item: {type(first)}")
                # If it's a torch_geometric Data object, print its structure
                if hasattr(first, 'keys'):
                    print(f"Keys/Attributes in Data object: {first.keys()}")
                    print(f"Example atom_type shape: {first.atom_type.shape}")
        
        elif isinstance(data, dict):
            print(f"Dictionary keys: {data.keys()}")
            
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_pickle(file_path)

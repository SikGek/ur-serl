import pickle
import os

def clean_serl_pickle(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    original_len = len(data)
    cleaned_data = []

    for i, trans in enumerate(data):
        obs = trans.get('observations', {})
        
        # Check both the root of obs AND the nested 'images' dict
        has_shoulder = False
        if 'shoulder' in obs:
            has_shoulder = True
        elif 'images' in obs and 'shoulder' in obs['images']:
            has_shoulder = True
            
        if has_shoulder:
            cleaned_data.append(trans)
        else:
            print(f"  > Transition {i} in {os.path.basename(file_path)} is missing 'shoulder' key. Dropping.")

    new_len = len(cleaned_data)
    
    # Save back to the folder (I'm adding a '_cleaned' suffix to be safe)
    new_file_path = file_path.replace(".pkl", "_cleaned.pkl")
    with open(new_file_path, 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"Finished: {os.path.basename(file_path)}")
    print(f"  Original Length: {original_len}")
    print(f"  New Length:      {new_len}")
    print(f"  Saved to:        {new_file_path}\n")

if __name__ == "__main__":
    data_dir = "classifier_data"
    files_to_clean = [
        "door_opening_250_success_images_2026-02-23_13-57-21.pkl",
        "door_opening_failure_images_2026-02-23_13-57-21.pkl"
    ]

    for filename in files_to_clean:
        full_path = os.path.join(data_dir, filename)
        clean_serl_pickle(full_path)
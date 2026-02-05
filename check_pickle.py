import pickle
import matplotlib.pyplot as plt
import numpy as np

def inspect_and_display_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        _, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes.imshow(np.squeeze(data[159]['observations']['wrist'].astype('uint8')))
        plt.tight_layout()
        plt.show()
    # num_to_show = min(len(images), 1)
    # _, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    
    # # Ensure axes is always iterable even if only 1 image
    # if num_to_show == 1:
    #     axes = [axes]

    # for i in range(num_to_show):
    #     # Convert to uint8 for proper color scaling
    #     axes[i].imshow(images[80+i].astype('uint8'))
    #     axes[i].set_title(f"Wrist Cam {i}")
    #     axes[i].axis('off')
    #42 97 159
    
    axes[0].imshow(data[0]['observations']['wrist'][159].astype('uint8'))
    axes[0].set_title(f"Wrist Cam {42}")
    axes[0].axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Total transitions in file: {len(data)}")

    images = []
    for entry in data:
        if 'observations' in entry and 'wrist' in entry['observations']:
            img_data = entry['observations']['wrist']
            
            # 1. Squeeze out the extra (1, ...) dimension
            img_data = np.squeeze(img_data) 
            
            # 2. Check for (C, H, W) vs (H, W, C)
            # If shape is (3, 128, 128), move the 3 to the end
            if img_data.ndim == 3 and img_data.shape[0] == 3:
                img_data = img_data.transpose(1, 2, 0)
                
            images.append(img_data)

    if not images:
        print("No images found.")
        return

    # Display settings

    # with open(file_path, 'wb') as f:
    #     pickle.dump(data_to_save, f)

inspect_and_display_pickle('classifier_data/ur5e_aruco_pick_200_success_images_2026-02-05_21-27-53.pkl')

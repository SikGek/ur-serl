import pickle
import numpy as np
import cv2
import os

def interactive_cleaner(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # file_path2 = 'classifier_data/door_opening_250_success_images_2026-02-24_21-29-20.pkl'

    # with open(file_path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # with open(file_path, 'wb') as f:
    #     pickle.dump(data+data2, f)
    indices_to_remove = []
    print("--- Controls ---")
    print("Any Key: Next Image")
    print("'d'    : Mark for Deletion")
    print("'q'    : Save and Quit")
    
    cv2.namedWindow("Robotics Data Review", cv2.WINDOW_NORMAL)

    for i, entry in enumerate(data):
        try:
            # Extract and Squeeze image
            img = np.array(entry['observations']['wrist']).squeeze()
            
            # Convert (C, H, W) to (H, W, C)
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            
            # Convert RGB to BGR for OpenCV display
            img_bgr = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
            
            # Add text overlay so you know which index you are looking at
            display_img = img_bgr.copy()
            cv2.putText(display_img, f"Index: {i} | Total: {len(data)}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Robotics Data Review", display_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('d'):
                indices_to_remove.append(i)
                print(f"Marked index {i} for deletion.")
            elif key == ord('q'):
                print("Quitting review...")
                break
                
        except Exception as e:
            print(f"Error at index {i}: {e}")

    cv2.destroyAllWindows()

    # Step 2: Save the cleaned data
    if indices_to_remove:
        clean_data = [entry for i, entry in enumerate(data) if i not in indices_to_remove]
        output_path = file_path
        with open(output_path, 'wb') as f:
            pickle.dump(clean_data, f)
        print(f"Done! Removed {len(indices_to_remove)} items. Saved to {output_path}")
    else:
        print("No items were marked for deletion.")

# Run it
interactive_cleaner('classifier_data/door_opening_300_success_images_2026-02-25_17-25-38.pkl')


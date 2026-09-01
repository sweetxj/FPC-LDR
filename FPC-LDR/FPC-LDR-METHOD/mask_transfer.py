import os
import cv2
import numpy as np

# ======================  ======================

original_mask_dir = r""

output_mask_dir = r""
# =================================================================

def convert_visa_masks():
    os.makedirs(output_mask_dir, exist_ok=True)

    img_suffix = ('.png', '.jpg', '.jpeg', '.bmp')
    mask_files = [f for f in os.listdir(original_mask_dir) if f.lower().endswith(img_suffix)]

    if len(mask_files) == 0:
        print("Not found mask!！")
        return

    print(f"total_mask: {len(mask_files)}，processing...\n")

    for idx, mask_name in enumerate(mask_files):
        
        mask_path = os.path.join(original_mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"skkip not found ：{mask_name}")
            continue

        
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

       
        save_path = os.path.join(output_mask_dir, mask_name)
        cv2.imwrite(save_path, binary_mask)

        if (idx + 1) % 20 == 0 or idx + 1 == len(mask_files):
            print(f"processed：{idx+1}/{len(mask_files)}")

    print(f"\nfinished,saved to：{output_mask_dir}")

if __name__ == "__main__":
    convert_visa_masks()

    
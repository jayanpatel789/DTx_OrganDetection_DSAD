import DSAD_Wrapper
import os
import re

def dataset_test():
    pass

    # Initialise test dataset
    Dataset = DSAD_Wrapper.DSAD_Dataset('test', transform=None)

    # Obtain file paths
    file_paths = Dataset.file_paths

    # For loop to ensure that image and mask paths are corresponding
    for pair in file_paths:
        image_path, mask_path = pair
        if verify_paths(image_path, mask_path):
            # print("Hooray")
            pass
        else:
            # print(f"{image_path} + {mask_path}")
            print("Oh dear")
            return
        
    print("Hooray")


def verify_paths(image_path, mask_path):
    # Extract directory paths
    image_dir, image_file = os.path.split(image_path)
    mask_dir, mask_file = os.path.split(mask_path)

    # Extract file numbers
    img_match = re.search(r'\d+', image_file)
    img_no = img_match.group()
    mask_match = re.search(r'\d+', mask_file)
    mask_no = mask_match.group()
    

    # Check if the directories are the same
    # print(f"{image_file}")
    if image_dir != mask_dir:
        return False

    # Check if the prefixes are 'image' and 'mask' and the numbers are the same
    if img_no != mask_no:
        return False

    return True

if __name__ == "__main__":
    dataset_test()
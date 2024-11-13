"""
A program to organise the multilabel data drom the DSAD, in order to create COCO panoptic annotations.
"""

from PIL import Image
import os
import re

def multilabelOrganise(root, target):
    """
    root: path to multilabel folder
    target: path for reorganised multilabel folder
    """
    
    # Initialise train, test and val folders
    train_folder = os.path.join(target, 'train')
    test_folder = os.path.join(target, 'test')
    val_folder = os.path.join(target, 'val')
    # Create folders if not already existing
    makedir(target)
    folders = [train_folder, test_folder, val_folder]
    for path in folders:
        makedir(path)

    # Loop through each surgery within multilabel folder
    for surgery_no in os.listdir(root):
        print(surgery_no)
        try:
            _ = int(surgery_no)
        except:
            print(f"Invalid folder name: {surgery_no}. Folder skipped.")
            continue
        subset = getSubset(surgery_no) # Obtain the subset for correct storage of image and mask later on
        surgery_path = os.path.join(root, surgery_no)
        # Error prevention, but shouldn't be called
        if not os.path.isdir(surgery_path):
            print("WARNING: Loop skipped due to invalid surgery path. Investigate issue.")
            continue
        img_files = []
        for f in os.listdir(surgery_path):
            if f.endswith('.png'):
                img_files.append(f)

        for file_name in img_files:
            # Create image object from the png
            source = os.path.join(surgery_path, file_name)
            img_obj = Image.open(source)
            # Crop image
            img_obj = crop_image(img_obj)

            # Create the image id to be used
            # Take the digits from the png filename
            number_match = re.search(r'\d+', file_name)
            number = number_match.group()
            # Create image id
            image_id = f"2{surgery_no}{number}"

            # Save image to correct location
            if 'image' in file_name:
                dest_dir = os.path.join(target, subset, 'images')
                makedir(dest_dir)
                new_file_name = image_id + '.jpg'
                destination = os.path.join(dest_dir, new_file_name)
                img_obj.save(destination)
            else: # If its a mask
                dest_dir = os.path.join(target, subset, 'individual_masks')
                makedir(dest_dir)
                # Create filename
                organ_name = file_name.split('.')
                organ_name = organ_name[0].split('_')
                organ_name = organ_name[1:]
                organ_name = '_'.join(organ_name)
                new_file_name = f'{image_id}_{organ_name}.png'
                # Save
                destination = os.path.join(dest_dir, new_file_name)
                img_obj.save(destination)

def getOrganCode(input):
    """
    Function to return the code of the organ or vice versa, depending on
    the input type.
    """
    lookup = [
        ("abdominal_wall", "01"),
        ("colon", "02"),
        ("inferior_mesenteric_artery", "03"),
        ("intestinal_veins", "04"),
        ("liver", "05"),
        ("pancreas", "06"),
        ("small_intestine", "07"),
        ("spleen", "08"),
        ("stomach", "09"),
        ("ureter", "10"),
        ("vesicular_glands", "11"),
    ]

    for organ, code in lookup:
        if organ == input:
            return code
        elif code == input:
            return organ
        
    raise ValueError("Input to getOrganCode not found")

def getSubset(surgery_no):
    """
    Function to obtain the correct subset for each surgery number using
    the split from Kolbinger et al. (2023)
    """
    # Define dictionary lookup
    lookup = {
        "01": 'train',
        "04": 'train',
        "05": 'train',
        "06": 'train',
        "08": 'train',
        "09": 'train',
        "10": 'train',
        "12": 'train',
        "15": 'train',
        "16": 'train',
        "17": 'train',
        "19": 'train',
        "22": 'train',
        "23": 'train',
        "24": 'train',
        "25": 'train',
        "27": 'train',
        "28": 'train',
        "29": 'train',
        "30": 'train',
        "31": 'train',
        "03": 'val',
        "21": 'val',
        "26": 'val',
        "02": 'test',
        "07": 'test',
        "11": 'test',
        "13": 'test',
        "14": 'test',
        "18": 'test',
        "20": 'test',
        "32": 'test'
    }

    # These lines can be used if the filename is used as input
    # match = re.match(r'DSAD_img_\d{2}_(\d{2})_\d{2}\.jpg', filename)
    # surgery_no = match.group(1)

    return lookup[surgery_no]

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)
          
def crop_image(image_object):
    # Define amount to crop the image
    top_border = 70
    bottom_border = 80
    left_border = 10
    right_border = 10

    # Crop image
    width, height = image_object.size
    cropped_image = image_object.crop((left_border, top_border, width - right_border, height - bottom_border))

    return cropped_image

def main():
    # Define root and target paths
    root = r"" # Insert path to multilabel folder
    target = r"" # Insert path to location for reorganised multilable folder
    multilabelOrganise(root, target)

if __name__ == "__main__":
    main()
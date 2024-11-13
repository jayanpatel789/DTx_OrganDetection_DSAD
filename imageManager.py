""""
A program to organise the images and masks of the DSAD
into separate directories.
"""
from PIL import Image
import os
import re

def imageManager(root, target):
    """
    Function to provide filename code for each single class image in the
    dataset and to then copy the image into a single directory.
    Filename code structure: DSAD_XX_YY_ZZ.jpg
        XX - Organ structure code
        YY - Surgery number
        ZZ - Image number

        root: path to DSAD folder
        target: path to desired location for reorganised DSAD
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

    # Initialise annotation ID for use in filenames for masks later
    annotation_id = 0
    # Loop through each folder in the root directory except multilabel and mocktest
    for organ_name in os.listdir(root):
        print(organ_name)
        organ_path = os.path.join(root, organ_name)
        if not os.path.isdir(organ_path) or organ_name in ['multilabel', 'mocktest']:
            continue
        # Obtain the corresponding number for the filename
        organ_code = getOrganCode(organ_name)
        
        # Loop through each surgery folder within each organ folder
        for surgery_no in os.listdir(organ_path):
            print(surgery_no)
            try:
                _ = int(surgery_no)
            except:
                print(f"Invalid folder name: {surgery_no}. Folder skipped.")
                continue
            subset = getSubset(surgery_no) # Obtain the subset for correct storage of image and mask later on
            surgery_path = os.path.join(organ_path, surgery_no)
            if not os.path.isdir(surgery_path):
                continue
            # Obtain list of all image and mask .png files within the surgery_path
            img_files = []
            for f in os.listdir(surgery_path):
                if f.endswith('.png'):
                    img_files.append(f)

            # Copy each file to the destination directory with a new name
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
                # Create image ID integer
                image_id = f'1{organ_code}{surgery_no}{number}'
                # Save file in correct path depending on file type
                if 'image' in file_name:
                    dest_dir = os.path.join(target, subset, 'images')
                    makedir(dest_dir)
                    new_file_name = image_id + '.jpg'
                    destination = os.path.join(dest_dir, new_file_name)
                    img_obj.save(destination)
                else: # If a mask
                    dest_dir = os.path.join(target, subset, 'masks')
                    makedir(dest_dir)
                    annotation_id += 1
                    new_file_name = f'{image_id}_{organ_name}_{annotation_id}.png'
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
    top_border_height = 60
    bottom_border_height = 70
    # Crop image
    width, height = image_object.size
    cropped_image = image_object.crop((0, top_border_height, width, height - bottom_border_height))

    return cropped_image


def main():
    # Define root and target paths
    root = r"" # Insert path to DSAD folder
    target = r"" # Insert path to location for reorganised DSAD folder
    imageManager(root, target)

if __name__ == "__main__":
    main()
from PIL import Image
import numpy as np
import os
from glob import glob
import copy
from torch.utils.data import Dataset

class DSAD_Dataset(Dataset):
    """
    Dataset class to control the training, validation and testing of the Dresden
    Surgical Anatomy Dataset.
    """

    # Definition of the training sets. These were used in Kobinger et al. (2023).
    # Same split used for fair comparison of the results.
    ref_train_set = [1, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 22, 23, 24, 25,
    27, 28, 29, 30, 31]
    ref_val_set = [3, 21, 26]
    ref_test_set = [2, 7, 11, 13, 14, 18, 20, 32]
    
    # Initialise the file path roots for both PC and ARC
    paths = [r'c:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD',
             r'tbc']

    def __init__(self, mode, transform=None):
        # Initialise configurations
        self.mode = mode
        # Select the surgeries to be used
        if mode == 'train':
            self.surgery_set = self.__class__.ref_train_set
        elif mode == 'val':
            self.surgery_set = self.__class__.ref_val_set
        elif mode == 'test':
            self.surgery_set = self.__class__.ref_test_set
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'val', or 'test'.")
        
        # Select the path of the DSAD depending on the device being used.
        if os.path.isdir(self.__class__.paths[0]):
            self.path = self.__class__.paths[0]
        else:
            self.path = self.__class__.paths[1]
        
        self.transform = transform

        # Go through the surgeries of the selected mode in the DSAD path and
        # collate all the image and mask paths
        self.file_paths = DSADFilePathCollection(self.mode, self.path)

    def __len__(self):
        # Return length of the data structure storing paths
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB") # Ensure image is in colour
        mask = Image.open(mask_path).convert("L") # Ensure mask in in greyscale

        # Convert to correct dimensions and a tensor
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    

        

def DSADFilePathCollection(surgery_set, root_dir):
    """
    Function to navigate the directory and obtain the paths for all the
    images within the chosen set of surgeries.
    
    Inputs:
        image_set: Array of integers corresponding to the surgery numbers
            to be selected
        root_dir: root directory of the DSAD folder
    Output:
        images: Array of tuples of all the images and their masks
    """

    # Convert image set in to 2 width strings e.g. 1 -> '01'
    surgery_set_str = [str(num).zfill(2) for num in surgery_set]

    file_paths = []
    # Loop through all organ-specific folders
    for organ_folder in os.listdir(root_dir):
        organ_folder_path = os.path.join(root_dir, organ_folder)
        # Skip non-directory entries and excluded folders
        if not os.path.isdir(organ_folder_path) or organ_folder in ['multilabel', 'mocktest']:
            continue
        # Loop through all specified surgeries within each organ folder
        for surgery in surgery_set_str:
            surgery_dir = os.path.join(organ_folder_path, surgery)
            if os.path.exists(surgery_dir):
                images = glob(os.path.join(surgery_dir, 'image*.png'))
                masks = glob(os.path.join(surgery_dir, 'mask*.png'))
                images.sort()  # Ensuring the images and masks are aligned
                masks.sort()   # Sorting is important if glob doesn't guarantee order
                for img, mask in zip(images, masks):
                    file_paths.append((img, mask))
    return file_paths


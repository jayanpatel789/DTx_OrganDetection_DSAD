from PIL import Image
import numpy as np
import copy
from torch.utils.data import Dataset

class DSAD_Dataset(Dataset):
    """
    Dataset class to control the training, validation and testing of the Dresden
    Surgical Anatomy Dataset.
    """

    # Definition of the training sets. These were used in Kobinger et al. (2023).
    # Same split used for fair comparison of the results
    ref_training_set = [1, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 22, 23, 24, 25,
    27, 28, 29, 30, 31]
    ref_validation_set = [3, 21, 26]
    ref_test_set = [2, 7, 11, 13, 14, 18, 20, 32]
    
    # Initialise the file path roots for both PC and HPC

    def __init__(self):
        # Initialise configurations
        # mode
        # shuffle
        # transform

        # Go through the folders and select paths and masks for each surgery
        
        pass

    def __len__(self):
        # Return length of the data structure storing paths
        pass

    def __getitem__(self):
        # 
        pass


# Import necessary libraries
import os
import numpy as np
from PIL import Image
import json
import cv2

## Function definitions
def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)
		
def getOrganCode(input):
    """
    Function to return the code of the organ or vice versa, depending on
    the input type.
    """
    lookup = [
        ("background", "00"),
        ("abdominal_wall", "01"),
        ("colon", "02"),
        ("liver", "03"),
        ("pancreas", "04"),
        ("small_intestine", "05"),
        ("spleen", "06"),
        ("stomach", "07"),
    ]

    for organ, code in lookup:
        if organ == input:
            return code
        elif code == input:
            return organ
        
    raise ValueError("Input to getOrganCode not found")

def createPanopticCategories():
    """
    A function to create all of the categories for the COCO panoptic annotations
    """
    categories = []
    no_of_organs = 7 # Number of organs, not including background
    supercat = "Organ structure"

    predefined_colors = [
        [128, 128, 128], # Grey
        [255, 0, 0],     # Red
        [0, 255, 0],     # Green
        [0, 0, 255],     # Blue
        [255, 255, 0],   # Yellow
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Cyan
        [128, 0, 0],     # Maroon
    ]

    # Assign background class
    category = {
            "supercategory": "none",
            "id": 0,
            "name": "background",
            "isthing": 0,
            "color": predefined_colors[0]
        }
    categories.append(category)

    for i in range(no_of_organs):
        id = i+1
        code = str(id).zfill(2)
        organ_name = getOrganCode(code)
        category = {
            "supercategory": supercat,
            "id": id,
            "name": organ_name,
            "isthing": 1,
            "color": predefined_colors[id]
        }
        categories.append(category)

    return categories

def get_image_dimensions(image_path):
    """
    Open an image file and obtain its dimensions.
    
    Parameters:
    - image_path (str): Path to the image file
    
    Returns:
    - (int, int): Width and height of the image
    """
    with Image.open(image_path) as img:
        return img.size

def createImages(root, subset):
    """
    Function to create the image annotations of the images within a subset directory
    """
    # Initialise list to store images
    images = []
    # Error check for correct folder files
    subsets = ['train', 'test', 'val']
    if subset not in subsets:
        raise ValueError('Invalid subset input to function.')
    
    image_dir = os.path.join(root, subset, 'images')
    image_names = os.listdir(image_dir)

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        width, height = get_image_dimensions(image_path)
        image_id = image_name[:-4]
        if len(image_id) > 5:
            print(f"Image id greater than 5 characters. {image_id}")
        image = {
            "license": 1,
            "file_name": image_name,
            "coco_url": "",
            "height": height,
            "width": width,
            "date_captured": "2000-01-01 00:00:00",
            "flickr_url": "",
            "id": int(image_id)
        }
        images.append(image)

    return images

def getContoursHeightWidth(filepath):
    """
    A function to obtain the contours of the objects within a given
    binary segmentation mask. The input is the path to the mask.
    """
    # Load the binary segmentation mask
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Get width and height
    height, width = mask.shape[:2]

    # Ensure the mask is binary black and white
    _, binary_mask = cv2.threshold(mask, 127, 255, 0)

    # Find the contours (outlines) in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, height, width

def getBBoxPanoptic(contours):
    """
    Function to return the bounding box coordinates in an array for correct COCO formatting

    return:
        labels: list of dicts
    """
    # Initialise to store all contour infos in
    labels = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, w, h]
       
        # Calculate segmentation area
        area = cv2.contourArea(contour)
        # Complete dictionary element for contour
        contour_dict = {
            "area": area,
            "bbox": bbox
        }
        labels.append(contour_dict)
    
    return labels


def getPanMaskAndLabels(mask_filepath, mask=None, color=(255, 255, 255)):
    """
    Draw the contours obtained from the binary mask onto a new image and fill with a specific color.
    Return this image (the mask) and the labels of bbox, segmentation and area
    """

    contours, height, width = getContoursHeightWidth(mask_filepath)
    labels = getBBoxPanoptic(contours)
    
    if mask is None:
        # Create a blank image with the same dimensions
        mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw and fill the contours with the specified color
    cv2.drawContours(mask, contours, -1, color, thickness=cv2.FILLED)

    return mask, labels

def getBackgroundLabels(mask, color):
    """
    Function to take in a panoptic binary masks with all organs segmented, and then obtain
    the segment info for the background class and as the background class to the mask
    """
    
    # Turn the background class white in mask
    bg_mask = np.where(mask == 0, [255, 255, 255], [0, 0, 0]).astype(np.uint8)
    
    # Create a mask with organ areas as black
    mask_white = np.all(bg_mask == [255, 255, 255], axis=-1)
    bg_mask = np.zeros_like(mask)
    bg_mask[mask_white] = [255, 255, 255]

    new_mask_gray = cv2.cvtColor(bg_mask, cv2.COLOR_RGB2GRAY)

    # Find the contours for the background
    external_contours, _ = cv2.findContours(new_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(new_mask_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    labels = getBBoxPanoptic(external_contours)
    
    # Draw and fill the contours with the specified color
    cv2.drawContours(mask, contours, -1, color, thickness=cv2.FILLED)

    return mask, labels

def createCocoJSONDataFile(images, categories, path, subset, annotations=None):
    """
    A function to create the json file, complete with all labels for the images within a particular subset
    NOTE: Annotations field not yet added.
    """

    # Define info and licenses
    # Info
    info = {
        "description": "The Dresden Surgical Anatomy Dataset for Abdominal" +
            " Organ Segmentation",
        "url": "https://springernature.figshare.com/articles/dataset/" +
            "The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_" +
            "segmentation_in_surgical_data_science/21702600",
        "version": "1.0",
        "year": "2022",
        "contributor": "Carstens, M., Rinner, F.M., Bodenstedt, S., Jenke, A.C." +
        ", Weitz, J., Distler, M., Speidel, S. and Kolbinger, F.R., 2023. " + 
        "The Dresden Surgical Anatomy Dataset for abdominal organ segmentation " +
        "in surgical data science. Scientific Data, 10(1), pp.1-8.",
        "date_created": "2022/12/12"
    }
    # Licenses
    licenses = [
        {
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "id": "1",
            "name": "Attribution License"
        }
    ]
    
    # Create json file structure
    data = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    # Save to file location
    filename = f"{subset}_panoptic_annotations.json"
    dest = os.path.join(path, subset, 'annotations')
    makedir(dest)
    dest = os.path.join(dest, filename)
    with open(dest, 'w') as f:
        print(f"Json file opened. {subset}")
        json.dump(data, f)

    return

def main():
    # Insert path to DSAD multilabel folder within the main DSAD directory into the root variable
    root = r""
    subsets = ['val', 'test', 'train']

    # Universal information
    categories = createPanopticCategories()

    # Loop through each subset
    for subset in subsets:
        print(f"Starting json file creation for {subset}.")
        # Create initial json file, for information extraction
        image_labels = createImages(root, subset)
        createCocoJSONDataFile(images=image_labels, categories=categories, path=root,
                               subset=subset)
        
        # Get category and image ID information
        annotations_file = os.path.join(root, subset, 'annotations', f"{subset}_panoptic_annotations.json")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        categories = coco_data['categories']
        image_labels = coco_data['images']

        mask_folder = os.path.join(root, subset, 'individual_masks')

        annotation_labels = []

        for image_label in image_labels:
            image_id = image_label['id']

            # Initialise mask information
            new_mask = None
            segment_id = 1
            segments_info = []

            for category in categories:
                # Skip background class - will be completed later
                if category['name'] == 'background':
                    continue

                class_id = category['id']
                mask_path = os.path.join(mask_folder, f"{image_id}_{category['name']}.png")

                if not os.path.exists(mask_path):
                    continue

                color = category['color']
                color.reverse()
                new_mask, category_labels = getPanMaskAndLabels(mask_path, new_mask, color)
                for label in category_labels:
                    label['id'] = segment_id
                    label['category_id'] = class_id
                    label['iscrowd'] = 0
                    segment_id += 1
                # Add labels into list of all labels
                segments_info = segments_info + category_labels

            # Create background annotation
            bg_category = categories[0]
            new_mask, bg_labels = getBackgroundLabels(new_mask, bg_category['color'])
            for label in bg_labels:
                label['id'] = segment_id
                label['category_id'] = bg_category['id']
                label['iscrowd'] = 0
                segment_id += 1
            segments_info = segments_info + bg_labels

            # Define the output file path
            output_dir = os.path.join(root, subset, 'masks')
            makedir(output_dir)
            mask_name = f"{image_id}.png"
            output_filepath = os.path.join(output_dir, mask_name)

            # Save the mask to the file location
            cv2.imwrite(output_filepath, new_mask)

            # Create annotations field for image and add to list
            annotation_label = {
                "image_id": image_id,
                "file_name": mask_name,
                "segments_info": segments_info
            }
            annotation_labels.append(annotation_label)

        # Create new json file with annotations
        createCocoJSONDataFile(images=image_labels, categories=categories, path=root,
                               subset=subset, annotations=annotation_labels)
        print(f"Completed json file creation for {subset}.")
        print("")

if __name__ == "__main__":
    main()
    
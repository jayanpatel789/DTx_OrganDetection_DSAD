""""
A program to produce the COCO annotations of the Dresden dataset using the COCO
dataset format.
"""
from PIL import Image
import os
import cv2
import json

def DSADtoCOCO(root):
    """
    Function to create the json annotations for the DSAD dataset
        root: path to the parent folder of the train, test and val datasets
    """
    # Define path to store json files
    # json files will be stored within DSAD directory, within 'annotations' subdirectory defined later
    jsonpath = root
    # Categories
    categories = createCategories()

    # Sort through the folder structure
    subsets = ['train', 'test', 'val']

    for subset in subsets:
        # Path to the annotations folder
        images, annotations = createImagesAndAnnotations(root, subset)
        createCocoJSONDataFile(images, annotations, categories, jsonpath, subset)
        print(f"{subset.capitalize()} json file has been created.")
        
        
    
def createCocoJSONDataFile(images, annotations, categories, path, mode):
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
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to file location
    filename = f"{mode}_annotations.json"
    dest = os.path.join(path, mode, 'annotations')
    makedir(dest)
    dest = os.path.join(dest, filename)
    with open(dest, 'w') as f:
        print(f"Json file opened. {mode}")
        json.dump(data, f)

    return

def createImagesAndAnnotations(root, subset):
    # Initialise images and annotations lists
    images = []
    annotations = []
    annotation_id = 0
    # Error check for correct folder files
    subsets = ['train', 'test', 'val']
    if subset not in subsets:
        raise ValueError('Invalid subset input to function.')
    # Create path to the masks folder
    path = os.path.join(root, subset, 'masks')
    # Obtain filename of all masks within annotation folder
    mask_names = os.listdir(path)
    # Loop through each file in annotations folder
    for mask in mask_names:
        image = {} # Initialise dict for the image
        # Obtain info for each image
        mask_path = os.path.join(path, mask)
        image_id = mask[:7] # Image id stored in first 7 characters of filename
        name = image_id + '.jpg' # Create filename
        organ = '_'.join(mask.split('_')[1:-1]) # Obtain the organ name from the filename
        cat_id = int(getOrganCode(organ)) # Find the corresponding category for the organ
        contours, height, width = getContoursHeightWidth(mask_path)
        # Create image dictionary and store in list
        image = {
            "license": 1,
            "file_name": name,
            "coco_url": "",
            "height": height,
            "width": width,
            "date_captured": "2000-01-01 00:00:00",
            "flickr_url": "",
            "id": int(image_id)
        }
        images.append(image)
        # Get annotation specific information
        seg_list = getSegAndBBox(contours)
        for seg_dict in seg_list:
            # Initialise annotation dictionary
            annotation = {}
            # Get segmentation info
            segm = seg_dict["segm"]
            area = seg_dict["area"]
            bbox = seg_dict["bbox"]
            # Annotation ID
            annotation_id += 1
            # Create annotations dictionary and store in list
            annotation = {
                "segmentation": segm,
                "area": area,
                "iscrowd": 0,
                "image_id": int(image_id),
                "bbox": bbox,
                "category_id": int(cat_id),
                "id": int(annotation_id)
            }
            annotations.append(annotation)

    return images, annotations
    

def createCategories():
    """
    A function to create all of the categories for the COCO annotations
    """
    categories = []
    no_of_organs = 11
    supercat = "Organ structure"
    for i in range(no_of_organs):
        id = i+1
        code = str(id).zfill(2)
        organ_name = getOrganCode(code)
        category = {
            "supercategory": supercat,
            "id": id,
            "name": organ_name
        }
        categories.append(category)

    return categories

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

def getBoundingBox(contours, display=False, filepath=None):
    """
    ***DEPRECATED*** (can be used for visualisation of the bounding boxes)
    Function to obtain the bounding boxes from a given contours array from a binary
    segmentation mask. Option to display bounding box, for testing.
    """
    # Create list to store the bounding boxes
    bounding_boxes = []

    # Loop over the contours to create bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # If image has chosen to be displayed
    if display == True and filepath is not None:
        # Obtain original mask
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Define location for test image
        dir = os.getcwd()
        name = 'bbox_img.png'
        path = os.path.join(dir, name)
        # Draw bounding boxes on original mask
        mask_with_boxes = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Save and display image
        cv2.imwrite(path, mask_with_boxes)
        Image.open(path).show()
    elif display == True or filepath is not None:
        print('Argument issue. One of display or filepath not selected. No action taken')
    
    # Bounding boxes is a list of tuples of format (x, y, w, h)
    return bounding_boxes

def getSegmentation(contours):
    """
    ***DEPRECATED***
    Function to obtain the bounding boxes from a given contours array from a binary
    segmentation mask
    """
    # Initialise list to store all segmentations and area
    area = 0
    segmentations = []
    # Loop through contours, extracting and appending information in COCO format (continuous x, y
    # pixel values)
    for contour in contours:
        segmentation = []
        for point in contour:
            x, y = point[0]
            segmentation.append(float(x))
            segmentation.append(float(y))
        segmentations.append(segmentation)
        area += cv2.contourArea(contour)

    return segmentations, area

def getSegAndBBox(contours):
    """
    Function to return the bounding box coordinates and the segmentations points
    coupled together in an array for correct COCO formatting
    """
    # Initialise to store all contour infos in
    labels = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, w, h]
        # Get segmentation
        segmentation = []
        for point in contour:
            x, y = point[0]
            segmentation.append(float(x))
            segmentation.append(float(y))
        # Calculate segmentation area
        area = cv2.contourArea(contour)
        # Complete dictionary element for contour
        contour_dict = {
            "segm": [segmentation],
            "area": area,
            "bbox": bbox
        }
        labels.append(contour_dict)
    
    return labels


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

def main():
    # Insert the file location of the DSAD into path variable
    path = r"/path/to/DSAD/here"
    DSADtoCOCO(path)

if __name__ == "__main__":
    main()
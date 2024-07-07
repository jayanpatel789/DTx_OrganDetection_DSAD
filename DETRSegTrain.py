import os, sys
from pathlib import Path

####################################
# CONFIGURATION
####################################

# Define model configuration
# Define training configuration
n_devices      = 1
epochs         = 100 # 20000 is maximum value but will be prevented by early stopping
weight_decay   = 1e-4
learning_rate  = 1e-4
learning_rate_backbone = 1e-5
check_val_every_n_epoch = 5
load_from_checkpoint = False
checkpoint_path = None
last_manual_checkpoint = 1
last_epoch = 0
## LEARNING_SCHEDULER parameters for ReduceLROnPlateau from pytorch
factor          = 1e-1
lr_patience        = 10
lr_delta           = 1e-5
lr_monitored_var   = "training_loss"
min_lr          = 1e-8
cooldown        = 5
### Parameters for FixedStep from pytorch
fix_step   = False
step_size  = 60
## STOP_CRITERIA parameters for EarlyStopping from pytorch_lightning
stop_monitored_var    = "validation_loss"
stop_delta            = 1e-5
mode             = "min"
stop_patience         = 10   ## Real_patient = patience * check_val_every_n_epoch
## Custom loss function
loss_tags = None
loss_components = []
loss_weights = []

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # Set memory allocation for the GPU

####################################
# LOGGING FUNCTION
####################################

def update_log_screen(exp_path, file_name, mode = 'a'):
    """ Update the log screen to save the output of the model"""
    log_file = f"{file_name}.txt"
    screen_path = exp_path/log_file
    if mode == 'a':
        sys.stdout.close()  
    f = open(screen_path,mode)
    sys.stderr = f
    sys.stdout = f

####################################
# PATH DEFINITIONS
####################################

# Setup results locations
exp_path_bbox = Path.cwd() / 'Results' / 'DETRSegBBox'
if not exp_path_bbox.exists():
    os.makedirs(exp_path_bbox)
print(f"Exp path bbox: {exp_path_bbox}")

exp_path_seg = Path.cwd() / 'Results' / 'DETRSegHead'
if not exp_path_seg.exists():
    os.makedirs(exp_path_seg)
print(f"Exp path bbox: {exp_path_seg}")

####################################
# DATASET CLASS AND SETUP
####################################

print("Dataset setup")

# Create dataset class
import torch
import json
from pathlib import Path
from PIL import Image

class CocoPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')

        img = Image.open(img_path).convert('RGB')
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco['images'])
    
# Create dataset class using the paths to the images and masks
from transformers import DetrFeatureExtractor
import numpy as np
import os

# we reduce the size and max_size to be able to fit the batches in GPU memory
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)

def get_folder_paths(subset, device='HPC'):
    if device == 'HPC':
        root = fr"../DSAD4DeTr_multilabel"
    else:
        root = fr"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr_multilabel"
    img_folder = os.path.join(root, subset, 'images')
    ann_folder = os.path.join(root, subset, 'masks')
    ann_file = os.path.join(root, subset, 'annotations', f"{subset}_panoptic_annotations.json")
    return [img_folder, ann_folder, ann_file]

train_paths = get_folder_paths('train', device='CPU')
test_paths = get_folder_paths('test', device='CPU')
val_paths = get_folder_paths('val', device='CPU')

train_dataset = CocoPanoptic(img_folder=train_paths[0], ann_folder=train_paths[1], ann_file=train_paths[2], feature_extractor=feature_extractor)
test_dataset = CocoPanoptic(img_folder=test_paths[0], ann_folder=test_paths[1], ann_file=test_paths[2], feature_extractor=feature_extractor)
val_dataset = CocoPanoptic(img_folder=val_paths[0], ann_folder=val_paths[1], ann_file=val_paths[2], feature_extractor=feature_extractor)

print("Number of training examples:", len(train_dataset))
print("Number of test examples:", len(test_dataset))
print("Number of validation examples:", len(val_dataset))

print("Dataset setup complete")
print("")

####################################
# COLLATE FUNCTION AND DATALOADER SETUP
####################################
print("Dataloader setup")

from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoded_input = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoded_input['pixel_values']
  batch['pixel_mask'] = encoded_input['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=4)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)

print("Dataloader setup complete")
print("")

####################################
# MODEL CLASS
####################################

print("Model class setup")

import pytorch_lightning as pl
import torch
from transformers import DetrConfig, DetrForSegmentation

id2label = {
    0: 'background',
    1: 'abdominal_wall',
    2: 'colon',
    3: 'liver',
    4: 'pancreas',
    5: 'small_intestine',
    6: 'spleen',
    7: 'stomach'
}

import pytorch_lightning as pl
import torch

class DetrPanoptic(pl.LightningModule):
    def __init__(self, model, lr, lr_backbone, weight_decay):
        super().__init__()
    
        self.model = model

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
    
    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)
        
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                    factor=factor, patience=lr_patience, threshold=lr_delta,
                                    cooldown=cooldown, min_lr=min_lr, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': learning_rate_scheduler,
                'monitor': lr_monitored_var
            }
        }

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
    
print("Model class setup complete")
print("")
    
####################################
# MODEL INTIALISATION FOR BBOX TRAINING
####################################

print("Model intialisation for bbox training")

# Initialise segmentation model for training of bbox predictor
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic", num_labels=len(id2label),
                                            ignore_mismatched_sizes=True)
state_dict = model.state_dict()

# Remove class weights for class labels classifier and bbox predictor
del state_dict["detr.class_labels_classifier.weight"]
del state_dict["detr.class_labels_classifier.bias"]
del state_dict["detr.bbox_predictor.layers.0.weight"]
del state_dict["detr.bbox_predictor.layers.0.bias"]
del state_dict["detr.bbox_predictor.layers.1.weight"]
del state_dict["detr.bbox_predictor.layers.1.bias"]
del state_dict["detr.bbox_predictor.layers.2.weight"]
del state_dict["detr.bbox_predictor.layers.2.bias"]

# Redefine model using the new state_dict with reinitialised layers
model.load_state_dict(state_dict, strict=False)

# Create model class instance
model = DetrPanoptic(model=model, lr=learning_rate, lr_backbone=learning_rate_backbone, weight_decay=weight_decay)

print("Model intialisation for bbox training complete")
print("")

####################################
# BBOX TRAINING
####################################

print("Training of bbox model")

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

stop_criteria = EarlyStopping(monitor=stop_monitored_var, mode=mode, 
                    patience=stop_patience, min_delta=stop_delta, verbose=True)

update_log_screen(exp_path_bbox, 'train_screen')

# Change accelerator here to change from gpu to cpu running
trainer = Trainer(accelerator='gpu', devices=n_devices, 
            enable_progress_bar = True,
            max_epochs          = epochs,
            gradient_clip_val   = 0.1,
            callbacks           = [stop_criteria],
            check_val_every_n_epoch = check_val_every_n_epoch,
            default_root_dir    = exp_path_bbox)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

update_log_screen(exp_path_bbox, 'train_screen')

print("Training complete")
print("")
                
####################################
# SAVE FIRST MODEL
####################################

print("Saving model")

nets_path = exp_path_bbox / "nets"
if not (nets_path).exists():
    os.makedirs(nets_path)
torch.save(model.state_dict(), nets_path / 'final_model.pt')

print("************* Complete *************")

update_log_screen(exp_path_bbox, 'train_screen')

####################################
# INITIALISE MODEL FOR SEG HEAD TRAINING
####################################

print("Model intialisation for seg head training")

# Initialise segmentation model for training of bbox predictor
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic", num_labels=len(id2label),
                                            ignore_mismatched_sizes=True)

state_path = exp_path_bbox / 'nets' / 'final_model.pt'
state_dict = torch.load(state_path)

# Define the mask head keys to remove
keys_to_remove = [
    'mask_head.lay1.weight', 'mask_head.lay1.bias',
    'mask_head.gn1.weight', 'mask_head.gn1.bias',
    'mask_head.lay2.weight', 'mask_head.lay2.bias',
    'mask_head.gn2.weight', 'mask_head.gn2.bias',
    'mask_head.lay3.weight', 'mask_head.lay3.bias',
    'mask_head.gn3.weight', 'mask_head.gn3.bias',
    'mask_head.lay4.weight', 'mask_head.lay4.bias',
    'mask_head.gn4.weight', 'mask_head.gn4.bias',
    'mask_head.lay5.weight', 'mask_head.lay5.bias',
    'mask_head.gn5.weight', 'mask_head.gn5.bias',
    'mask_head.out_lay.weight', 'mask_head.out_lay.bias',
    'mask_head.adapter1.weight', 'mask_head.adapter1.bias',
    'mask_head.adapter2.weight', 'mask_head.adapter2.bias',
    'mask_head.adapter3.weight', 'mask_head.adapter3.bias'
]

# Remove all keys from the state dict
for key in keys_to_remove:
    if key in state_dict:
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)

# Create model class instance
model = DetrPanoptic(model=model, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

print("Model intialisation for seg head training complete")
print("")

####################################
# SEG HEAD TRAINING
####################################

print("Training of seg head model")

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

stop_criteria = EarlyStopping(monitor=stop_monitored_var, mode=mode, 
                    patience=stop_patience, min_delta=stop_delta, verbose=True)

update_log_screen(exp_path_seg, 'train_screen')

# Change accelerator here to change from gpu to cpu running
trainer = Trainer(accelerator='gpu', devices=n_devices, 
            enable_progress_bar = True,
            max_epochs          = epochs,
            gradient_clip_val   = 0.1,
            callbacks           = [stop_criteria],
            check_val_every_n_epoch = check_val_every_n_epoch,
            default_root_dir    = exp_path_seg)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

update_log_screen(exp_path_seg, 'train_screen')

print("Training complete")
print("")

####################################
# SAVE SECOND MODEL
####################################

print("Saving model")

nets_path = exp_path_seg / "nets"
if not (nets_path).exists():
    os.makedirs(nets_path)
torch.save(model.state_dict(), nets_path / 'final_model.pt')

print("************* Complete *************")

update_log_screen(exp_path_seg, 'train_screen')





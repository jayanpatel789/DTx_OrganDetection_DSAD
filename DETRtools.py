import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
import torchvision
import torchvision.transforms as transforms
import os

# DEFINITIONS

# Define classes
ID2LABEL = {
    1: 'abdominal_wall',
    2: 'colon',
    3: 'liver',
    4: 'pancreas',
    5: 'stomach',
    6: 'small_intestine',
    7: 'spleen'
    }

## LEARNING_SCHEDULER parameters for ReduceLROnPlateau from pytorch
factor          = 1e-1
lr_patience        = 10
lr_delta           = 1e-5
lr_monitored_var   = "training_loss"
min_lr          = 1e-8
cooldown        = 5

###### STANDARD DETR #######
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, config, backbone,
                 train_dataloader=None, val_dataloader=None, freeze=None):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if backbone == 101:
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101",
                                                                config=config,
                                                                ignore_mismatched_sizes=True)
        else:
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                                config=config,
                                                                ignore_mismatched_sizes=True)
            
        if freeze == 'CNN':
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False
        elif freeze == 'Tx':
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.model.decoder.parameters():
                param.requires_grad = False
        
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

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
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
    

# Dataset class
class DSADDetection(torchvision.datasets.CocoDetection):
    # Change to initialisation arguments. train argument removed and data_tag
    # argument added so that train, test and val datasets can be created
    def __init__(self, img_folder, processor, data_tag, device='HPC',
                 transformations=None):
        """
        img_folder: path to the root directory with all images
        """
        # Change by adding error checking for data_tag
        if data_tag not in ['train', 'test', 'val']:
            raise ValueError("Incorrect data tag used for initialisation of dataset")
        if device == 'HPC':
            root = fr"../DSAD4DeTr_multilabel_OD/{data_tag}"
        else:
            root = fr"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr_multilabel_OD\{data_tag}"
        ann_file = os.path.join(root, 'annotations', f"{data_tag}_OD_annotations.json") # Change to path for ann_file
        print(f"{data_tag} ann file: {ann_file}")
        super(DSADDetection, self).__init__(img_folder, ann_file)
        self.processor = processor
        self.transforms = transformations

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(DSADDetection, self).__getitem__(idx)

        # Apply transformations if available
        if self.transforms is not None:
            img, anno = self.transforms(img, anno) 

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
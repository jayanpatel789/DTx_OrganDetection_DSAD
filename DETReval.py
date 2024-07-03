import torchvision
import os, sys
from pathlib import Path

# Define training configuration
n_devices      = 1
epochs         = 20000 # Maximum value but will be prevented by early stopping
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

# Logging function
def update_log_screen(exp_path, file_name = 'log_screen', mode = 'a'):
    """ Update the log screen to save the output of the model"""
    log_file = f"{file_name}.txt"
    screen_path = exp_path/log_file
    if mode == 'a':
        sys.stdout.close()  
    f = open(screen_path,mode)
    sys.stderr = f
    sys.stdout = f

# Dataset class
class CocoDetection(torchvision.datasets.CocoDetection):
    # Change to initialisation arguments. train argument removed and data_tag
    # argument added so that train, test and val datasets can be created
    def __init__(self, img_folder, processor, data_tag):
        """
        img_folder: path to the root directory with all images
        """
        # Change by adding error checking for data_tag
        if data_tag not in ['train', 'test', 'val']:
            raise ValueError("Incorrect data tag used for initialisation of dataset")
        root = fr"../DSAD4DeTr/{data_tag}"
        ann_file = os.path.join(root, 'annotations', f"{data_tag}_annotations.json") # Change to path for ann_file
        print(f"{data_tag} ann file: {ann_file}")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    

## Set up the CocoDetection class and image processor
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def get_img_folder_path(data_tag, device='HPC'):
    if device == 'HPC':
        img_folder = fr"../DSAD4DeTr/{data_tag}/images"
    else:
      img_folder = fr"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr\{data_tag}\images"
    return Path(img_folder)

# Setup model location
exp_path = Path.cwd() / 'Results' / 'DETRexample'
if not exp_path.exists():
    os.makedirs(exp_path)
print(f"Model directory: {exp_path}")

update_log_screen(exp_path, 'eval_screen', 'w')

####################################
# Set data tag to train, test or val
####################################
test_img_folder = get_img_folder_path('test', device='HPC')
val_img_folder = get_img_folder_path('val', device='HPC')
train_img_folder = get_img_folder_path('train', device='HPC')

test_dataset = CocoDetection(img_folder=test_img_folder, processor=processor, data_tag='test')
val_dataset = CocoDetection(img_folder=val_img_folder, processor=processor, data_tag='val')
train_dataset = CocoDetection(img_folder=train_img_folder, processor=processor, data_tag='train')

# Check dataset sizes
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
print("Number of test examples:", len(test_dataset))

print("************** Datasets made *****************")

update_log_screen(exp_path, 'eval_screen')


####################################
# Now setting up dataloader
####################################
from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=4)

print("************* Dataloaders made ***************")

update_log_screen(exp_path, 'eval_screen')

####################################
# Evaluation using PyTorch Lightning
####################################
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

id2label = {
    1: 'abdominal_wall',
    2: 'colon',
    3: 'inferior_mesenteric_artery',
    4: 'intestinal_veins',
    5: 'liver',
    6: 'pancreas',
    7: 'small_intestine',
    8: 'spleen',
    9: 'stomach',
    10: 'ureter',
    11: 'vesicular_glands'
    }

print("*************** Model setup ****************")

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)
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

# Setup evaluation

from coco_eval import CocoEvaluator
import torch
from pprint import pprint

# For HPC
state_path = exp_path / 'nets' / 'final_model.pt'

# For PC
# state_path = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\Model\DTx_SurgToolDetector_Dev\Models\DETRexample.pt"

model = Detr(lr=learning_rate, lr_backbone=learning_rate_backbone, weight_decay=weight_decay)
state_dict = torch.load(state_path)
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(val_dataloader):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.5)

    # provide to metric
    # metric expects a list of dictionaries, each item
    # containing image_id, category_id, bbox and score keys
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    # pprint(predictions)
    # print("")
    # predictions = evaluator.prepare_for_coco_detection(predictions)
    # pprint(predictions)

    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()


update_log_screen(exp_path, 'eval_screen')

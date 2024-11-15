import os, sys
from pathlib import Path

# Define training configuration
n_devices      = 1
epochs         = 200 # 20000 is maximum value but will be prevented by early stopping
weight_decay   = 1e-4
learning_rate  = 1e-4
learning_rate_backbone = 1e-5
check_val_every_n_epoch = 5
load_from_checkpoint = False
checkpoint_path = None
last_manual_checkpoint = 1
last_epoch = 0
## STOP_CRITERIA parameters for EarlyStopping from pytorch_lightning
stop_monitored_var    = "validation_loss"
stop_delta            = 1e-5
mode             = "min"
stop_patience         = 10   ## Real_patient = patience * check_val_every_n_epoch

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

def update_log_screen(exp_path, file_name = 'train_screen', mode = 'a'):
    """ Update the log screen to save the output of the model"""
    log_file = f"{file_name}.txt"
    screen_path = exp_path/log_file
    if mode == 'a':
        sys.stdout.close()  
    f = open(screen_path,mode)
    sys.stderr = f
    sys.stdout = f

def get_img_folder_path(data_tag, device='HPC'):
    if device == 'HPC':
        img_folder = fr"../DSAD4DeTr_multilabel_OD/{data_tag}/images"
    else:
        img_folder = fr"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr_multilabel_OD\{data_tag}\images"
    return Path(img_folder)
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # Set memory allocation for the GPU

def main():
    # Obtain parameters from the command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Train a DETR model")

    parser.add_argument('--model_name', type=str, help='Name for results folder')
    parser.add_argument('--qs', type=int, default=100, help='Number of queries')
    parser.add_argument('--backbone', type=int, default=50, help='Insert 101 for resnet-101')
    parser.add_argument('--TxELs', type=int, default=6, help='Number of Tx encoder layers')
    parser.add_argument('--TxDLs', type=int, default=6, help='Number of Tx decoder layers')
    parser.add_argument('--TxAHs', type=int, default=8, help='Number of Tx attention heads')
    parser.add_argument('--augment', type=str, default='False', help='Should data augmentation be used')
    parser.add_argument('--freeze', type=str, default=None, help='Freeze CNN or Tx')

    args = parser.parse_args()

    # Set args to parameter names for clarity
    model_name = args.model_name
    queries = args.qs
    backbone = args.backbone
    TxEncoderLayers = args.TxELs
    TxDecoderLayers = args.TxDLs
    TxAttentionHeads = args.TxAHs
    do_augmentation = args.augment
    freeze = args.freeze

    # Setup results locations
    exp_path = Path.cwd() / 'NewResults' / model_name
    if not exp_path.exists():
        os.makedirs(exp_path)

    update_log_screen(exp_path, mode='w')

    print(f"Exp path: {exp_path}")
    
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    ####################################
    # Create Datasets
    ####################################
    from transformers import DetrImageProcessor
    from DETRtools import DSADDetection
    from data.DataTools import trns_register

    # Define transformations
    if do_augmentation in ['True', 'true', 't', 'T']:
        transformations = {
            'hsv'                : (8,8,5),
            'flip_horizontal'    : 1,
            'scale'              : 0.1,
            'translate'          : 0.1,
            'rotate'             : 10,
            'shear'              : 0.1,
            'probability'        : 0.33
        }
        transform_reg = trns_register(transformations)
    else:
        transform_reg = None


    processor = DetrImageProcessor.from_pretrained(f"facebook/detr-resnet-{backbone}")

    val_img_folder = get_img_folder_path('val', device='HPC')
    train_img_folder = get_img_folder_path('train', device='HPC')

    val_dataset = DSADDetection(img_folder=val_img_folder, processor=processor, data_tag='val')
    train_dataset = DSADDetection(img_folder=train_img_folder, processor=processor, data_tag='train', transformations=transform_reg)

    # Check dataset sizes
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    print("************** Datasets made *****************")

    update_log_screen(exp_path)

    ####################################
    # Now setting up dataloader
    ####################################
    from torch.utils.data import DataLoader

    # Define collate function here as it uses processor
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=16)

    print("************* Dataloaders made ***************")

    update_log_screen(exp_path)

    ####################################
    # Training using PyTorch Lightning
    ####################################
    from transformers import DetrConfig
    import torch
    from DETRtools import Detr

    print("*************** Model setup ****************")

    # Initialise DetrConfig using argument parameters
    config = DetrConfig.from_pretrained(f'facebook/detr-resnet-{backbone}')
    config.num_queries = queries
    config.num_labels = len(ID2LABEL)
    config.encoder_layers = TxEncoderLayers
    config.decoder_layers = TxDecoderLayers
    config.encoder_attention_heads = TxAttentionHeads
    config.decoder_attention_heads = TxAttentionHeads

    # Setup model
    model = Detr(lr=learning_rate, lr_backbone=learning_rate_backbone, weight_decay=weight_decay,
                 config=config, backbone=backbone, train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader, freeze=freeze)

    update_log_screen(exp_path)

    # Start training
    print("************* Training starting *************")

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    stop_criteria = EarlyStopping(monitor=stop_monitored_var, mode=mode, 
                        patience=stop_patience, min_delta=stop_delta, verbose=True)

    update_log_screen(exp_path)

    trainer = Trainer(accelerator='gpu', devices=n_devices, 
                enable_progress_bar = False,
                max_epochs          = epochs,
                gradient_clip_val   = 0.1,
                callbacks           = [stop_criteria],
                check_val_every_n_epoch = check_val_every_n_epoch,
                default_root_dir    = exp_path)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    update_log_screen(exp_path)
                    
    print("************* Saving model *************")

    nets_path = exp_path / "nets"
    if not (nets_path).exists():
        os.makedirs(nets_path)
    torch.save(model.state_dict(), nets_path / 'final_model.pt')

    print("************* Complete *************")

    update_log_screen(exp_path)

if __name__ == "__main__":
    main()
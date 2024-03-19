##### Fix the warning for coming versions!!!
import warnings
warnings.filterwarnings("ignore")
##############################################

import os,sys,pathlib
from datetime import datetime
from config.config import get_cfg_defaults, combine_cfgs

from tools import DETR_Wrapp, update_log_screen
from modeling.ModelTools import get_model

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # Set memory allocation for the GPU

if __name__ == '__main__':
    # Argument for a custom model configuration file, a .yaml file, proceeding -c or --configuration. For default configuration, do
    # not add a configuration argument
    import argparse
    parser = argparse.ArgumentParser("Training script for DETR on endoscopic images.") # Displayed when --help or -h argument is used
    parser.add_argument("-c", "--configuration", type=str, default=None, help = "Path to experiment specifications (.YAML file)")
    args = parser.parse_args()
    # Check for valid config file path
    if args.configuration:
        if pathlib.Path(args.configuration).exists():
            # If config file present, return combined config file
            config = combine_cfgs(args.configuration)
        else:
            raise FileNotFoundError(args.configuration)
    else:
        # If no config file specified, return default config file
        config = get_cfg_defaults()

    # Define the output path for the results
    exp_path = pathlib.Path.cwd() / config.OUTPUT_LOG.path / config.OUTPUT_LOG.exp_tag
    if not exp_path.exists():
        os.makedirs(exp_path)

    # Define path to store the final model
    if not (exp_path/"nets").exists():
        os.makedirs(exp_path/"nets")

    update_log_screen(config.OUTPUT_LOG, 'train_screen', 'w')
    print(f"Training of DETR model on {config.DATA.name}")
    print("Date: ", datetime.now())
    print(f"Used configuration: {config.name}")
    print(f"Folder results: {exp_path}, attempt: {config.OUTPUT_LOG.attept}")
    
    ###################################### Select dataset ############################################
    print("-----------------------------------\n",
    "#####\t Loading dataset",
    "\n-----------------------------------")
    if config.DATA.name == 'MICCAI16_tool_loc':
        from data.DataTools import get_data
        train_dataset, train_dataloader =  get_data(config, data_tag='train', shuffle=True)
        val_dataset  , val_dataloader   =  get_data(config, data_tag='val')
    elif config.DATA.name == 'SurgToolLoc_1126':
        from data.SurgToolLoc_wraper import get_data
        train_dataset, train_dataloader =  get_data(config, data_tag='train', shuffle=True)
        val_dataset  , val_dataloader   =  get_data(config, data_tag='val')

    update_log_screen(config.OUTPUT_LOG, 'train_screen')

    ###################################### Select Loss Criteria #####################################
    print("-----------------------------------\n",
    "#####\t Loss criteria",
    "\n-----------------------------------")
    if config.TRAIN.loss_tags is None:
        print("Using default loss")
        criteria = None
    else:
        print("Using custom loss")
        from Solver.SolverTools import get_loss
        criteria = get_loss(config)
    
    ###################################### Select Model ############################################
    print("-----------------------------------\n",
    "#####\t Model creation",
    "\n-----------------------------------")
    detector = get_model(config)
    model = DETR_Wrapp(detector, config, criteria)
    print("Model correctly initialized")

    update_log_screen(config.OUTPUT_LOG, 'train_screen')


    ######################################    Training     ########################################
    print("-----------------------------------\n",
    "#####\t Training",
    "\n-----------------------------------")
    # Obtain early stopping criteria from config file, apply the criteria to pl EarlyStopping class
    stop_cfg = config.TRAIN.STOP_CRITERIA
    stop_criteria = EarlyStopping(monitor=stop_cfg.monitored_var, mode=stop_cfg.mode, 
                    patience=stop_cfg.patience, min_delta=stop_cfg.delta, verbose=True)

    # Change accelerator here to change from gpu to cpu running
    trainer = Trainer(accelerator='gpu', devices=config.TRAIN.n_devices, 
                enable_progress_bar = True,
                max_epochs          = config.TRAIN.epochs,
                gradient_clip_val   = 0.1,
                callbacks           = [stop_criteria],
                check_val_every_n_epoch = config.TRAIN.check_val_every_n_epoch,
                default_root_dir    = exp_path)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path = config.TRAIN.checkpoint_path)
    
    ######################################    Saving results       ############################################
    print("-----------------------------------\n",
    "#####\t Saving final model",
    "\n-----------------------------------")

    if not (exp_path/"nets").exists():
        os.makedirs(exp_path/"nets")
    torch.save(detector.state_dict(), exp_path/"nets"/f"final{config.OUTPUT_LOG.attept}.pt")

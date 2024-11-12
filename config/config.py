from yacs.config import CfgNode as CN
from pathlib import Path
#########################
# DEFAULT CONFIGURATION #
#########################

_C = CN()
_C.name = "Default configuration"
_C.DATA = CN()
## DATA holds all the configuration to process the data before going into
## the model. This is a configuration for images dataset so a folder of
## images is expected as well as a folder for the different sets
# FOR HPC
# _C.DATA.name              = "DSAD"
# _C.DATA.root              = "../DSAD4DeTr/"
# _C.DATA.train_imgs_path   = "../DSAD4DeTr/train/images/"
# _C.DATA.val_imgs_path     = "../DSAD4DeTr/val/images/"
# _C.DATA.test_imgs_path    = "../DSAD4DeTr/test/images/"
# _C.DATA.sets_path         = "../DSAD4DeTr/"
# _C.DATA.train_set         = "train/annotations/train_annotations.json"
# _C.DATA.validation_set    = "val/annotations/val_annotations.json"
# _C.DATA.test_set          = "test/annotations/test_annotations.json"
# _C.DATA.num_classes       = 11 # This is changed to the number of organs. No background class as it is just bounding and segmenting desired regions
# _C.DATA.remove_background = False # This transformation not currently ready due to issues with its implementation

# FOR PC (testing)
_C.DATA.name              = "DSAD"
_C.DATA.root              = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr"
_C.DATA.train_imgs_path   = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr\train\images"
_C.DATA.val_imgs_path     = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr\val\images"
_C.DATA.test_imgs_path    = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr\test\images"
_C.DATA.sets_path         = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\DSAD4DeTr"
_C.DATA.train_set         = r"train\annotations\train_annotations.json"
_C.DATA.validation_set    = r"val\annotations\val_annotations.json"
_C.DATA.test_set          = r"test\annotations\test_annotations.json"
_C.DATA.num_classes       = 11 # This is changed to the number of organs. No background class as it is just bounding and segmenting desired regions
_C.DATA.remove_background = False # This transformation not currently ready due to issues with its implementation

## In evaluation the background is not removed since the labels are not updated in the COCO format. Fix this in the future.
_C.DATA.normalize         = True
_C.DATA.image_mean        = [0.3373, 0.2634, 0.2446]
_C.DATA.image_std         = [0.2227, 0.1844, 0.1723]
_C.DATA.resize_shortedge  = 450 
_C.DATA.resize_longedge   = 900
### AUGMENTATION holds the specification for the transformations. All the 
### transforms are randomly applied based on the set probability.
_C.DATA.AUGMENTATION = CN()
_C.DATA.Do_augmentation                 = False
_C.DATA.AUGMENTATION.hsv                = None
_C.DATA.AUGMENTATION.flip_horizontal    = None
_C.DATA.AUGMENTATION.scale              = None
_C.DATA.AUGMENTATION.translate          = None
_C.DATA.AUGMENTATION.rotate             = None
_C.DATA.AUGMENTATION.shear              = None
_C.DATA.AUGMENTATION.probability        = 0.0
## TRAIN holds the hyperparameters to be used during the training of the 
## model.
_C.TRAIN = CN()
_C.TRAIN.n_devices      = 1
_C.TRAIN.epochs         = 20000 # Maximum value but will be prevented by early stopping
_C.TRAIN.train_batch    = 16
_C.TRAIN.val_batch      = 8
_C.TRAIN.test_batch     = 8 
_C.TRAIN.weight_decay   = 1e-4
_C.TRAIN.learning_rate  = 1e-4
_C.TRAIN.check_val_every_n_epoch = 5
_C.TRAIN.load_from_checkpoint = False
_C.TRAIN.checkpoint_path = None
_C.TRAIN.last_manual_checkpoint = 1
_C.TRAIN.last_epoch = 0
## LEARNING_SCHEDULER parameters for ReduceLROnPlateau from pytorch
_C.TRAIN.LEARNING_SCHEDULER = CN()
_C.TRAIN.LEARNING_SCHEDULER.factor          = 1e-1
_C.TRAIN.LEARNING_SCHEDULER.patience        = 10
_C.TRAIN.LEARNING_SCHEDULER.delta           = 1e-5
_C.TRAIN.LEARNING_SCHEDULER.monitored_var   = "Train_loss"
_C.TRAIN.LEARNING_SCHEDULER.min_lr          = 1e-8
_C.TRAIN.LEARNING_SCHEDULER.cooldown        = 5
### Parameters for FixedStep from pytorch
_C.TRAIN.LEARNING_SCHEDULER.fix_step   = False
_C.TRAIN.LEARNING_SCHEDULER.step_size  = 60
## STOP_CRITERIA parameters for EarlyStopping from pytorch_lightning
_C.TRAIN.STOP_CRITERIA = CN()
_C.TRAIN.STOP_CRITERIA.monitored_var    = "Val_loss"
_C.TRAIN.STOP_CRITERIA.delta            = 1e-5
_C.TRAIN.STOP_CRITERIA.mode             = "min"
_C.TRAIN.STOP_CRITERIA.patience         = 10   ## Real_patient = patience * check_val_every_n_epoch
## Custom loss function
_C.TRAIN.loss_tags = None
_C.TRAIN.loss_components = []
_C.TRAIN.loss_weights = []
## MODEL 
_C.MODEL = CN()
_C.MODEL.pretrained               = True
## Backbone configuration
_C.MODEL.add_multi_scale_backbone = False
_C.MODEL.mix_backbone_layers      = False
_C.MODEL.MS_config                = "res2net50_26w_4s"
## Transformer configuration
_C.MODEL.TX = CN()
_C.MODEL.TX.queries = None
_C.MODEL.TX.custom         = False
_C.MODEL.TX.load_weights   = True   # Load as many weights as possible from pretrained transformer
_C.MODEL.TX.encoder_layers = 6
_C.MODEL.TX.decoder_layers = 6
_C.MODEL.TX.hidden_dim     = 256    # The length of the feature vectors (projection from the backbone) NOT WORKING YET
## Interface between backbone and transformer
_C.MODEL.dense_inference = False
_C.MODEL.feature_selection = [2,3]  # layer1:0, layer2:1, layer3:2, layer4:3
_C.MODEL.normalize_position = True  # Normalize the position of the features
_C.MODEL.position_scale = None
_C.MODEL.add_pos_activation = False

## OUTPUT_LOG
_C.OUTPUT_LOG = CN()
_C.OUTPUT_LOG.path              = "Results"
_C.OUTPUT_LOG.exp_tag           = "default"
_C.OUTPUT_LOG.save_log          = False
_C.OUTPUT_LOG.attempt            = "0"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def combine_cfgs(path_cfg_data: Path=None):
    """
    An internal facing routine that combined CFG in the order provided.
    :param path_cfg_data: path to path_cfg_data files
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data=Path(path_cfg_data)

    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    return cfg_base
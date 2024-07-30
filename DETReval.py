import torchvision
import torch
import os, sys
from pathlib import Path
import time

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

def update_log_screen(exp_path, file_name = 'eval_screen', mode = 'a'):
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

def speed_test(model, device, dataset, dataloader):
    model.to(device)
    total_time = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            
            # forward pass
            t1 = time.time()
            outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            t2 = time.time()
            total_time += (t2-t1)
            del outputs

    print(f"Total time {total_time} for {len(dataset)} samples in {len(dataloader)} batches.\n",
          f"Time per image {total_time/len(dataset)}, time per batch {total_time/len(dataloader)}\n",
          f"Frames per second {len(dataset)/total_time}, batches per second {len(dataloader)/total_time}")
    total_time = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            pixel_values,_ = dataset[i]
            pixel_values =  pixel_values[None,:,:,:].to(device)
            # forward pass
            t1 = time.time()
            outputs = model.model(pixel_values=pixel_values)
            t2 = time.time()
            total_time += (t2-t1)
    print(f"Test on one by one image, total time {total_time} for {len(dataset)} images",
          f"\n FPS: {len(dataset)/total_time}\n seconds per image {total_time/len(dataset)}")
    
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
    parser.add_argument('--remove', type=str, default=None, help='Remove CNN backbone or Tx weights from model')
    parser.add_argument('--freeze', type=str, default=None, help='Freeze CNN or Tx')
    parser.add_argument('--untrained', type=int, default=0, help='Use untrained model')

    args = parser.parse_args()

    # Set args to parameter names for clarity
    model_name = args.model_name
    queries = args.qs
    backbone = args.backbone
    TxEncoderLayers = args.TxELs
    TxDecoderLayers = args.TxDLs
    TxAttentionHeads = args.TxAHs
    remove = args.remove
    freeze = args.freeze
    untrained = args.untrained

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

    processor = DetrImageProcessor.from_pretrained(f"facebook/detr-resnet-{backbone}")

    test_img_folder = get_img_folder_path('test', device='HPC')

    test_dataset = DSADDetection(img_folder=test_img_folder, processor=processor, data_tag='test')

    # Check dataset sizes
    print("Number of training examples:", len(test_dataset))

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

    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=16)

    print("************* Dataloaders made ***************")

    update_log_screen(exp_path)

    ####################################
    # Evaluation using PyTorch Lightning
    ####################################
    # Setup evaluation
    from coco_eval import CocoEvaluator
    import torch
    from DETRtools import Detr
    from transformers import DetrConfig

    # For HPC
    state_path = exp_path / 'nets' / 'final_model.pt'

    # For PC
    # state_path = r"C:\Users\jayan\Documents\MECHATRONICS YR4\MECH5845M - Professional Project\Model\DTx_SurgToolDetector_Dev\Models\DETRexample.pt"

    # Initialise DetrConfig using argument parameters
    config = DetrConfig.from_pretrained(f'facebook/detr-resnet-{backbone}')
    config.num_queries = queries
    config.num_labels = len(ID2LABEL)
    config.encoder_layers = TxEncoderLayers
    config.decoder_layers = TxDecoderLayers
    config.encoder_attention_heads = TxAttentionHeads
    config.decoder_attention_heads = TxAttentionHeads

    model = Detr(lr=learning_rate, lr_backbone=learning_rate_backbone, weight_decay=weight_decay,
                 config=config, backbone=backbone, freeze=freeze)
    state_dict = torch.load(state_path)

    # Choose whether to remove backbone or transformer weights
    if remove == None:
        new_state_dict = state_dict
    elif remove == 'CNN':
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.backbone.')}
    elif remove == 'Tx':
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.encoder') and not k.startswith('model.model.decoder')}
    elif remove == 'TxE':
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.encoder')}
    elif remove == 'TxD':
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.decoder')}

    if untrained == 0:
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize evaluator with ground truth (gt)
    evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])

    print("Running evaluation...")
    for idx, batch in enumerate(test_dataloader):
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

    torch.cuda.empty_cache()

    speed_test(model, device, test_dataset, test_dataloader)

    update_log_screen(exp_path)

if __name__ == "__main__":
    main()
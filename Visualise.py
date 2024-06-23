import torchvision.transforms as T
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio 

########################################################
#         See object representation space              #
########################################################
def display_latent_space(location, labels, font_size = 10,dot_size = 5):
    """ Visualise the latent space of the model
    Args:
        location (torch.Tensor): Latent space representation of the data
        labels (torch.Tensor): Labels of the data
        font_size (int): Font size of the labels
        dot_size (int): Size of the dots in the scatter plot
    """
    # Define number of columns based on the number of dimensions
    dim = location.shape[0]
    col = np.prod(np.arange(dim,0,-1))/np.prod(np.arange(dim-2,0,-1))/2
    _, ax = plt.subplots(nrows=1,ncols=int(col) ,figsize=(25, 5))
    ax_index = 0
    for i in range(0,dim):
        for j in range(i+1,dim):
            xval, yval = location[i,:], location[j,:]
            ax[ax_index].scatter(xval, yval,c = labels, cmap='tab10', s=dot_size)
            ax[ax_index].set_xlabel(f'component-{i}', fontsize=font_size)
            ax[ax_index].set_ylabel(f'component-{j}', fontsize=font_size)
            ax_index += 1




########################################################
#         Visualisation of intermediate  layers        #
########################################################

def generate_pos_embed_gift(pos_embedding, num_chanels, file_name = 'pos_embed.gif'):
    
    def set_axe_img(axe, chanel, title):
        image = see_position_embedding(chanel) 
        axe.imshow(image)
        axe.set_xlabel(f"min:\t{chanel.min():3.5f}\n max:\t{chanel.max():3.5f}", fontsize = 10)
        axe.set_title(title,fontsize=10)

    def set_axe_plot(axe, slices_xy, title):
        axe.plot(slices_xy[0], label='x', color='red')
        axe.plot(slices_xy[1], label='y', color='blue')
        axe.set_ylim(-1.1,1.1)
        axe.legend()
        axe.set_title(title,fontsize=10)

    frames = []
    for i in range(num_chanels):
        cols = len(pos_embedding)
        fig, axes= plt.subplots(cols,4,figsize=(12,int(5+cols*2)))
        for n_pe in range(cols):
            pos_e = pos_embedding[n_pe]
            chanel = pos_e[0,i*2,:,:]
            slices_xy = chanel[0,:],chanel[:,0]
            title = f'Sin Embedding of\n the {i*2}th channel'
            set_axe_img(axes[n_pe,1], chanel, title)
            set_axe_plot(axes[n_pe,0], slices_xy, title)


            chanel = pos_e[0,i*2+1,:,:]
            slices_xy = chanel[0,:],chanel[:,0]
            title = f'Cos Embedding of\n the {i*2+1}th channel'
            set_axe_img(axes[n_pe,2], chanel, title)
            set_axe_plot(axes[n_pe,3], slices_xy, title)

        plt.tight_layout()
        plt.savefig(f'./Results/frame.png', 
                    transparent = False,  
                    facecolor = 'white'
                )
        plt.close()
        image = imageio.imread(f'./Results/frame.png')
        frames.append(image)
    imageio.mimsave(f'./Results/{file_name}', frames, duration=400, loop=50)

def see_position_embedding(possition_map):
    """ Visualise the positional embedding of one chanel in the feature map
    """
    possition_map = (possition_map+1)/2 # Scale to 0-1 range
    transform = T.ToPILImage()
    image = transform(possition_map)
    return image

########################################################
#               Image Visualisation I/O                #  
########################################################

def see_sample(idx, dataset,id2label, config):
    """ Visualise a sample from the dataset
    Args:
        idx (int): Index of the sample to visualise
        dataset (torch.utils.data.Dataset): Dataset to visualise
        config (dict): Config dictionary
    Returns:
        img (PIL.Image): Image with bounding boxes drawn on it
    """
    print("Normalisation is ", config.DATA.normalize)
    
    img, labels = dataset[idx]
    print("idx:",idx, "image ID:",labels['image_id'])
    print("Labels\n",
          'Img ID:',labels['image_id'], 'Size',labels['size'], 'Classes:',labels['class_labels'], 'Boxes:',labels['boxes'])
    if config.DATA.normalize:
        print(" Format of the boxes (center x,center y,w,h). Normilised values.")
    else:
        print(" Format of the boxes (X1, Y1, X2, Y2)")
    
    img, boxes = format_data(img,labels,config)
    boxes = boxes['boxes']
    draw = ImageDraw.Draw(img, "RGBA")

    for i in range(len(labels['class_labels'])):
        class_idx = labels['class_labels'].numpy()[i]
        x,y,x2,y2 = tuple(boxes.numpy()[i,:])
        draw.rectangle((x,y,x2,y2), outline='red', width=2)
        draw.text((x, y), id2label[class_idx], fill='white')
    
    return img

def see_batch(batch, id2label, config, save_path):
    """ Visualise a batch from the dataloader
    Args:
        batch (dict): Batch of samples to visualise
        id2label (dict): Dictionary mapping class indices to class names
        config (dict): Config dictionary
    """
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

    if int(len(labels)/4)<1:
        rows,cols = 1,len(labels)
    else:
        rows,cols = int((3+len(labels))/4) , 4
    
    fig, axes = plt.subplots(rows,cols,figsize=(18,6))
    for item,ax in enumerate(fig.axes):
        ax.set_axis_off()
        img = pixel_values[item,:,:,:]
        img, boxes = format_data(img,labels[item],config)
        boxes = boxes['boxes']
        draw = ImageDraw.Draw(img, "RGBA")
        for i in range(len(labels[item]['class_labels'])):
            class_idx = labels[item]['class_labels'].numpy()[i]
            x,y,x2,y2 = tuple(boxes.numpy()[i,:])
            draw.rectangle((x,y,x2,y2), outline='red', width=2)
            draw.text((x, y), id2label[class_idx], fill='white')
        ax.set_title(f"Image ID: {labels[item]['image_id'].numpy()}")
        ax.imshow(img)

        img.save(save_path)

def see_output(output, batch, id2label, config, save_path):
    """ Visualise the output of the model
    Args:
        output (dict): Output of the model
        batch (dict): Batch of samples to visualise
        id2label (dict): Dictionary mapping class indices to class names
        config (dict): Config dictionary
    """
    GT_COLOR = (10,70,0)
    P_COLOR = (20,0,255)
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]
    batch_size = len(labels)

    if int(batch_size/4)<1:
        rows,cols = 1,batch_size
    else:
        rows,cols = int((3+batch_size)/4) , 4
    
    fig, axes = plt.subplots(rows,cols,figsize=(25,13))
    for item,ax in enumerate(fig.axes):
        ax.set_axis_off()
        if item<batch_size:
            img = pixel_values[item,:,:,:]
            labels[item].update(output[item])
            img, boxes = format_data(img,labels[item],config)

            if 'image_id' in labels[item].keys():
                img_id = labels[item]['image_id'].numpy()[0]
                ax.set_title(f"Image ID: {img_id}")
            
            if 'boxes' in boxes.keys():
                gt_boxes = boxes['boxes']
                tags = [f"{id2label[idx]}"  for idx in labels[item]['class_labels'].numpy()]
                draw_boxes(img, gt_boxes, tags, True, GT_COLOR)

            if 'p_boxes' in boxes.keys():
                pred_boxes = boxes['p_boxes']
                tags = [f"{id2label[idx]} {100*score:3.2f}%"  for idx,score 
                        in zip(labels[item]['p_class'].numpy(),labels[item]['p_score'].numpy())]
                draw_boxes(img, pred_boxes, tags, False, P_COLOR)

            ax.imshow(img)

            img.save(save_path)

def see_attention_maps(batch, output, id2label, config):
    """ Visualise the attention maps of the model in only one image. Batch size must be 1
    Args:
        batch (dict): Batch of samples to visualise
        output (dict): Output of the model
        config (dict): Config dictionary
    """
    ## Plot the attention maps
    pixel_values = batch["pixel_values"][0,:,:,:]
    labels = batch["labels"][0]
    labels.update(output[0])

    fig, axs = plt.subplots(nrows=3,figsize=(25,13))
    axs[0].set_axis_off()
    axs[0].imshow(get_attention_image(labels['attention_maps'][0], img)) # combination of attention maps in layer 3
    axs[1].set_axis_off()
    axs[1].imshow(get_attention_image(labels['attention_maps'][1], img)) # combination of attention maps in layer 4
    axs[2].set_axis_off()

    ## The image with the boxes
    img, boxes= format_data(pixel_values,labels,config)
    if 'image_id' in labels.keys():
        img_id = labels['image_id'].numpy()[0]
        axs[2].set_title(f"Image ID: {img_id}")

    if 'boxes' in boxes.keys():
        gt_tags = [f"{id2label[idx]}"  for idx in labels['class_labels'].numpy()]
        gt_boxes = boxes['boxes']
        img = draw_boxes(img, gt_boxes, gt_tags, False, (20,70,0))

    if 'p_boxes' in boxes.keys():
        tags = [f"{id2label[idx]} {100*score:3.2f}%"  for idx,score
            in zip(labels['p_class'].numpy(),labels['p_score'].numpy())]
        pred_boxes = boxes['p_boxes']
        img = draw_boxes(img, pred_boxes, tags, True, (0,0,255))
    
    axs[2].imshow(img)
    

def get_attention_image(att_maps, img):
    chnl = 1 #highlighting the attention on the green channel
    image = np.array(img)*.25
    for at in  att_maps:
        heat_map = Image.fromarray(at/at.max()*255)
        heat_map = heat_map.resize(img.size, resample=Image.BILINEAR)
        image[:,:,chnl] = image[:,:,chnl] + 3*image[:,:,chnl]*np.array(heat_map)/255
    image = np.clip(image, 0, 255)

    return Image.fromarray(image.astype(np.uint8))

def draw_boxes(im, boxes, labels, text_onTop=True, color=(0,150,0)):
    """ Draw bounding boxes on an image
    Args:
        im (PIL.Image): Image to draw on
        boxes (torch.Tensor): Bounding boxes
        labels (list(str)): Labels
        text_onTop (bool): Whether to draw the text on top of the box or below
        color (tuple(int)): Color of the bounding boxes
    Returns:
        im (PIL.Image): Image with bounding boxes drawn on it
    """
    draw = ImageDraw.Draw(im, "RGBA")
    f_size = 5
    transparency = 0.5
    try:
        font = ImageFont.truetype("arial.ttf", f_size)
    except:
        path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        font  = ImageFont.truetype(path, f_size)
    for i in range(len(labels)):
        x,y,x2,y2 = tuple(boxes[i,:])
        if text_onTop:
            xt,yt = x,y
        else:
            xm= (x2-x)/2 
            xt,yt = x2-xm,y2
        draw.rectangle((xt,yt,xt+25*len(labels[i]),yt+f_size+2), fill=color + (int(255*transparency),))
        draw.rectangle((x,y,x2,y2), outline=color + (int(255),), width=4)
        draw.text((xt, yt), labels[i], fill='yellow', font=font)
    return im


def format_data(img,label,config):
    """ Format the data to be visualised. It takes care of the cases where the data is normalised or not.
    Args:
        img (torch.Tensor): Image tensor
        label (dict): Labels dictionary from where to extract the bounding boxes
        config (dict): Config dictionary to check if the data is normalised
    Returns:
        img (PIL.Image): Image with bounding boxes drawn on it
        boxes (torch.Tensor): Bounding boxes
        """
    transform = T.ToPILImage()
    boxes = {}
    if config.DATA.normalize:
        # Format of the boxes (center x,center y,w,h). Normilised values.
        # Covert image
        img = (img * torch.tensor(config.DATA.image_std)[:,None,None]) + torch.tensor(config.DATA.image_mean)[:,None,None]
        img = transform(img)
        # Convert gt boxes
        if 'boxes' in label.keys():
            boxes['boxes'] = rescale_bboxes(label['boxes'], img.size)
        # Convert pred boxes
        if 'p_boxes' in label.keys():
            boxes['p_boxes'] = rescale_bboxes(label['p_boxes'], img.size)
    else:
        # Format of the boxes (X1, Y1, X2, Y2)"
        # Covert image
        img = transform(img)
        # Convert gt boxes
        if 'boxes' in label.keys():
            boxes['boxes'] = label['boxes']
        
    return img, boxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)#

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
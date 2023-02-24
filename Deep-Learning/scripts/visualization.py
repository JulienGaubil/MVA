from pathlib import Path
import sys, hydra, os, os.path
sys.path.append('..')
from omegaconf import open_dict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from learnable_typewriter.data.dataset import LineDataset
from learnable_typewriter.utils.defaults import PROJECT_PATH, RUNS_PATH
from learnable_typewriter.typewriter.encoder.encoder import Encoder

from learnable_typewriter.data.dataloader import get_subset_dataloader, collate_fn_pad_to_max
from learnable_typewriter.data import get_dataset
from learnable_typewriter.data.dataset import LineDataset
from torch.utils.data import DataLoader

from learnable_typewriter.typewriter.encoder.window import Window
from learnable_typewriter.utils.image import img

from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose
import PIL
from PIL import Image, ImageDraw
import seaborn as sns
import colorcet as cc

# from tools import get_alphabet, load_data, load_config_model, load_supervised_model_trained, \
# inference_supervised_model, pad_right_batch, reset_compositor, load_object, save_object, load_trainer_supervised_trained

from tools import *

def load_img(path):
    '''Loads an image into a tensor of size (C,H,W), values between 0.0-1.0'''
    totensor = Compose([ToTensor()]) 
    x = Image.open(path).convert('RGB')
    return totensor(x)


# if __name__ == "__main__":
    # x = load_img("C:\\Users\\gaubi\\Documents\\CDocuments\\Cours PC local\\ECL3A\\Stage\\Code\\learnable-typewriters\\datasets\\Alto\\alto_word\\images\\mss-dates_2.png")


def draw_box(img, coords, linewidth = 4, color = "red"):
    '''
    Draw edges of rectangle on image
    Inputs :
        - img : 3d tensor (C,H,W) of an image
        - coords : list [x0, y0, x1, y1] that defines the rectange
    Output :
        - 3d tensor (C,H,W) of image with rectangle drawn
    '''
    topil = Compose([transforms.ToPILImage()])
    totensor = Compose([ToTensor()]) 
    img_ = topil(img)
    draw = ImageDraw.Draw(img_)
    draw.rectangle(coords, fill =None, outline =color, width = linewidth)
    return totensor(img_)


# if __name__ == "__main__":
#     x = load_img("C:\\Users\\gaubi\\Documents\\CDocuments\\Cours PC local\\ECL3A\\Stage\\Code\\learnable-typewriters\\datasets\\Alto\\alto_word\\images\\mss-dates_2.png")
#     coords = [0,0,70,86]  #rectangle défini par coin inférieur gauche (0,0) et coin supérieur droit (70,86)
#     rec = draw_box(x, coords)


def draw_boxes_positions(model, batch, linewidth = 4):
    '''
    Draws rectangle edges around positions of an image in tltw model
    Inputs :
        - model : LearnableTypewriter model initialized with the right alphabet
        - batch : dict that contains for key 'x' a 4d tensor (B,C,H,W) of loaded images

    Outputs : 
        - drawn_imgs : 5d tensor (n_cells,B,C,H,W), contains for each position the batch of images with rectangle corresponding to 
        position drawn onto it
    '''
    imgs = batch['x']
    B,C,H,W = imgs.size()

    #initializes compositor to get positions
    tsf_bkgs, all_tsf_layers, all_tsf_masks = transform_supervised_model_trained(model,batch)
    reset_compositor(model,imgs, tsf_bkgs, all_tsf_layers, all_tsf_masks)
    n_cells = model.compositor.n_cells

    drawn_imgs = torch.empty((n_cells,B,C,H,W))

    colors = sns.color_palette('hls', n_cells)
    colors = torch.Tensor(colors) *255
    colors = colors.int()

    #draws edges
    for p in range(n_cells):  #loop on positions
        ws, we, lws, lwe = model.compositor.get_local_index(p)
        coords_rectangle = [ws,0,we,H]  #coordinates of the rectangle to draw for pos p

        for b in range(B):  #loop on instances of the batch
            img = imgs[b]            
            drawn_img = draw_box(img, coords_rectangle, linewidth, color = tuple(colors[p]))  #draws rectangle of pos p for instance b
            drawn_imgs[p,b,:,:,:] = drawn_img.clone()
            
    return drawn_imgs


# if __name__ == "__main__":
#     path_model = "C:\\Users\\gaubi\\Documents\\CDocuments\\Cours PC local\\ECL3A\\Stage\\Code\\learnable-typewriters\\trained_models\\supervised\\alto_word\\100ep_35pc_100times.pth"
#     data_path = os.path.join(PROJECT_PATH,"datasets\\Alto\\alto_word")

#     # dataloader = load_data(data_path, percentage = 0.35, n_min = 100, split = 'train', shuffle = True)
#     dataloader = load_data(data_path,subset = [0,1,2,3], batch_size=4,split = 'test')
#     batch = next(iter(dataloader))

#     # alphabet = get_alphabet(dataloader)
#     alphabet = load_object(os.path.join(PROJECT_PATH,"trained_models/supervised/alto_word/alphabet_400ep_100times.pth"))
#     trained_model =  load_supervised_model_trained(path_model, alphabet)
    
#     rec_pos = draw_boxes_positions(trained_model,batch, 3)

#     # saves drawn images
#     for p in range(len(rec_pos)):
#         img_name = 'box_pos_{}.png'.format(p)
#         save_image(rec_pos[p], os.path.join(PROJECT_PATH,'outputs\\tests', img_name))


def create_colors_sprites(model):
    '''Creates a color per sprite of the model'''
    colors = sns.color_palette(cc.glasbey, len(model.sprites))
    colors = torch.Tensor(colors)

    return colors


def expand_colors(colors, base):
    colors = colors.unsqueeze(-1).unsqueeze(-1)  #size (K,3,1,1)
    colors = colors.expand(len(base), *base[0].size()) #size (K,3,H_sprite,W_sprite)
    return colors.to(base[0].device)


def expand_colors_layers(colors, tsf_layers, selection):
    n_cells = len (tsf_layers)
    tsf_layers_colored = list()
    colors = torch.cat([torch.Tensor(colors), torch.zeros((1,3))], dim = 0)

    for p in range(n_cells):
        H, W_predicted = tsf_layers[p].size(-2), tsf_layers[p].size(-1)
        colors_p = colors[selection[:,:,p].argmax(0)].unsqueeze(-1).unsqueeze(-1)  #size (N,3,1,1)
        layers_colored = colors_p.expand(colors_p.size(0),*tsf_layers[p][0].size())  #size (N,3,H,W_predicted)
        tsf_layers_colored.append(layers_colored)

    return tsf_layers_colored


def segmentation_trained_model(trained_model, batch):
    '''
    Outputs a segmentation of the composition of a trained model on a given batch
    Inputs :
        - trained_model : LearnableTypewriter model trained
        - selection : 3d tensor (K,B,n_cells) that contains the sprite selection for each instance of batch
        - batch : dict that contains for key 'x' a 4d tensor (B,C,H,W) of loaded images
        - tsf_bkgs : 4d tensor (B,C,H,W), contains transformed backgrounds for all images of the batch
        - all_tsf_layers : list of n_cells elements, contains at each position p a 5d tensor (K, B, C, H, W_predicted),
        all the RGB layers of the transformed sprites at pos p
        - all_tsf_masks : list of n_cells elements, contains at each position p a 5d tensor (K, B, C, H, W_predicted),
        all the transformed masks of sprites at pos p

    Outputs : 
        - segmentation : 4d tensor (B,C,H,W), segmented reconstructed images
    '''
    batch['x'] = batch['x'].to(trained_model.device)

    trained_model.eval()
    colors = create_colors_sprites(trained_model)

    with torch.no_grad():
        out = inference_supervised_model(trained_model, batch)
        proto = trained_model.prototypes.data.clone() # size (K, 3, H_sprite, W_sprite)
        trained_model.prototypes.data.copy_(expand_colors(colors, proto))

        tsf_layers = out['tsf_layers']
        selection = out['selection']  #size (K,N,n_cells)
        tsf_layers_colored = expand_colors_layers(colors,tsf_layers,selection)
        r = (batch['x'], out['tsf_bkgs'], tsf_layers_colored, out['tsf_masks'])
        segmentation = trained_model.compositor(*r)['cur_img']

        trained_model.prototypes.data.copy_(proto)
    
    return segmentation


def make_segmentation_grid(imgs,reconstructed_imgs,segmentations):
    '''
    Creates a grid to show segmentation results
    Inputs : 
        - imgs : 4d tensor (B,C,H,W) that contains the original images
        - reconstructed_imgs : 4d tensor (B,C,H,W) that the images reconstructed with the sprite selection
        - segmentations : 4d tensor (B,C,H,W) that contains the reconstruted images segmented 
    Outputs :
        - grid : 3d tensor (3,H_cat,W_cat), image that has 3 lines of concatenated images, first for original images,
         second for reconstructed images, last for segmented reconstruction
    '''
    nrow = imgs.size()[0]
    transformed_imgs = torch.cat([imgs.unsqueeze(0), reconstructed_imgs.unsqueeze(0), segmentations.unsqueeze(0)], dim=0)
    transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
    grid = make_grid(transformed_imgs, nrow=nrow)
    grid = torch.clamp(grid, 0, 1)
    return grid 


# if __name__ == '__main__' :
#     path_model = "C:\\Users\\gaubi\\Documents\\CDocuments\\Cours PC local\\ECL3A\\Stage\\Code\\learnable-typewriters\\trained_models\\supervised\\alto_word\\100ep_35pc_100times.pth"
#     data_path = os.path.join(PROJECT_PATH,"datasets\\Alto\\alto_word")
#     train_loader = load_data(data_path, batch_size = 16, split = 'train', percentage = 0.35, n_min = 100)
#     val_loader = load_data(data_path, batch_size = 16, percentage = 0.35, n_min = 100)
#     alphabet = get_alphabet(train_loader)
#     trained_model =  load_supervised_model_trained(path_model, alphabet)
#     batch = next(iter(val_loader))
#     tsf_bkgs,all_tsf_layers,all_tsf_masks = transform_supervised_model_trained(trained_model, batch)
#     selection, reconstructed_img = inference_supervised_model(trained_model,batch,tsf_bkgs,all_tsf_layers,all_tsf_masks)
#     segmentations = segmentation_trained_model(trained_model,selection,batch,tsf_bkgs,all_tsf_layers,all_tsf_masks)
    
#     grid_segmentations = make_segmentation_grid(batch['x'], reconstructed_img, segmentations)
#     save_image(grid_segmentations, os.path.join(PROJECT_PATH,'outputs\\tests', 'segmentation_test.png'))


def segmentation_gt_model(trainer_model):
    trainer_model.model.eval()
    obj = trainer_model.decompositor(trainer_model.images_to_tsf.to(trainer_model.device), trainer_model.transcriptions_images_to_tsf, trainer_model.widths_images_to_tsf)
    reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()

    nrow = trainer_model.images_to_tsf.size()[0]
    transformed_imgs = torch.cat([trainer_model.images_to_tsf.unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
    transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
    grid = make_grid(transformed_imgs, nrow=nrow)
    grid = torch.clamp(grid, 0, 1)

    topil = Compose([transforms.ToPILImage()])
    x = np.array(grid if isinstance(grid, PIL.Image.Image) else img(grid))
    if len(x.shape) == 2:
        x = np.repeat(x[:, :, None], 3, axis=2)
    plt.imshow(x)
    plt.show()

    # display(topil(x))


def visualize_sprites(trained_model, val_loader):
    '''
    Outputs a dict of keys the sprites ids and that contains for each sprites all its transforms in the val set
    Inputs :
        - trained_model : LearnableTypewriterSupervised model trained
        - val_loader : Dataloader object with val dataset loaded into
    Outputs :
        - sprite_visu : dict of keys the sprites ids. Contains for each sprite a dict of keys
            -> char : string, character represented by the sprite
            -> mask : 3d tensor (1,H,W_predicted), learned mask of the sprites
            -> cur_imgs : 4d tensor (N,3,H,W_predicted), all N windows of original images around transformations of 
            the sprite selected in the val set
            -> cur_imgs : 4d tensor (N,3,H,W_predicted), all N windows of composed images around transformations of 
            the sprite selected in the val set
            -> segmentations : 4d tensor (N,3,H,W_predicted), all N windows of segmented compositions of images 
            around transformations of the sprite selected in the val set
            -> tsf_mask : 4d tensor (N,1,H,W_predicted), all N masks of transformations of the sprite selected in 
            the val set
            -> tsf_fgs : 4d tensor (N,1,H,W_predicted), all N RGB channels of transformations of the sprite selected
            in the val set
    '''
    masks = trained_model.masks  #size (K, 1, H_sprites, W_sprites)  
    colors = create_colors_sprites(trained_model)  
    matching_inverse = {v:k for (v,k) in enumerate(trained_model.matching)}
    H, W_predicted = trained_model.window.H_cell, trained_model.window.W_predicted 

    #initializes output
    sprite_visu = {k : {'char' : matching_inverse[k], 'mask' : masks[k].to(torch.device('cpu')),\
                        'original_imgs' : torch.zeros((0, 3, H, W_predicted)),\
                        'cur_imgs' : torch.zeros((0, 3, H, W_predicted)),\
                        'segmentations' : torch.zeros((0, 3, H, W_predicted)),\
                        'tsf_masks' : torch.zeros((0, 1, H, W_predicted)),\
                        'tsf_fgs' : torch.zeros((0, 3, H, W_predicted))}\
                        for k in range(len(trained_model.alphabet_))} 
    
    for i, batch in enumerate(val_loader):
        print(i)
        x = batch['x'].to(trained_model.device)

        #predicts transformations and selections for the batch
        tsf_bkgs,all_tsf_layers,all_tsf_masks = transform_supervised_model_trained(trained_model, batch)
        reset_compositor(trained_model, x, tsf_bkgs, all_tsf_layers, all_tsf_masks)        
        selections, reconstruted_imgs = inference_supervised_model(trained_model,batch,tsf_bkgs,all_tsf_layers,all_tsf_masks)
        segmentations = segmentation_trained_model(trained_model,selections,batch,tsf_bkgs, all_tsf_layers, all_tsf_masks, colors).clone()
        reset_compositor(trained_model, x, tsf_bkgs, all_tsf_layers, all_tsf_masks)
        
        #updates the compositor with the given selection
        K, B, _, H, W_predicted = all_tsf_masks[0].size()
        n_cells = len(all_tsf_layers)
        for p in range(n_cells):
            trained_model.compositor.update(p, selections[:, :, p])
        
        #gets composed images, segmentations, local masks and  foregrounds
        original_imgs = x.clone()
        cur_imgs = trained_model.compositor.cur_img.clone()
        
        #saves masks, foregrounds, windows of composed images/segmentations at each position
        for p in range(n_cells):  #runs through positions in the batch
            local = trained_model.compositor.get_local(p)
            ws, we = local['bounds']

            #gets local images/masks
            original_imgs_p = pad_right_batch(original_imgs[:,:,:,ws:we], W_predicted)        
            cur_imgs_p = pad_right_batch(cur_imgs[:,:,:,ws:we], W_predicted)
            segmentations_p = pad_right_batch(segmentations[:,:,:,ws:we], W_predicted)
            masks_p = pad_right_batch(local['cur_mask'][p,:,:,:,:], W_predicted)
            fgs_p = pad_right_batch(local['cur_foreground'][p,:,:,:,:], W_predicted)
            k_p, idx_p = torch.where(selections[:,:,p]==1)  #indexes of sprites selected at position p for all instances

            for j in range(len(k_p)):  #runs through instances of the batch
                b = idx_p[j]
                k = k_p[j].item()
                if k != K-1 : 
                    masks_p_b, fgs_p_b = masks_p[b,:,:,:].unsqueeze(0), fgs_p[b,:,:,:].unsqueeze(0) 
                    
                    sprite_visu[k]['original_imgs'] = torch.cat((sprite_visu[k]['original_imgs'],original_imgs_p[b,:,:,:].unsqueeze(0).to(torch.device('cpu'))),dim = 0)
                    sprite_visu[k]['cur_imgs'] = torch.cat((sprite_visu[k]['cur_imgs'],cur_imgs_p[b,:,:,:].unsqueeze(0).to(torch.device('cpu'))),dim = 0)
                    sprite_visu[k]['segmentations'] = torch.cat((sprite_visu[k]['segmentations'],segmentations_p[b,:,:,:].unsqueeze(0).to(torch.device('cpu'))),dim = 0)
                    sprite_visu[k]['tsf_masks'] = torch.cat((sprite_visu[k]['tsf_masks'],masks_p_b.to(torch.device('cpu'))),dim = 0)
                    sprite_visu[k]['tsf_fgs'] = torch.cat((sprite_visu[k]['tsf_fgs'],fgs_p_b.to(torch.device('cpu'))),dim = 0)
                
    return sprite_visu


def save_sprite_visualization(sprite_visu, filename, n_transforms = 5):
    '''
    Saves a .pkl object that contains the visualization ready to be displayed
    Inputs : 
        - sprite_visu : dict of dicts, the output of visualize_sprites
        - filename : str, full path of the file in which visualization will be saved
        - n_transforms : int, maximal number of instances displayed in a line
    Outputs : 
        - save_visu : a dict of keys the ids of sprites in sprite_visu, contains for each sprite a dict of keys :
            -> char : str, the character represented by the sprite
            -> mask : mask : 3d tensor (1,H,W_predicted), learned mask of the sprites
            -> grid : 3d tensor (3,H,W), image that represents a grid with max n_transforms instances of the
            transformed sprite displayed. 1st/3rd/4th rows: window of original image/composition/segmentation, 2nd row : transformed masks
    '''
    save_visu = dict()

    for k in sprite_visu.keys():  #runs through sprites
        if len(sprite_visu[k]['tsf_masks']) > 0 :
            #selects max nrow random indexes of instances of transformations of the sprite            
            nrow = min(len(sprite_visu[k]['tsf_masks']), n_transforms)
            idx = np.random.randint(0,len(sprite_visu[k]['tsf_masks']),nrow)
            
            #concatenates transformed masks, compositions, segmentation
            tsf_masks_ = sprite_visu[k]['tsf_masks'][idx].unsqueeze(0)
            tsf_masks = torch.cat((tsf_masks_, tsf_masks_, tsf_masks_), dim = 2)  #to match dimensions
            original_imgs = sprite_visu[k]['original_imgs'][idx].unsqueeze(0)
            cur_imgs = sprite_visu[k]['cur_imgs'][idx].unsqueeze(0)
            segmentations = sprite_visu[k]['segmentations'][idx].unsqueeze(0)
            transformations = torch.cat((original_imgs,tsf_masks, cur_imgs, segmentations), dim = 0)
            transformations = torch.flatten(transformations, start_dim=0, end_dim=1)

            #creates the grid
            grid = make_grid(transformations, nrow=nrow, pad_value = 1)
            # grid = torch.clamp(grid, 0, 1)

            save_visu[k] = {'char' : sprite_visu[k]['char'], 'mask' : sprite_visu[k]['mask'], 'grid' : grid}
    
    save_object(save_visu, filename)
    return  save_visu    
                        
                    
def print_sprite_visualization(sprite_visu):
    '''
    Displays the content of sprite_visu when called in a Jupyter Notebook
    Input : 
        - sprite_visu : dict of dicts, the output of save_sprite_visualization
    '''
    topil = Compose([transforms.ToPILImage()])

    for k in sprite_visu.keys() :
        print('Character : {}, occurences : {} (train set)'.format(sprite_visu[k]['char'], sprite_visu[k]['occurences']))
        print('Mask : ')
        display(topil(sprite_visu[k]['mask']))

        print('Original images - Transformed masks - Compositions - Segmentations : ')
        display(topil(sprite_visu[k]['grid']))
        print('')

        print('----------------------------------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------------------------------')
        print('')
           


# if __name__ == '__main__' :
#     model_path = os.path.join(PROJECT_PATH,"trained_models/supervised/alto_word/400ep_100times.pth")
#     data_path = os.path.join(PROJECT_PATH,"datasets/Alto/alto_word")

#     # model_path = os.path.join(PROJECT_PATH,"trained_models/supervised/syn_3l/SupervisedModel.pth")
#     # data_path = os.path.join(PROJECT_PATH,"datasets/synset/3-letter")

#     train_loader = load_data(data_path, batch_size = 16, split = 'train', percentage = 0.35, n_min = 100, num_workers = 8)
#     val_loader = load_data(data_path, batch_size = 16, n_min = 100, shuffle = True, num_workers = 8)

#     alphabet = get_alphabet(train_loader)
#     trained_model =  load_supervised_model_trained(model_path, alphabet)
#     # trained_model =  load_supervised_model_trained(model_path, alphabet).to(torch.device(("cuda:0")))  #for gpu execution (possible cuda memory error)

#     sprite_vis = visualize_sprites(trained_model, val_loader)
#     save_sprite_visualization(sprite_vis, 'sprite_vis.pkl', 10)

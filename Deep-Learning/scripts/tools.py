from pathlib import Path
import sys, hydra, os, os.path
sys.path.append('..')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from omegaconf import open_dict
import time
import timeit
import pickle
from hydra import initialize, compose, initialize_config_dir
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Pad, ToPILImage
from PIL import Image, ImageDraw

from learnable_typewriter.data.dataset import LineDataset
# from learnable_typewriter.data.dataloader import get_subset_dataloader, collate_fn_pad_to_max
from learnable_typewriter.data import get_dataset
from torch.utils.data import DataLoader

from learnable_typewriter.utils.defaults import PROJECT_PATH, RUNS_PATH
from learnable_typewriter.typewriter.typewriter.encoder import Encoder
from learnable_typewriter.typewriter.typewriter.window import Window
from learnable_typewriter.typewriter.typewriter.selection import Selection
from learnable_typewriter.typewriter.model import LearnableTypewriter
from learnable_typewriter.trainer import Trainer

from learnable_typewriter.typewriter.typewriter.sprites.transformations import Transformation



def save_object(obj, filename):
    '''Saves an object with pickle'''
    with open(filename, 'wb') as f:
        pickle.dump(obj,f)

def load_object(filename):
    '''Loads an object saved with pickle'''
    with open(filename, 'rb') as f :
        obj = pickle.load(f)
    return obj


def pad_right(x, max_w, padding_mode = 'constant'):
    '''Pads an image to width max_w with black pixels'''
    w = x.size()[-1]
    x = ToPILImage()(x)
    if padding_mode == 'constant' :
        x = Pad((0, 0, max_w - w, 0), fill = 0)(x)
    elif padding_mode == 'edge' :
        x = Pad((0, 0, max_w - w, 0), padding_mode='edge')(x)

    x = ToTensor()(x)
    if len(x.size()) == 3:
        x = x.unsqueeze(0)
    if len(x.size()) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    return x


def pad_right_batch(batch, max_w, padding_mode='constant'):
    '''Pads a batch of image to width max_w with pad_right function'''
    xs = []
    for i in range(len(batch)):
        x = batch[i]
        xs.append(pad_right(x, max_w,padding_mode))
    return torch.cat(xs, 0)



def get_alphabet(dataloader, space=' ', sep=None):
    '''Outputs a dict of the characters in the dataloader and their number of occurences'''
    alphabet = dict()
    for data in dataloader :
        for i in range(len(data['y'])):
            print(data['y'][i])
            transcription = data['y'][i].replace(space, '')
            if sep is not None :
                transcription = transcription.split(sep)

            for character in transcription :
                if character in alphabet.keys():
                    alphabet[character] += 1
                else :
                    alphabet[character] = 1
    return alphabet


def reset_compositor(model, x, tsf_bkgs, all_tsf_layers, all_tsf_masks):
    '''Resets the compositor of a model'''

    if 'cur_img' in model.compositor.__dict__.keys():
            delattr(model.compositor,'cur_img')
    model.compositor.set(x, tsf_bkgs, all_tsf_layers, all_tsf_masks)
    model.compositor.init()    


def load_data(data_path, split = 'test',  batch_size=32, percentage=1.0, N_min=0, W_max=float('inf'), dataset_size=None, shuffle=False, num_workers=0, **dataset_args):
    '''Outputs a dataloader
    Inputs :
        - subset : list of indexes of instances, if specified the dataloader will be formed only of those instances
        - n_min : int, threshold of minimal apparitions in the dataset for least frequent character in transcriptions
    '''

    dataset_args = dict(dataset_args)

    # data_path = os.path.join(PROJECT_PATH,"datasets\\Alto\\alto_word")
    assert 0 < percentage <= 1.0
    if batch_size is None:           
        batch_size = self.batch_size

    dataset = LineDataset(path=data_path, split=split, crop_width=None, height=64, N_min=N_min, W_max=W_max, dataset_size=dataset_size, **dataset_args)

    print('{} set loaded, len dataset : {}. Instances of width <= {} and N_min >= {}.'.format(split,\
    len(dataset), W_max, N_min))

    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn_pad_to_max, num_workers = num_workers)
    
    return loader   









def get_val_images(dataset_path, num_tsf, padding_mode = 'constant'):

    #gets batches
    val_loader_train = next(iter(load_data(dataset_path,split='train', batch_size=num_tsf//2, dataset_size=num_tsf//2)))
    val_loader_test = next(iter(load_data(dataset_path,split='test', batch_size=(num_tsf - num_tsf//2), dataset_size=(num_tsf - num_tsf//2))))
    
    #process images
    images = [val_loader_train['x']] + [val_loader_test['x']]
    w_max = max(images[0].size()[-1],images[1].size()[-1])
    images[0] = pad_right_batch(images[0],w_max,padding_mode)
    images[1] = pad_right_batch(images[1],w_max,padding_mode)
    val_images = torch.cat(images, dim=0)

    #gets widths and transcriptions
    transcriptions_images = val_loader_train['y'] + val_loader_test['y']
    widths_images = val_loader_train['w'] + val_loader_test['w']

    return {'x' : val_images, 'y' : transcriptions_images, 'w' : widths_images}


# if __name__ == '__main__' :
#     data_path = os.path.join(PROJECT_PATH,"datasets/Google1000/Volume_0002")
#     val_batch = get_val_images(data_path,20, padding_mode='edge')
#     images_tsf = torch.cat([val_batch['x'][k].unsqueeze(0) for k in range(len(val_batch['x']))], dim=0)
#     # images_tsf = torch.flatten(images_tsf, start_dim=0, end_dim=1)
#     print(images_tsf.size())

#     grid = make_grid(images_tsf, nrow=1)
#     grid = torch.clamp(grid, 0, 1)
#     print(grid.size())
#     save_image(grid , os.path.join(PROJECT_PATH,'outputs','dataset.png'))


def load_config_model(config_name, path_config = None):
    '''
    Loads the config file of a model
    Inputs :
        - config_name : str, name of the config file
        - path_config : str, path of the dir where the config file is located. If specified, must be absolute.
    '''
    if path_config == None :
        initialize('../configs/model/')  #initialise avec un path relatif  
    else : 
        initialize_config_dir(path_config)  #initialise avec un path absolu
    cfg = compose(config_name=config_name)
    hydra.core.global_hydra.GlobalHydra.instance().clear()  #supprime l'initialisation
    return cfg


def load_supervised_model_trained(path_model, alphabet, device_id = None):
    '''
    Loads a trained model
    Inputs :
        - path_model : str, full path of the file of the model
        - alphabet : set, contains the characters corresponding to the learned sprites of the model
        - device_id : int, id of the gpu
    '''
    device =  torch.device((f"cuda:{device_id}" if device_id is not None else 'cpu'))
    saved_model = torch.load(path_model, map_location=device)
    saved_model = torch.load(path_model)
    config_model = saved_model['model_cfg']
    if len(alphabet) != config_model['sprites']['n'] : 
        print('Error in loading supervised model, number of sprites ({}) and alphabet length ({}) should match'.format(config_model['sprites']['n'], len(alphabet)))
        return 

    model_state_dict = saved_model['model_state']
    #creates the model
    trained_model = LearnableTypewriterSupervised(config_model, alphabet)
    trained_model.load_state_dict(model_state_dict)  #loads parameters

    return trained_model


if __name__ == '__main__' :
    path_model = "/home/jgaubil/codes/ltw-marionette/trained_models/copiale/supervised/blank_sprite/model_best_recons_val.pth"
    data_path = os.path.join(PROJECT_PATH,"datasets/copiale")
    # alphabet_occurences = get_alphabet(load_data(data_path, split = 'train'))
    alphabet_occurences = {k:k for k in range(121)}
    model = load_supervised_model_trained(path_model, set(alphabet_occurences.keys()))
    print(model.matching)



def inference_supervised_model(trained_model, batch):
    '''
    Uses a trained model to perform prediction of transformations for sprites at each positions
    Inputs :
        - trained_model : LearnableTypewriterSupervised model trained
        - batch : dict that contains for key 'x' a 4d tensor (B,C,H,W) of loaded images
    Outputs : 
        - tsf_bkgs : 4d tensor (B,C,H,W), contains transformed backgrounds for all images of the batch
        - all_tsf_layers : list of n_cells elements, contains at each position p a 5d tensor (K, B, C, H, W_predicted),
        all the RGB layers of the transformed sprites at pos p
        - all_tsf_masks : list of n_cells elements, contains at each position p a 5d tensor (K, B, C, H, W_predicted),
        all the transformed masks of sprites at pos p
    '''
    x = batch['x'].to(trained_model.device)
    y = batch['y']
    widths = torch.tensor(batch['w'])
    B,C,H,W = x.size()
    trained_model.eval()

    #Features creation
    features = trained_model.encoder(x)

    #Prediction of parameters for transformations
    tsf_sprites_params, tsf_layers_params, tsf_bkgs_param = trained_model.predict_parameters(x, features)
    trained_model.transform_layers_ = tsf_layers_params

    # transform backgrounds
    if trained_model.background:
        backgrounds = trained_model.background.backgrounds.unsqueeze(0).expand(B, C, -1, -1).unsqueeze(0)
        tsf_bkgs = trained_model.transform_background(backgrounds, tsf_bkgs_param, (B, C, H, W), trained_model.device)

    # transform sprites
    selection = trained_model.selection(features, trained_model.sprites)

    all_tsf_layers, all_tsf_masks = trained_model.transform_sprites(selection['S'], tsf_sprites_params, tsf_layers_params)
    
    composed = trained_model.compositor(x, tsf_bkgs, all_tsf_layers, all_tsf_masks)


    output = {'selection' : selection['selection'], 'S' : selection['S'], 'reconstructed' : composed['cur_img'],\
    'tsf_layers' : all_tsf_layers, 'tsf_masks' : all_tsf_masks} 

    if trained_model.background:
        output['tsf_bkgs'] = tsf_bkgs   
    return output


def load_transformation(tsf_sequence, model):
    
    # self.layer_transformation = Transformation(self, 1, self.cfg['transformation']['layer'])
    new_cfg = model.cfg['transformation']['layer'].copy()
    new_cfg['ops'].append('tps')
    new_Transformation = Transformation(model,1,new_cfg)
    # print('Color module : ', new_Transformation._modules['tsf_sequences'][0]._modules['tsf_modules'][0])
    # print('Position module : ', new_Transformation._modules['tsf_sequences'][0]._modules['tsf_modules'][1])
    # print('TPS module : ', new_Transformation._modules['tsf_sequences'][0]._modules['tsf_modules'][2])
    # print( new_Transformation._modules['tsf_sequences'][0]._modules['tsf_modules'][0].__dict__.keys())

    state = model.layer_transformation.state_dict()
    print(state.keys())
        
    diff = (state_dict.keys() | state.keys()) - (state_dict.keys() & state.keys())
    if len(diff) > 0:
        diff_amb = state_dict.keys() - self.state_dict()
        if len(diff_amb):
            warnings.warn(f'load_state_dict: The following keys were found in loaded dict but not in self.state_dict():\n{diff_amb}')

        diff_bma = self.state_dict() - state_dict.keys()
        if len(diff_amb):
            warnings.warn(f'load_state_dict: The following keys were found in self.state_dict() dict but not in loaded dict:\n{diff_bma}')

    for name, param in state_dict.items():
        if name in state:
            if isinstance(param, nn.Parameter):
                param = param.data
            state[name].copy_(param)




    # for module in new_Transformation.modules():
        # print('module : ', module)

   
    # self.layer_transformation.load()


# if __name__ == '__main__' :
#     path_model = "/home/jgaubil/codes/ltw-marionette/trained_models/G100_V2_words_small/supervised/path_select/model.pth"
#     data_path = os.path.join(PROJECT_PATH,"datasets/Google1000/Volume_0002_words_small")
#     dataloader = load_data(data_path, split = 'train')
#     alphabet_occurences = get_alphabet(dataloader)
#     trained_model =  load_supervised_model_trained(path_model, set(alphabet_occurences.keys()))
#     batch = next(iter(dataloader))
#     inference_supervised_model(trained_model, batch)



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
# from learnable_typewriter.data import get_dataset
# from torch.utils.data import DataLoader

from learnable_typewriter.utils.defaults import PROJECT_PATH, RUNS_PATH
from learnable_typewriter.typewriter.model import LearnableTypewriter
from learnable_typewriter.trainer import Trainer


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



def get_alphabet(dataloader):
    '''Outputs a dict of the characters in the dataloader and their number of occurences'''
    alphabet = dict()
    for data in dataloader :
        for i in range(len(data['y'])):
            transcription = data['y'][i]
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



def load_trainer(path_model, cfg):
    seed = 4321
    trainer = Trainer(cfg, seed=seed)
    trainer.load_from_dir(path_model, resume=True)

    return trainer

# if __name__ == '__main__':
#     path_model = Path(os.path.join(PROJECT_PATH,'trained_models/unsupervised'))
#     data_path = os.path.join(PROJECT_PATH,"datasets/copiale")

#     cfg = load_config_model('config.yaml','/home/jgaubil/codes/learnable-typewriters/trained_models/unsupervised')
#     cfg['training']['device'] = None

#     val_images = get_val_images(data_path,20, padding_mode='edge')
    
#     trainer = load_trainer(path_model,cfg,supervised=False)
#     trainer.model.eval()
#     val_images = val_images['x'].to(trainer.device)
#     obj = trainer.decompositor(val_images)
#     reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()

#     nrow = val_images.size()[0]
#     val_images = val_images.to('cpu')
#     transformed_imgs = torch.cat([val_images.unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
#     transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)

#     grid = make_grid(transformed_imgs, nrow=nrow)
#     grid = torch.clamp(grid, 0, 1)

#     save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/transformations.png'))



if __name__ == '__main__':
    path_model = Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_humanistique/tsf_tps/model.pth'))
    data_path = os.path.join(PROJECT_PATH,"datasets/Alto/alto_word_humanistique")

    cfg = load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_humanistique/tsf_tps/')
    # cfg['training']['device'] = 3
    # cfg['run_dir'] =  os.path.join(RUNS_PATH, cfg["dataset"]["alias"], "test")
    
    trainer = load_trainer(path_model,cfg)
    prototypes, masks = trainer.model.prototypes, trainer.model.masks
    grid = make_grid(masks, nrow=20, padding=2, pad_value=1)
    grid = torch.clamp(grid, 0, 1)
    grid = 1-grid
    for i, dl in enumerate(trainer.get_dataloader(split='train', batch_size=10, remove_crop=True)):
        for j,batch in enumerate(dl):
            if j == 1:
                data = batch
            save_image(batch['x'], os.path.join(PROJECT_PATH,'outputs/preds/batch_{}.png'.format(j)))
    save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/masks.png'))


    trainer.model.eval()
    decompose = trainer.decompositor
    obj = decompose(data)
    reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
    print(obj['tsf_masks'][0].size(), masks.size())

    for i,mask in enumerate(obj['tsf_masks']):
        mask = mask[3].unsqueeze(0)
        grid = make_grid(mask, nrow=20, padding=2, pad_value=1)
        grid = torch.clamp(grid, 0, 1)
        grid = 1-grid
        save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/masks_{}.png'.format(i)))


    transformed_imgs = torch.cat([data['x'].cpu().unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
    transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
    # self.save_image_grid(transformed_imgs, f'{header}/examples/{mode}/{alias}', nrow=images_to_tsf['x'].size()[0])
    grid = make_grid(transformed_imgs, nrow=data['x'].size()[0])
    grid = torch.clamp(grid, 0, 1)
    save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/preds.png'))
    



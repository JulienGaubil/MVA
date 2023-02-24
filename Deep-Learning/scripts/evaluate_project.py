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

from learnable_typewriter.evaluate.quantitative.sprite_matching.evaluate import er_evaluate, er_evaluate_supervised, metrics_to_average_sub
from learnable_typewriter.evaluate.quantitative.sprite_matching.metrics import error_rate 


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


@torch.no_grad()
def best_path(s):
    return torch.argmax(s['selection'], dim=0)

@torch.no_grad()
def cer_aggregate(x, k):
    x = torch.unique_consecutive(x, dim=-1)
    x = x.tolist()
    return [e for e in x if e != k]

@torch.no_grad()
def evaluate_batch(trainer, batch):
    widths = torch.tensor(batch['w'])
    s = trainer.model.predict_cell_per_cell(batch)

    #reco error
    reco_err = ((s['reconstruction'] - batch['x'])**2).flatten(1).mean(1)


    #cer computation
    n_cells = s['selection'].size()[-1]
    k = s['selection'].size()[0]-1
    true_widths_pos = trainer.model.true_width_pos(batch['x'], widths, n_cells)
    s = best_path(s)
    s = [cer_aggregate(s[i][:true_widths_pos[i]], k) for i in range(s.size()[0])]
    samples =  list(zip(s, batch['y']))
    sep = trainer.train_loader[0].dataset.sep; delim = trainer.train_loader[0].dataset.space; map_pd = trainer.transcribe; map_gt = dict(trainer.transcribe)
    map_gt[-1] = '_'

    er = error_rate(samples, verbose=False, average=False, delim=delim, sep=sep, map_pd=map_pd, map_gt=map_gt)

    return er, reco_err





if __name__ == '__main__':

    data_path = os.path.join(PROJECT_PATH,"datasets/Alto/alto_word")

    
    trainers = {'base':dict(), 'tps':dict()}
    trainers['base']['word'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word/baseline/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word/baseline/'))
    trainers['base']['humanistique'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_humanistique/baseline/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_humanistique/baseline/'))
    trainers['base']['praegothica'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_praegothica/baseline/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_praegothica/baseline/'))
    trainers['base']['textualis'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_textualis/baseline/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_textualis/baseline/'))

    # trainers['tps']['word'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word/tsf_tps/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word/tsf_tps/'))
    # trainers['tps']['humanistique'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_humanistique/tsf_tps/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_humanistique/tsf_tps/'))
    # trainers['tps']['praegothica'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_praegothica/tsf_tps/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_praegothica/tsf_tps/'))
    # trainers['tps']['textualis'] = load_trainer(Path(os.path.join(PROJECT_PATH,'runs/DL/alto_word_textualis/tsf_tps/model.pth')),load_config_model('config.yaml','/home/jgaubil/codes/ltw-marionette/runs/DL/alto_word_textualis/tsf_tps/'))

    loaders = {'humanistique':trainers['base']['humanistique'].get_dataloader(split='test', batch_size=16, remove_crop=True),\
    'praegothica':trainers['base']['praegothica'].get_dataloader(split='test', batch_size=16, remove_crop=True),
    'textualis':trainers['base']['textualis'].get_dataloader(split='test', batch_size=16, remove_crop=True)}

    corresp = ['humanistique','praegothica','textualis'] 
    for tsf in ['base', 'tps']:
        for tr in trainers[tsf].values():
            tr.model.eval()

    for k in range(len(corresp)):  #parcourt les datasets humanistique, textualis..
        for i, dl in enumerate(loaders[corresp[k]]):
            with torch.no_grad():
                for j,batch in enumerate(dl):
                    er_huma, reco_err_huma = evaluate_batch(trainers['base']['humanistique'], batch)
                    er_prae, reco_err_prae = evaluate_batch(trainers['base']['praegothica'], batch)
                    er_text, reco_err_text = evaluate_batch(trainers['base']['textualis'], batch)


                    reco = torch.stack([reco_err_huma,reco_err_prae,reco_err_text],dim=1)
                    er = torch.stack([torch.tensor(er_huma['cer']),torch.tensor(er_prae['cer']),torch.tensor(er_text['cer'])],dim=1)
                    
                    #index in corresp of the best model for the given metric
                    pred_reco = reco.argmin(1)
                    pred_er = er.argmin(1)

                    #number of errors
                    error_pred_reco = (pred_reco==k).sum().item()
                    error_pred_er = (pred_er==k).sum().item()
                    print(error_pred_reco, error_pred_er)

                    #### Problem : les sprites n'ont pas toutes le mm id dans les différents datasets/les différents modèles...



                




                
                


    # trainer.model.eval()
    # decompose = trainer.decompositor
    # obj = decompose(data)
    # reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
    # print(obj['tsf_masks'][0].size(), masks.size())

    # for i,mask in enumerate(obj['tsf_masks']):
    #     mask = mask[3].unsqueeze(0)
    #     grid = make_grid(mask, nrow=20, padding=2, pad_value=1)
    #     grid = torch.clamp(grid, 0, 1)
    #     grid = 1-grid
    #     save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/masks_{}.png'.format(i)))


    # transformed_imgs = torch.cat([data['x'].cpu().unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
    # transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
    # # self.save_image_grid(transformed_imgs, f'{header}/examples/{mode}/{alias}', nrow=images_to_tsf['x'].size()[0])
    # grid = make_grid(transformed_imgs, nrow=data['x'].size()[0])
    # grid = torch.clamp(grid, 0, 1)
    # save_image(grid, os.path.join(PROJECT_PATH,'outputs/preds/preds.png'))
import os
import sys
import json
import PIL
from tqdm import trange, tqdm
from os.path import join

import torch
import numpy as np
from torchvision.transforms import ToPILImage
sys.path.append('/home/ysig/nicolas/learnable-typewriter-supervised')
from learnable_typewriter.utils.loading import load_pretrained_model
from torchvision.utils import make_grid
import plotly.express as px
import pandas as pd

def get_documents(path):
    dirs = set() #set(os.listdir(fontenay_path)) if os.path.isdir(fontenay_path) else set()
    with open(path, 'r') as f:
        annotation = json.load(f)
    
    documents = set(['_'.join(k.split('_')[:-2]) for k, v in annotation.items() if v['split'] == 'train'])
    documents = [d for d in documents if d not in dirs]
    return sorted(documents)

def eval(trainer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for images_to_tsf in trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=False)[0]:
        decompose = trainer.decompositor
        obj = decompose(images_to_tsf)
        reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
        nrow = images_to_tsf['x'].size()[0]
        transformed_imgs = torch.cat([images_to_tsf['x'].cpu().unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
        transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
        grid = make_grid(transformed_imgs, nrow=nrow)
        ToPILImage()(torch.clamp(grid, 0, 1)).save(path)
        return

def plot_sprites(trainer, drr):
    masks = trainer.model.sprites.masks
    os.makedirs(drr, exist_ok=True)
    for k in range(len(trainer.model.sprites)):
        ToPILImage()(masks[k]).save(join(drr, f'{k}.png'))
    ToPILImage()(make_grid(masks, nrow=4)).save(join(drr, f'grid.png'))
    ToPILImage()(make_grid(masks, nrow=len(trainer.model.sprites))).save(join(drr, f'grid-1l.png'))

def finetune(trainer, max_epochs, save_sprites_dir, reconstructions_path, name):
    trainer.__init_decompositor__()
    train_loss = []
    for i in trange(max_epochs):
        losses = []
        for x in trainer.train_loader[0]:
            trainer.model.train()
            trainer.optimizer.zero_grad()
            loss = trainer.model(x)['loss']
            losses.append(loss.item())
            loss.backward()
            trainer.optimizer.step()
        train_loss.append((i, np.mean(losses)))
        
        plot_sprites(trainer, join(save_sprites_dir, str(i).zfill(3)))
        eval(trainer, join(reconstructions_path, str(i).zfill(3) + '.png'))
        trainer.model.eval()

    plot_sprites(trainer, join(save_sprites_dir, 'final'))
    eval(trainer, join(reconstructions_path, f'final.png'))
    fig = px.line(pd.DataFrame(train_loss, columns=['epoch', 'training-loss']), x="epoch", y="training-loss", title=name)
    fig.write_image(join(reconstructions_path, f'loss.png'))

def stack(imgs):
    dst = PIL.Image.new('RGB', (imgs[0].width, len(imgs)*imgs[0].height))
    for i, img in enumerate(imgs):
        dst.paste(img, (0, i*imgs[0].height))
    return dst

def run(args):
    documents = get_documents(args.annotation_file)
    os.makedirs(args.sprites_path, exist_ok=True)
    pbar = tqdm(documents)
    for document in documents:
        pbar.set_description(f"Processing {document}")
        trainer = load_pretrained_model(path=args.model_path, device=str(args.device), kargs=args.kargs)
        transcribe_file = join(args.sprites_path, 'transcribe.json')
        if not os.path.isfile(transcribe_file):
            with open(transcribe_file, 'w') as f:
                json.dump(trainer.model.transcribe, f, indent=4)

        plot_sprites(trainer, join(args.tag, 'baseline'))

        for k in trainer.dataset_kwargs:
            k['filter_by_name'] = document

        trainer.train_loader = trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True)
        trainer.val_loader, trainer.test_loader = [], []
        finetune(trainer, max_epochs=args.max_epochs, save_sprites_dir=join(args.tag, document, args.sprites_path), reconstructions_path=join(args.tag, document, args.reconstructions_path), name=document)
        torch.cuda.empty_cache()
    
    baseline = PIL.Image.open(join(args.tag, 'baseline', f'grid-1l.png'))
    png_list = [stack([baseline for _ in range(len(documents))])]
    png_list += [stack([baseline] + [PIL.Image.open(join(args.tag, document, args.sprites_path, str(i).zfill(3), f'grid-1l.png')) for document in documents]) for i in range(args.max_epochs)]
    png_list[0].save(join(args.tag, 'progress.gif'), save_all=True, duration=len(png_list)*0.1, append_images=png_list[1:])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Helper script to finetune fontenay on a specific set of documents and generate figures.')
    parser.add_argument('-t', "--tag", required=True, default=None)
    parser.add_argument('-o', "--reconstructions_path", default='reco', type=str)
    parser.add_argument('-s', "--sprites_path", default='sprites', type=str)
    parser.add_argument('-i', "--model_path", default='reco', type=str)
    parser.add_argument("--max_epochs", required=False, default=25, type=int)
    parser.add_argument('-a', "--annotation_file", required=False, default='/home/ysig/nicolas/learnable-typewriter-supervised/datasets/fontenay/annotation.json', type=str)
    parser.add_argument('-k', "--kargs", required=False, default='training.optimizer.lr=0.001', type=str)
    parser.add_argument('-d', "--device", default=0, type=str)
    args = parser.parse_args()

    run(args)

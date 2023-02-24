from itertools import chain
import torch
import numpy as np

from learnable_typewriter.evaluate.quantitative.sprite_matching.metrics import error_rate 
from learnable_typewriter.evaluate.quantitative.sprite_matching.data import Data, Dataset
from learnable_typewriter.evaluate.quantitative.sprite_matching.trainer import TrainerBatch, Trainer

def metrics_to_average_sub(v):
    return {p: (q if p not in {'cer', 'wer', 'ser'} else np.mean(q)) for p, q in v.items()}

def metrics_to_average(obj):
    return {k: metrics_to_average_sub(v) for k, v in obj.items()}

def er_evaluate(
        trainer,
        mapping=None,
        dataloader_batch_size=16,
        matching_model_batch_size=256,
        lr=1,
        train_percentage=0.1,
        batch=True,
        cer_num_epochs=10,
        verbose=True,
        average=True,
        space_threshold=10,
        eval_train=True
    ):

    trainer.log(f'Unsupervised Evaluation on {train_percentage}% of the data with a batch size of {trainer.batch_size}', eval=True)

    if mapping is None:
        trainer.log('Inferring mapping', eval=True)
        train_loader = trainer.get_dataloader(split='train', percentage=train_percentage, batch_size=trainer.batch_size, num_workers=trainer.n_workers, remove_crop=True)
        TrainerClass = (TrainerBatch if batch else Trainer)
        data_train = Data(trainer, chain(*train_loader), tag=('er-build-dataset-train' if verbose else None))
        dataset = Dataset(data_train, A=len(trainer.transcribe_dataset), S=len(trainer.model.sprites))
        with torch.enable_grad():
            tr = TrainerClass(dataset, device=trainer.device, verbose=verbose)
            tr.train(num_epochs=cer_num_epochs, lr=lr, batch_size=matching_model_batch_size, print_progress=verbose)
        mapping = {s: trainer.transcribe_dataset[a] for s, a in tr.mapping.items()}
        trainer.log(f'Inferred mapping: {mapping}', eval=True)

    output = {}
    loaders_ = {
        'train_loader': trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True, remove_crop=True),
        'test_loader': trainer.test_loader
    }

    map_gt = dict(trainer.transcribe_dataset)
    map_gt[-1] =  '_'
    for split, loaders in loaders_.items():
        for loader in loaders:
            output[(loader.dataset.alias, split.split('_')[0])] = error_rate(Data(trainer, loader), verbose=verbose, average=average, delim=loader.dataset.space, sep=loader.dataset.sep, map_pd=mapping, map_gt=map_gt)

    output = {'metrics': output, 'mapping': mapping}
    output.update(loaders_) 
    return output

######### TODO REFACTOR -> move all the sampling part on inference #############
def best_path(s):
    return torch.argmax(s['selection'], dim=0)

def cer_aggregate(x, k):
    x = torch.unique_consecutive(x, dim=-1)
    x = x.tolist()
    return [e//2 for e in x if e != k]

def inference(trainer, sample):
    widths = torch.tensor(sample['w'])
    s = trainer.model.selection_head(sample)
    n_cells = s['selection'].size()[-1]
    k = s['selection'].size()[0]-1
    true_widths_pos = trainer.model.true_width_pos(sample['x'], widths, n_cells)
    s = best_path(s)
    s = [cer_aggregate(s[i][:true_widths_pos[i]], k) for i in range(s.size()[0])]
    return s

def er_evaluate_supervised(trainer, algo='best-path', verbose=False, average=True, eval_train=True, dataloader_batch_size=16, splits=None):
    loaders_ = {'val': trainer.val_loader, 'test': trainer.test_loader}
    if eval_train:
        loaders_['train'] = trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True, remove_crop=True)

    output = {}
    map_pd, map_gt = trainer.transcribe, dict(trainer.transcribe)
    map_gt[-1] = '_'
    for split, loaders in loaders_.items():
        if splits is not None and split not in splits:
            continue

        for loader in loaders:
            samples = []
            for sample in loader:
                samples += list(zip(inference(trainer, sample), sample['y']))

            output[(loader.dataset.alias, split)] = error_rate(samples, verbose=verbose, average=average, delim=loader.dataset.space, sep=loader.dataset.sep, map_pd=map_pd, map_gt=map_gt)

    output = {'metrics': output, 'mapping': trainer.transcribe}
    output.update(loaders_) 
    return output

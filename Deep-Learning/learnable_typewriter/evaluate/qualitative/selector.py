# Needs cleaning / possibly through away the html
from tqdm import tqdm
import numpy as np
import io, os, PIL
from os.path import join
from dominate.tags import img

import torch

import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d

from learnable_typewriter.utils.image import plot_image, img
from learnable_typewriter.evaluate.qualitative.decompositor import Decompositor, DecompositorSupervised


class QualitativeSelector(object):
    def __init__(self, trainer, dataloader, reduction='mean', verbose=False):
        self.trainer = trainer
        self.model = trainer.model
        self.tensorboard = trainer.tensorboard

        self.reduction = reduction
        self.verbose = verbose

        self.device = trainer.device
        self.dataloader = dataloader
        self.decompositor = Decompositor(trainer)

        self.metrics_ = None
        self.metrics_offline = False
        self.metric_name = 'reconstruction'

        self.x = {}
        self.segmentations = {}
        self.reconstructions = {}
        self.mask_sequences = {}
        self.tsf_proto_sequences = {}
        self.labels = {}
        self.predictions = {}

        self.criterion_ = torch.nn.MSELoss(reduction='none')

        self.run_dir = trainer.run_dir
        self.output_folder = 'qualitative'

    def load_offline(self, offline):
        self.metrics_ = offline['metrics']
        self.labels = offline['labels']
        if isinstance(self.labels, list):
            self.labels = {i: l for i, l in enumerate(self.labels)}

        self.predictions = offline['predictions']
        if isinstance(self.predictions, list):
            self.predictions = {i: l for i, l in enumerate(self.predictions)}
        self.metric_name = offline.get('name', 'metric')
        self.metrics_offline = True

    @property
    def iteration(self):
        return self.trainer.cur_iter

    def set_metrics(self, metrics):
        self.metrics_ = metrics

    def set_labels(self, labels):
        self.labels = labels
        if isinstance(self.labels, list):
            self.labels = dict(enumerate(self.labels))

    def set_predictions(self, predictions):
        self.predictions = predictions
        if isinstance(self.predictions, list):
            self.predictions = dict(enumerate(self.predictions))

    def criterion_mse(self, x, xp):
        dist = self.criterion_(x, xp)
        if self.reduction == 'mean':
            return dist.flatten(2).mean(2).mean(1)
        elif self.reduction == 'sum':
            return dist.flatten(2).sum(2).sum(1)
        else:
            raise NotImplementedError

    @property
    def metrics(self):
        if self.metrics_ is None:
            self.metrics_ = self.calculate_criterion_()
        return self.metrics_

    def get_object_label(self, idx):
        try: 
            x = self.dataloader.dataset[idx]
        except KeyError:
            x = self.dataloader.dataset.dataset[idx]

        if idx not in self.labels:
            try:
                self.labels[idx] = x[1]
            except KeyError:
                self.labels[idx] = ''

        return x

    @torch.no_grad()
    def get_segment_from_idx(self, idxs):
        for idx in idxs:
            x = self.get_object_label(idx)[0].unsqueeze(0).to(self.device)
            y = self.decompositor(x)
            self.reconstructions[idx] = y['reconstruction'][0].cpu()
            self.segmentations[idx] = y['segmentation'][0].cpu()
            self.x[idx] = x[0].cpu()
            self.mask_sequences[idx] = y['mask_sequence'][0]
            self.tsf_proto_sequences[idx] = y['tsf_proto_sequence'][0]

    def prepare_idx(self, idxs):
        if self.metrics_offline:
            idxs = [idx for idx in idxs if idx not in self.reconstructions]
            self.get_segment_from_idx(idxs)

    @torch.no_grad()
    def calculate_criterion_(self):
        metrics = []

        iterator = self.dataloader
        if self.verbose:
            iterator = tqdm(iterator, desc='get_metric-for-qualitative')
        
        id = 0
        for data in iterator:
            x = data['x'].to(self.device)
            y = self.decompositor(x)
            reco, seg, mask_sequence, tsf_proto_sequence = y['reconstruction'], y['segmentation'], y['mask_sequence'], y['tsf_proto_sequence']
            scores = self.criterion_mse(x, reco)
            for i in range(reco.size()[0]):
                self.reconstructions[id] = reco[i].cpu()
                self.x[id] = x[i].cpu()
                self.segmentations[id] = seg[i].cpu()
                self.mask_sequences[id] = mask_sequence[i]
                self.tsf_proto_sequences[id] = tsf_proto_sequence[i]
                metrics.append(scores[i].item())
                id += 1

        return np.array(metrics)

    def worst(self, k=1):
        return self.percentile(p=100, k=k)

    def best(self, k=1):
        return self.percentile(p=0, k=k)

    def median(self, k=1):
        return self.percentile(p=50, k=k)

    def percentile(self, p, k=1):
        arr = self.metrics
        pcen = np.percentile(arr, p)
        idxs = (np.asarray([np.abs(i - pcen) for i in arr])).argsort()[:k]
        return idxs

    def export(self, tag):
        if self.trainer.tensorboard is not None:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img = np.array(PIL.Image.open(img_buf).convert('RGB'))
            self.trainer.tensorboard.add_image(f'qualitative/{tag}', img, self.iteration, dataformats='HWC')

        plt.close()

    def percentile_plot(self, tag):
        data = []
        for p in range(0, 101):
            data.append((p, np.percentile(self.metrics, p)))
        x, y = zip(*data)
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.figure()
        plt.plot(x, y)
        plt.title('Percentile Plot')
        plt.ylabel('Error')
        plt.xlabel('Percentile')

        self.export(tag)

    def plot_by_index_(self, i, segmentation=False, mask_sequence=False, tsf_proto_sequence=False):
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rc('font', size=14)
        fig, axarr = plt.subplots(nrows=2 + int(segmentation) + int(mask_sequence) + int(tsf_proto_sequence), ncols=1)


        plot_image(axarr[0], img(self.x[i]), title='original', with_stick=True)

        m = round(self.metrics[i], 6)
        plot_image(axarr[1], img(self.reconstructions[i]), title=f'{self.metric_name}: {m}', with_stick=True)

        if segmentation:
            plot_image(axarr[2], img(self.segmentations[i]), title=f'segmentation', with_stick=True)

        if mask_sequence:
            plot_image(axarr[3], img(self.mask_sequences[i]), title=f'mask sequence', with_stick=False)

        if tsf_proto_sequence:
            plot_image(axarr[4], img(self.tsf_proto_sequences[i]), title=f'tsf prototype sequence', with_stick=False)

        if self.labels is not None or self.predictions is not None:
            label = self.labels.get(i, '').replace('$', '\$')
            prediction = self.predictions.get(i, '').replace('$', '\$')
            suptitle = 'GT: ' + label + '\n' + 'PRED: ' + prediction
            fig.suptitle(suptitle, fontsize=16)

        return fig

    def plot_by_indexes(self, idxs, tag, segmentation=False, mask_sequence=False):
        self.prepare_idx(idxs)
        for i in idxs:
            self.plot_by_index_(i, segmentation, mask_sequence)
            self.export(f"{tag}/{i}")

    def save_by_indexes(self, idxs, tag, segmentation=False, mask_sequence=False, tsf_proto_sequence=False):
        self.prepare_idx(idxs)
        folder = join(self.run_dir, self.output_folder, tag)
        os.makedirs(folder, exist_ok=True)
        for i in idxs:
            fig = self.plot_by_index_(i, segmentation, mask_sequence, tsf_proto_sequence)
            fig.savefig(join(folder, f"{i}.png"), dpi=200)
            plt.close(fig)

    def get_img(self, idxs):
        return [self.x[i] for i in idxs]

    def get_label(self, idxs):
        return [self.labels[i] for i in idxs]

    def get_reco(self, idxs):
        return [self.reconstructions[i] for i in idxs]

    def get_metric(self, idxs):
        return [self.metrics[i] for i in idxs]

class InferenceVisualization(QualitativeSelector):
    def __init__(self, trainer, dataloader, tag, saving_path='', reduction='mean', verbose=False):
        super().__init__(trainer, dataloader, reduction, verbose)
        self.saving_path = saving_path
        self.tag = tag
        self.shift = self.trainer.model.window.W_cell

    def save_error_smooth(self, idx, nostick=False, smoothing=True, sigma=3):
        """ compute a smooth reconstruction error between the input and reconstruction images """
        x, reco = self.x[idx], self.reconstructions[idx]
        C, H, W = x.shape
        list_value = []
        list_center_pixel = []
        for i in range(W - self.shift):
            list_center_pixel += [i + (self.shift // 2)]
            x_c = x[:, :, i:i + self.shift]
            r_c = reco[:, :, i:i + self.shift]
            diff = (r_c - x_c) ** 2
            error_i = diff.mean()
            error_i_np = error_i.cpu().detach().numpy()
            list_value += [error_i_np]
        if smoothing:
            list_value = gaussian_filter1d(list_value, sigma)

        h = 2
        f, ax = plt.subplots(figsize=(h * W / H, h))

        plt.xlim([0, W])
        plt.plot(list_center_pixel, list_value, color='r', linewidth=3)
        if not nostick:
            matplotlib.rcParams.update({'font.size': 22})
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            plt.rc('axes', labelsize=20)
            plt.xlabel('Pixel')
            plt.ylabel('MSE loss')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        path_output = join(self.saving_path, f"{idx}")
        fig_name = join(path_output, 'smoothed_local_error.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    def save_to_png(self, i):
        path = join(self.saving_path, f"{i}")
        os.makedirs(path, exist_ok=True)
        img(self.x[i]).save(join(path, f'ground_truth.png'))
        img(self.reconstructions[i]).save(join(path, f'reconstruction.png'))
        img(self.mask_sequences[i]).save(join(path, f'mask_sequence.png'))
        img(self.tsf_proto_sequences[i]).save(join(path, f'tsf_prototype_sequence.png'))
        img(self.segmentations[i]).save(join(path, f'segmentation.png'))
        self.save_error_smooth(i, nostick=True)

    def save_by_indexes_individually(self, idxs):
        idxs = [int(i) for i in idxs]
        self.prepare_idx(idxs)
        for i in idxs:
            self.save_to_png(i)



class QualitativeSelectorSupervised(QualitativeSelector):
    def __init__(self, trainer, dataloader, reduction='mean', verbose=False):
        super().__init__(trainer, dataloader, reduction, verbose)
        self.decompositor = DecompositorSupervised(trainer)

    @torch.no_grad()
    def get_segment_from_idx(self, idxs):
        for idx in idxs:
            # x = self.get_object_label(idx)[0].unsqueeze(0).to(self.device)
            x, label = self.get_object_label(idx)
            width = x.size(-1)
            x = x.unsqueeze(0).to(self.device)
            label = [label]
            width = [width]

            y = self.decompositor(x,label,width)
            self.reconstructions[idx] = y['reconstruction'][0].cpu()
            self.segmentations[idx] = y['segmentation'][0].cpu()
            self.x[idx] = x[0].cpu()
            self.mask_sequences[idx] = y['mask_sequence'][0]
            self.tsf_proto_sequences[idx] = y['tsf_proto_sequence'][0]

    @torch.no_grad()
    def calculate_criterion_(self):
        metrics = []

        iterator = self.dataloader
        if self.verbose:
            iterator = tqdm(iterator, desc='get_metric-for-qualitative')
        
        id = 0
        for data in iterator:
            x = data['x'].to(self.device)
            y = self.decompositor(x, data['y'], data['w'],supervised=False)
            reco, seg, mask_sequence, tsf_proto_sequence = y['reconstruction'], y['segmentation'], y['mask_sequence'], y['tsf_proto_sequence']
            scores = self.criterion_mse(x, reco)
            for i in range(reco.size()[0]):
                self.reconstructions[id] = reco[i].cpu()
                self.x[id] = x[i].cpu()
                self.segmentations[id] = seg[i].cpu()
                self.mask_sequences[id] = mask_sequence[i]
                self.tsf_proto_sequences[id] = tsf_proto_sequence[i]
                metrics.append(scores[i].item())
                id += 1

        return np.array(metrics)


class InferenceVisualizationSupervised(QualitativeSelectorSupervised):
    def __init__(self, trainer, dataloader, tag, saving_path='', reduction='mean', verbose=False):
        super().__init__(trainer, dataloader, reduction, verbose)
        self.saving_path = saving_path
        self.tag = tag
        self.shift = self.trainer.model.window.W_cell

    def save_error_smooth(self, idx, nostick=False, smoothing=True, sigma=3):
        """ compute a smooth reconstruction error between the input and reconstruction images """
        x, reco = self.x[idx], self.reconstructions[idx]
        C, H, W = x.shape
        list_value = []
        list_center_pixel = []
        for i in range(W - self.shift):
            list_center_pixel += [i + (self.shift // 2)]
            x_c = x[:, :, i:i + self.shift]
            r_c = reco[:, :, i:i + self.shift]
            diff = (r_c - x_c) ** 2
            error_i = diff.mean()
            error_i_np = error_i.cpu().detach().numpy()
            list_value += [error_i_np]
        if smoothing:
            list_value = gaussian_filter1d(list_value, sigma)

        h = 2
        f, ax = plt.subplots(figsize=(h * W / H, h))

        plt.xlim([0, W])
        plt.plot(list_center_pixel, list_value, color='r', linewidth=3)
        if not nostick:
            matplotlib.rcParams.update({'font.size': 22})
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            plt.rc('axes', labelsize=20)
            plt.xlabel('Pixel')
            plt.ylabel('MSE loss')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        path_output = join(self.saving_path, f"{idx}")
        fig_name = join(path_output, 'smoothed_local_error.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    def save_to_png(self, i):
        path = join(self.saving_path, f"{i}")
        os.makedirs(path, exist_ok=True)
        img(self.x[i]).save(join(path, f'ground_truth.png'))
        img(self.reconstructions[i]).save(join(path, f'reconstruction.png'))
        img(self.mask_sequences[i]).save(join(path, f'mask_sequence.png'))
        img(self.tsf_proto_sequences[i]).save(join(path, f'tsf_prototype_sequence.png'))
        img(self.segmentations[i]).save(join(path, f'segmentation.png'))
        self.save_error_smooth(i, nostick=True)

    def save_by_indexes_individually(self, idxs):
        idxs = [int(i) for i in idxs]
        self.prepare_idx(idxs)
        for i in idxs:
            self.save_to_png(i)
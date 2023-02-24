from math import ceil
import numpy as np
import torch
from einops import rearrange
from torch import nn, from_numpy
from torch.distributions import Beta

class CTC(torch.nn.Module):
    def __init__(self, blank, zero_infinity=True, reduction='mean'):
        super().__init__()
        self.loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def __call__(self, log_probs, y, input_lengths, target_lengths):
        return self.loss(log_probs, y, input_lengths.to(log_probs.device), target_lengths.to(log_probs.device))

class Loss(object):
    def __init__(self, model, cfg):
        self.model = model
        self._l2 = nn.MSELoss(reduction='none')
        self.ctc_factor = cfg['ctc_factor']
        if self.ctc_factor > 0:
            self.ctc = CTC(blank=len(model.sprites)//2)

        # Gaussian Weighting
        self.__init_gaussian__(cfg)

    def __init__blank_dummy__(self):
        val = 1e-9
        self.blank_dummy = torch.zeros((len(self.model.sprites)+1)) + torch.log(torch.Tensor([val/len(self.model.sprites)]))
        self.blank_dummy[self.model.empty_sprite_id] = torch.log(torch.Tensor([1-val])).item()

    @property
    def supervised(self):
        return self.model.supervised

    def __init_beta__(self, device):
        self.beta_dist = Beta(torch.tensor(2., device=device), torch.tensor(2., device=device))

    def __init_gaussian__(self, cfg):
        self.sigma_gaussian = cfg['sigma_gaussian']
        if self.sigma_gaussian:
            scale = (self.sigma_gaussian * 2 * np.pi)
        else:
            self.sigma_gaussian, scale = self.model.window.w * 2, 1

        self.model.log(f'Initializign gaussian with sigma={self.sigma_gaussian}, scale={scale}')
        self.global_loss_weight = gaussian_window_w(
            size_x=self.model.canvas_size[0], size_y=self.model.canvas_size[1],
            sigma_h=self.sigma_gaussian, windows_size=self.model.window.w * 2,
            normalized=not self.sigma_gaussian, scale=scale)
        
        self.l2_norm = self.global_loss_weight.sum().detach().item()

    def l2(self, gt, pred):
        B, C, H, W = pred.size()
        if gt['cropped']:
            mask = (self.global_loss_weight if self.global_loss_weight is not None else 1)
            norm = (self.l2_norm if self.global_loss_weight is not None else H*W)
            return (self._l2(pred, gt['x'])*mask).sum()/(norm*C*B)
        else:
            mask = self.get_mask_width(gt['x'], torch.tensor(gt['w']))
            return (self._l2(pred, gt['x'])*mask).sum(-1).mean(2).mean() 

    def reg_ctc(self, x):
        return self.ctc_factor > 0 and x['supervised']

    def get_mask_width(self, gt, widths):
        mask_widths = torch.zeros_like(gt)
        for b in range(len(gt)):
            mask_widths[b, :, :, :widths[b]] = 1/widths[b]
        return mask_widths

    def __call__(self, gt, pred):
        output = {}

        loss = self.l2(gt, pred['reconstruction'])
        output['reco_loss'] = loss.detach().item()

        if self.reg_ctc(gt):
            n_cells = self.model.transform_layers_.size(-1)
            transcriptions_padded, true_lengths = self.model.process_batch_transcriptions(gt['y'])
            true_widths_pos = self.model.true_width_pos(gt['x'], torch.Tensor(gt['w']), n_cells)
            ctc_loss = self.ctc_factor*self.ctc(pred['log_probs'], transcriptions_padded, true_widths_pos, true_lengths)

            output['ctc_loss'] = ctc_loss.detach().item()
            loss = loss + ctc_loss

        output['total'] = loss
        return output


def horizontal_gauss_map(size_x, size_y=None, sigma_y=5, normalized=True, scale=1):
    if size_y == None:
        size_y = size_x

    assert isinstance(size_x, int)
    assert isinstance(size_y, int)

    y = np.arange(0, size_y, dtype=float)[:, np.newaxis]
    y -= size_y // 2

    exp_part = y ** 2 / (2 * sigma_y ** 2)
    gauss = scale / (2 * np.pi * sigma_y) * np.exp(-exp_part)
    gauss = np.tile(gauss.reshape(1, size_y), (size_x, 1))
    if normalized:
        gauss = (gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss))
    return gauss


def gaussian_window_w(size_x, size_y, sigma_h, windows_size, normalized=True, scale=1, clamp=None):
    """It will return a gaussian weighting for the input image to not care about the border."""
    h_gauss_weights = horizontal_gauss_map(size_x, windows_size, sigma_h, normalized=normalized, scale=scale)
    if clamp is not None:
        h_gauss_weights = np.clip(h_gauss_weights, clamp, None)
    img_bigger = np.ones((size_x, size_y))*np.max(h_gauss_weights)
    img_bigger[:, 0:windows_size // 2] = h_gauss_weights[:, 0:windows_size // 2]
    img_bigger[:, -(windows_size // 2):] = h_gauss_weights[:, windows_size // 2:windows_size]
    return from_numpy(img_bigger)

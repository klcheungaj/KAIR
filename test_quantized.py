import os.path
import logging
import re

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_model

from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FakeQuantizeBase,
    default_fake_quant,
    default_dynamic_fake_quant,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    default_fused_act_fake_quant,
    default_fused_wt_fake_quant,
    FusedMovingAvgObsFakeQuantize,
    default_fused_per_channel_wt_fake_quant,
    default_embedding_fake_quant,
    default_embedding_fake_quant_4bit,
    fused_wt_fake_quant_range_neg_127_to_127,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)

from torch.ao.quantization.observer import (
    _PartialWrapper,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    PlaceholderObserver,
    ReuseInputObserver,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_float_qparams_observer_4bit,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_weight_observer,
    weight_observer_range_neg_127_to_127,
    per_channel_weight_observer_range_neg_127_to_127,
    default_reuse_input_observer,
    ObserverBase,
)

'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
(github: https://github.com/cszn/KAIR)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
testing demo for RRDB-ESRGAN
https://github.com/xinntao/ESRGAN
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
}
# --------------------------------------------
|--model_zoo             # model_zoo
   |--rrdb_x4_esrgan     # model_name, optimized for perceptual quality      
   |--rrdb_x4_psnr       # model_name, optimized for PSNR
|--testset               # testsets
   |--set5               # testset_name
   |--srbsd68
|--results               # results
   |--set5_rrdb_x4_esrgan# result_name = testset_name + '_' + model_name
   |--set5_rrdb_x4_psnr 
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = '51000_Gq'        # 'rrdb_x4_esrgan' | 'rrdb_x4_psnr'
    testset_name = 'Set14'                # test set,  'set5' | 'srbsd68'
    need_degradation = False              # default: True
    x8 = False                           # default: False, x8 to boost performance
    sf = 4
    show_img = False                     # default: False




    task_current = 'sr'       # 'dn' for denoising | 'sr' for super-resolution
    n_channels = 3            # fixed
    model_pool = 'model_zoo'  # fixed
    testsets = '/home/kenny/dataset'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name, 'LR') # L_path, for Low-quality images
    H_path = os.path.join(testsets, testset_name, 'HR') # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_rrdb_arch import SeparableRRDBNet as net
    model = net(num_in_ch=3, num_out_ch=3, scale=4, num_feat=32, num_block=5, num_grow_ch=16, quan=True)
    model.qconfig = torch.ao.quantization.QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                            quant_min=0,
                                                                            quant_max=255,
                                                                            reduce_range=False),
                                                    weight=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                            quant_min=-127,
                                                                            quant_max=127,
                                                                            dtype=torch.qint8,
                                                                            qscheme=torch.per_channel_symmetric))
    model = torch.ao.quantization.prepare_qat(model, inplace=True)
    model = torch.quantization.convert(model, inplace=True)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    # for key, param in model.state_dict().items():
    #     print(f"{key}: \n{param.int_repr()}")
    print(model.conv_first.weight().int_repr())
    print(model.conv_first.bias())
    # model = torch.load(model_path)

    print(type(model))
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)
        # degradation process, bicubic downsampling + Gaussian noise
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = util.imresize_np(img_L, 1/sf)
            # np.random.seed(seed=0)  # for reproducibility
            # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3, sf=sf)

        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()
            img_H = util.modcrop(img_H, sf)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

            if np.ndim(img_H) == 3:  # RGB image
                img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
        if np.ndim(img_H) == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('Average PSNR/SSIM( Y ) - {} - x{} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':

    main()

import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.utils_fit import VQGAN_Decoder
from vq_gan.utils import config
import numpy as np
from ddpm import Diffusion
from scipy.io import savemat
import cv2


if __name__ == "__main__":
    save_number = 5
    size = 5
    vqgan_decoder = VQGAN_Decoder(config=config,
                                  checkpoint='./checkpoint/lightning_logs/version_2/checkpoints/recon_loss=0.02.ckpt')

    ddpm = Diffusion(vqgan_decoder)
    pre, outs, outs_c = ddpm.generate_1x1_image(save_number, size)
    #outs_c_rounded = np.round(outs_c, decimals=3)
    #outs_c_rounded = np.round(outs_c, decimals=4)
    #outs_c_rounded = np.round(outs_c, decimals=5)
    #outs_c = outs_c_rounded


    for i, save_ in enumerate(pre):
        #savemat('./results/predict_out/test2/' + str(i) + '.mat',
        #        {'pattern': save_.astype(np.int8), 'parameter': outs[i], 'curvature': outs_c[i]})

        savemat('./results/predict_out/0801/test3/' + str(i) + '.mat',
                {'pattern': save_.astype(np.int8), 'parameter': outs[i], 'curvature': outs_c[i]})


    # plt.imshow(pre)
    # plt.show()



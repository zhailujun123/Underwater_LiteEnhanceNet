import numpy as np
import torchvision
from torchvision import transforms
import argparse
import time
from tqdm import tqdm
from model import *
from dataloader import myDataSet
from metrics_calculation import *
import os
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import imageio
from utils.uqim_utils import getUIQM
from utils.ssm_psnr_utils import getSSIM, getPSNR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = [
    "test",
    "setup",
    "testing",
]

@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            generate_img = test_model(img)
            torchvision.utils.save_image(generate_img, config.output_images_path + name[0])

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = torch.load(config.snapshot_path).to(config.device)
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])
    test_dataset = myDataSet(config.test_images_path, None, transform, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("Test Dataset Reading Completed.")
    return test_dataloader, model

def testing(config):
    ds_test, model = setup(config)
    test(config, ds_test, model)

def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contains ground-truth images
        - gen_dir contains generated images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # Get SSIM on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # Get PSNR on L channel (SOTA norm)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

def SSIMs_PSNRs_raw(gtr_dir, raw_dir, im_res=(256, 256)):
    """
        - gtr_dir contains ground-truth images
        - raw_dir contains raw input images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    raw_paths = sorted(glob(join(raw_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, raw_path in zip(gtr_paths, raw_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        raw_f = basename(raw_path).split('.')[0]
        if (gtr_f == raw_f):
            r_im = Image.open(gtr_path).resize(im_res)
            raw_im = Image.open(raw_path).resize(im_res)
            # Get SSIM on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(raw_im))
            ssims.append(ssim)
            # Get PSNR on L channel (SOTA norm)
            r_im = r_im.convert("L")
            raw_im = raw_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(raw_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

def measure_UIQMs(dir_name, im_res=(256, 256)):
    """
    - dir_name contains images to calculate UIQM
    """
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

if __name__ == '__main__':
    models = "./snapshots/model_epoch_" + str(99) + ".ckpt"
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_path', type=str, default=models, help='Model snapshot path')
    parser.add_argument('--test_images_path', type=str, default="./data/raw/", help='Path of input images (underwater images) for testing')
    parser.add_argument('--output_images_path', type=str, default='./data/output/', help='Path to save generated images.')
    parser.add_argument('--raw_images_path', type=str, default='./data/raw/', help='Path to raw input images for comparison')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size, default: 1")
    parser.add_argument('--resize', type=int, default=256, help="Resize images to this size, default: 256x256")
    parser.add_argument('--calculate_metrics', type=bool, default=False, help="Calculate PSNR, SSIM and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./data/label/", help='Path of ground truth images (clear images)')

    print("-------------------testing---------------------")
    config = parser.parse_args()
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)

    start_time = time.time()
    testing(config)
    print("Total testing time:", time.time() - start_time)

    # Directories for images
    GEN_im_dir = "./data/output/"
    GTr_im_dir = "./data/GT/"
    Raw_im_dir = config.raw_images_path

    # Calculate UIQM for generated images
    gen_uqims = measure_UIQMs(GEN_im_dir)
    print("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

    # Calculate UIQM for raw images
    raw_uqims = measure_UIQMs(Raw_im_dir)
    print("Raw UQIM >> Mean: {0} std: {1}".format(np.mean(raw_uqims), np.std(raw_uqims)))

    # Calculate SSIM and PSNR for generated images
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(GTr_im_dir, GEN_im_dir)
    print("Generated SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
    print("Generated PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

    # Calculate SSIM and PSNR for raw images
    SSIM_measures_raw, PSNR_measures_raw = SSIMs_PSNRs_raw(GTr_im_dir, Raw_im_dir)
    print("Raw SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures_raw), np.std(SSIM_measures_raw)))
    print("Raw PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures_raw), np.std(PSNR_measures_raw)))

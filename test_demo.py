import torch
from torchvision import utils
from models.Network import Network as HDRNet
import numpy as np
from pytorch_ssim import SSIM
from utils.metrics import PSNR
import os
import cv2
from PIL import Image
import json
from tqdm import tqdm
import argparse


def load_image(name_jpg, mode=1):
    return np.asarray(Image.open(name_jpg).convert('RGB')).astype(np.float32) / 255.0


def get_novel_size(ww, hh, size):
    if ww > hh:
        ratio = size / ww
        nw, nh = round(ratio * ww), round(ratio * hh)
        return nw, nh
    else:
        ratio = size / hh
        nw, nh = round(ratio * ww), round(ratio * hh)
        return nw, nh


def perform_test_size(h, size1, size2):
    if size1 <= h < size2:
        return size1
    else:
        return 0


def adapt_size(h, w):
    nh = 0,
    nw = 0
    sizes = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
             1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984,
             2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624]

    for i in range(len(sizes) - 1):
        nh = perform_test_size(h, sizes[i], sizes[i + 1])
        if nh != 0:
            break

    for i in range(len(sizes) - 1):
        nw = perform_test_size(w, sizes[i], sizes[i + 1])
        if nw != 0:
            break

    return nw, nh


def get_input_tensors(images, data_path):
    input_tensors = []
    names = []

    for i in range(len(images)):
        current_path = os.path.join(data_path, images[i])
        current_image = load_image(current_path)
        H, W, C = current_image.shape

        ww, hh = get_novel_size(W, H, 512)

        nw, nh = adapt_size(hh, ww)

        resized = cv2.resize(current_image, (nw, nh))

        # get torch tensor
        current_tensor = torch.from_numpy(resized).permute(2, 0, 1)

        # add to list
        names.append(images[i])
        input_tensors.append(current_tensor)

    return input_tensors, names


def load_and_transform_image(image_path):
    current_path = os.path.join(data_path, image_path)
    current_image = load_image(current_path)
    H, W, C = current_image.shape

    ww, hh = get_novel_size(W, H, 512)

    nw, nh = adapt_size(hh, ww)

    resized = cv2.resize(current_image, (nw, nh))

    # get torch tensor
    current_tensor = torch.from_numpy(resized).permute(2, 0, 1)

    # add to list
    return current_tensor



def load_config(file):
    """
    takes as input a file path and returns a configuration file
    that contains relevant information to the training of the NN
    :param file:
    :return:
    """

    # load the file as a raw file
    loaded_file = open(file)

    # conversion from json file to dictionary
    configuration = json.load(loaded_file)

    # returning the file to the caller
    return configuration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weight_path", type=str, default="checkpoints/checkpoint.pth")
    args = parser.parse_args()

    os.makedirs(args.output_dir)
    # weights of the models being loaded
    weights = torch.load(args.weight_path, map_location='cuda:0')

    # models creation
    model = HDRNet()
    model = torch.nn.DataParallel(model)
    model.to('cuda:0')
    model.load_state_dict(weights, strict=False)
    model = model.eval()

    images = os.listdir(args.input_dir)

    input_tensors, names = get_input_tensors(images, args.input_dir)

    #for i in range(len(input_tensors)):
    for name, input_image in tqdm(zip(names, input_tensors), desc="Running ECCMNet...", total=len(names)):
        input_image = torch.unsqueeze(input_image, 0)

        if torch.cuda.is_available():
            input_image = input_image.cuda()
            normalized_input = (input_image - 0.5) / 0.5

        with torch.no_grad():
            out, _ = model(normalized_input)

        display_data = torch.cat(
            [out], dim=0)
        save = os.path.join(args.output_dir, name)
        utils.save_image(display_data, save, nrow=1, padding=2, normalize=False)

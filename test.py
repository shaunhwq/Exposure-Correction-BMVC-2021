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


config = load_config('config.json')['config']

# dataset
data_path = config['test']['data_path']
weight_path = config['test']['weight_path']
output_path = config['test']['output_path']
target_path = config['test']['target_path']

# weights of the models being loaded
weights = torch.load(weight_path, map_location='cuda:0')

# models creation
model = HDRNet()
model = torch.nn.DataParallel(model)
model.to('cuda:0')
model.load_state_dict(weights, strict=False)
model = model.eval()

im_paths = [im_path for im_path in os.listdir(data_path) if im_path[0] != "."]

m_scores = [[], []]
metrics = [PSNR(max_value=1.0), SSIM()]

pbar = tqdm(im_paths, total=len(im_paths), desc="evaluating...")
for im_path in pbar:
    input_image = load_and_transform_image(im_path)
    input_image = torch.unsqueeze(input_image, 0)

    gt_name = "_".join(os.path.basename(im_path).split("_")[:-1]) + ".jpg"
    gt_image = load_and_transform_image(os.path.join(target_path, gt_name))
    gt_image = torch.stack([gt_image])

    if torch.cuda.is_available():
        input_image = input_image.cuda()
        gt_image = gt_image.cuda()
        normalized_input = (input_image - 0.5) / 0.5
        normalized_gt = gt_image


    with torch.no_grad():
        out, _ = model(normalized_input)

    for m_idx, metric in enumerate(metrics):
        m_scores[m_idx].append(metric(normalized_gt, out).item())
    
    pbar.set_description("evaluating... PSNR: {} | SSIM: {}".format(*[round(sum(m_score) / len(m_score), 3) for m_score in m_scores]))

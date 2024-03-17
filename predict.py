import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
import unet
from utils.utils import plot_and_save_pred, save_feature_map
from pathlib import Path
from typing import Union


output_feature_map = True
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def predict_img(net,
                full_img: Union[Image.Image, torch.Tensor],
                device,
                scale_factor=1,
                out_threshold=0.5) -> np.ndarray:
    net.eval()
    if isinstance(full_img, Image.Image):
        img = BasicDataset().preprocess(None, full_img, net.n_channels, scale_factor, is_mask=False)
        img = img.unsqueeze(0)
        img_size = full_img.size[1], full_img.size[0]   # H,W
    elif isinstance(full_img, torch.Tensor):
        img = full_img
        img_size = full_img.shape[-2:]
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, img_size, mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--name', help='Model name', required=True)
    parser.add_argument('--input_sar', nargs='+', help='Filenames of input sar images', required=True)
    parser.add_argument('--input_mask', nargs='+', help='Filenames of input mask images')
    parser.add_argument('--output', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_sars = args.input_sar
    if args.input_mask and len(args.input_mask):
        input_masks = args.input_mask
    else:
        input_masks = args.input_sar
    assert len(input_sars) == len(input_masks) > 0
    sar_files = []
    mask_files = []
    for i in range(len(input_sars)):
        input_sar, input_mask = input_sars[i], input_masks[i]
        if Path(input_sar).is_dir():
            for ff in list(Path(input_sar).glob('*.jpg')) + list(Path(input_sar).glob('*.png')):
                if '_mask' not in ff.name:
                    sar_files.append(ff)
                    if Path(input_mask).is_dir():
                        matches = list(Path(input_mask).glob(ff.stem+'_mask.png'))
                        if len(matches):
                            mask_files.append(matches[0])
                        else:
                            mask_files.append(None)
        elif Path(input_sar).is_file():
            sar_files.append(input_sar)
            if Path(input_mask).is_file():
                mask_files.append(input_mask)
            else:
                mask_files.append(None)
        else:
            logging.warning(f'not exist: {input_sar}')
    output = Path(args.output) if args.output else Path(args.input_sar[0])/'output'
    output.mkdir(exist_ok=True, parents=True)

    assert args.name in unet.models
    if args.name == "UNet":
        net = getattr(unet, args.name)(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    else:
        net = getattr(unet, args.name)(n_channels=args.channels, n_classes=args.classes, pretrained=False)
    net = net.to(memory_format=torch.channels_last)

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i in range(len(sar_files)):
        logging.info(f'Predicting image {sar_files[i]} ...')
        sar = Image.open(sar_files[i])
        if mask_files[i] is not None:
            true_mask = Image.open(mask_files[i])
            true_mask = BasicDataset().preprocess(mask_values, true_mask, net.n_channels, 1, is_mask=True).cpu().numpy()
        else:
            true_mask = None

        if output_feature_map:
            net.register_forward_hooks()

        matches = predict_img(net=net,
                           full_img=sar,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if output_feature_map:
            net.close_forward_hooks()
            dir_feature_map_output = Path(output)/'feature_map_output'
            dir_feature_map_output.mkdir(exist_ok=True)

            savepath = dir_feature_map_output/f'{i:04d}_input.png'
            image = torch.tensor(np.array(sar).transpose((2,0,1)))[:net.n_channels,:,:]
            save_feature_map(image, savepath)

            for n, (name, feature_map) in enumerate(net.feature_maps):
                savepath = dir_feature_map_output/f'{i:04d}_mod{n:02d}={name}.png'
                save_feature_map(feature_map, savepath)
                savepath = dir_feature_map_output/f'{i:04d}_mod{n:02d}={name}_fusion.png'
                feature_map = feature_map.mean(1, keepdim=True)
                save_feature_map(feature_map, savepath)
            net.clear_feature_maps()

        save_path = None if args.no_save else Path(output)/f'{sar_files[i].stem}.png'
        plot_and_save_pred(sar, matches, true_mask, save_path, args.viz)

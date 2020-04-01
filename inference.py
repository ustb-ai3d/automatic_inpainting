import os
import argparse
import glob
import numpy as np
import torch
from PIL import Image
from models.unet import UNet
from models.unet.utils import predict_img
from utils.selective_inpaint import search_inpaint_area, fill_inpaint_image_to_origin
from models.edge_connect.inpainting import inpainting

os.environ['TORCH_HOME'] = './'
device = torch.device('cuda')


def detect_noise_regions(image, args):
    # load noise segmentation network (U-Net)
    unet_model_path = os.path.join(args.checkpoints, 'unet', 'UNet.pth')
    net = UNet(n_channels=3, n_classes=1).to(device)
    net.load_state_dict(torch.load(unet_model_path))
    net.eval()

    # predict noise regions
    predict = predict_img(net, device, image)

    # search inpaint patches
    patches, labels, _, absolute_position, relative_position = search_inpaint_area(np.array(image),
                                                                                   np.array(predict.convert('RGB')))

    # save inpaint patches
    patches_dir = os.path.join(args.output, 'patches')
    labels_dir = os.path.join(args.output, 'labels')
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    filename = os.path.basename(args.input).split('.')[0]
    counter = 0
    for patch, label in zip(patches, labels):
        Image.fromarray(patch).save(os.path.join(patches_dir, '{}-{:0>3d}.png'.format(filename, counter)))
        Image.fromarray(label).save(os.path.join(labels_dir, '{}-{:0>3d}.png'.format(filename, counter)))
        counter += 1
    return patches_dir, labels_dir, absolute_position, relative_position


def inpaint_noise_regions(image, patches_dir, labels_dir, absolute_position, relative_position, args):
    # inpainting
    inpainted_patches_dir = os.path.join(args.output, 'inpainted_patches')
    os.makedirs(inpainted_patches_dir, exist_ok=True)
    inpainting(mode=2, conf={
        'checkpoints': os.path.join(args.checkpoints, 'edge_connect'),
        'input': patches_dir,
        'mask': labels_dir,
        'output': inpainted_patches_dir,
        'model': 3,
        'edge': None
    })

    # back filling
    inpainted_patches = sorted(glob.glob(inpainted_patches_dir + '/*.png'))
    for index, patch_path in enumerate(inpainted_patches):
        inpainted_patches[index] = np.array(Image.open(patch_path))
    target = fill_inpaint_image_to_origin(np.array(image), inpainted_patches, absolute_position, relative_position)
    return Image.fromarray(target)


def main(args):
    filename = os.path.basename(args.input).split('.')[0]
    image = Image.open(args.input).convert('RGB')

    # detect
    print('[Automatic Inpainting] Detect noise regions...')
    patches_dir, labels_dir, absolute_position, relative_position = detect_noise_regions(image, args)
    print(f'[Automatic Inpainting] Successfully detect [{len(absolute_position)}] noise regions to inpaint.')

    # inpaint
    print('[Automatic Inpainting] Inpainting noise regions...')
    target = inpaint_noise_regions(image, patches_dir, labels_dir, absolute_position, relative_position, args)
    print(f'[Automatic Inpainting] Successfully inpaint [{len(absolute_position)}] noise regions...')

    # cyclic validation strategy
    attempt = 3
    while attempt > 0:
        attempt -= 1
        # validate
        print(f'[Automatic Inpainting] Start cyclic validation strategy...')
        patches_dir, labels_dir, absolute_position, relative_position = detect_noise_regions(target, args)
        if len(absolute_position) == 0:
            break
        else:
            print('[Automatic Inpainting] Validate failed, try to inpaint again...')
            target = inpaint_noise_regions(target, patches_dir, labels_dir, absolute_position, relative_position, args)

    # save
    print('[Automatic Inpainting] Done!')
    target.save(os.path.join(args.output, f'{filename}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, required=False, default='./checkpoints',
                        help='The input image path.')
    parser.add_argument('--input', type=str, required=True, help='The input image path.')
    parser.add_argument('--output', type=str, required=False, default='./results', help='The output directory.')
    args = parser.parse_args()
    print(args)
    main(args)

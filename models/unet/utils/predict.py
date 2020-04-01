import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from .utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks


def predict_img(net, device, full_img, scale_factor=0.5, out_threshold=0.5):
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)

    X_left = X_left.to(device)
    X_right = X_right.to(device)

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_height),
            transforms.ToTensor()])

        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    mask = full_mask > out_threshold
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255

    return Image.fromarray(mask)

import numpy as np
from skimage import measure


def search_inpaint_area(_origin_img, mask_img, size=256):
    origin_img = np.copy(_origin_img)
    images = []
    masks = []
    image_with_masks = []

    relative_position = []
    absolute_position = []

    labels = measure.regionprops(measure.label(mask_img))
    for bbox in labels:
        x_min, y_min = bbox.centroid[1] - size // 2, bbox.centroid[0] - size // 2

        # x range limit
        if x_min < 0:
            x_min = 0
        elif x_min + size > origin_img.shape[1]:
            x_min = origin_img.shape[1] - size
        # y range limit
        if y_min < 0:
            y_min = 0
        elif y_min + size > origin_img.shape[0]:
            y_min = origin_img.shape[0] - size

        x_min, y_min = int(x_min), int(y_min)

        image = origin_img[y_min:y_min + size, x_min:x_min + size]
        mask = np.zeros(image.shape, dtype=np.uint8)
        # 寻找mask在本张图像和真实图像中的位置，方便进行还原
        # bbox: (y_min, x_min, y_max, x_max)
        rel_pos = (bbox.bbox[0] - y_min, bbox.bbox[1] - x_min, bbox.bbox[3] - y_min, bbox.bbox[4] - x_min)
        mask[rel_pos[0]:rel_pos[2], rel_pos[1]:rel_pos[3]] = 255
        image_with_mask = np.copy(image)
        image_with_mask[mask == 255] = 255

        images.append(image)
        masks.append(mask)
        image_with_masks.append(image_with_mask)
        absolute_position.append((bbox.bbox[0], bbox.bbox[1], bbox.bbox[3], bbox.bbox[4]))
        relative_position.append(rel_pos)

    return images, masks, image_with_masks, absolute_position, relative_position


def fill_inpaint_image_to_origin(origin_img, inpaint_images, absolute_position, relative_position):
    target_img = np.copy(origin_img)
    for inpaint_image, abs_bbox, rel_bbox in zip(inpaint_images, absolute_position, relative_position):
        patch = inpaint_image[rel_bbox[0]:rel_bbox[2], rel_bbox[1]:rel_bbox[3]]
        target_img[abs_bbox[0]:abs_bbox[2], abs_bbox[1]:abs_bbox[3]] = patch
    return target_img

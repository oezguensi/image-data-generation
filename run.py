import argparse
import ast
import datetime
import json
import os
import re
from glob import glob
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs-dir', type=str, default='assets/objects',
                        help='Directory containing subdirectories with label names which each contain masked out images of the objects')
    parser.add_argument('--save-dir', type=str, default='assets/result', help='Directory to save images and annotations to')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256], help='Output image size. Specify height and width seperated by whitespace')
    parser.add_argument('--num-imgs', type=int, default=10, help='Number of unaugmented images to produce')
    parser.add_argument('--bgs-dir', type=str, default=None, help='Directory containing random background images')
    parser.add_argument('--augs', type=str, nargs='*', default=['RandomHorizontalFlip(p=0.5)', 'RandomVerticalFlip(p=0.5)'],
                        help='`torchvision.transforms` augmentations to perform on the objects, in given order, seperated by whitespace')
    args = parser.parse_args()
    
    return args


def get_backround_images(num_imgs: int, img_size: Tuple[int, int], bgs_dir: str = None) -> List:
    """
    Generates random colored background images if no image directory is given
    :param num_imgs: Number of images to produce
    :param img_size: Size (height, width) of created images
    :param bgs_dir: Directory containing background images
    :return: `num_imgs` background images with given `img_size`
    """
    
    if bgs_dir is None:
        # Create images filled with random color if no background images are given
        cmap = plt.cm.get_cmap('hsv', num_imgs)
        colors = [(np.array(cmap(i)[:-1]) * 255).astype(np.uint8) for i in range(num_imgs)]
        bg_imgs = [Image.fromarray(np.full((*img_size, 3), color)) for color in colors]
    else:
        paths = glob(os.path.join(bgs_dir, '*.*'))
        if len(paths) >= num_imgs:
            paths = list(np.random.choice(paths, num_imgs, replace=False))
        else:
            paths += list(np.random.choice(paths, num_imgs - len(paths), replace=False))
        bg_imgs = [Image.open(path).resize(img_size) for path in paths]
    
    return bg_imgs


def place_objects_in_area(obj_sizes: List[Tuple[int, int]], area_size: Tuple[int, int], plot: bool = False) -> Tuple[
    List[Tuple[int, int, int, int]], List[int]]:
    """
    Randomly places objects with corresponding bounding boxes in a fixed sized area
    :param obj_sizes: Sizes (height, width) of objects to place in area
    :param area_size: Size (height, width) of the area
    :param plot: Whether to plot the positioned objects
    :return: Upper left corner (x, y) of the objects
    """
    area = np.ones(area_size)
    
    # Sort objects regarding to area in descending order
    obj_sizes = sorted(obj_sizes, key=lambda x: x[0] * x[1], reverse=True)
    
    bboxes, placed_obj_idxs = [], []
    for i, obj_size in enumerate(obj_sizes):
        tmp = area.copy()
        tmp[-obj_size[0]:, :] = 0
        tmp[:, -obj_size[1]:] = 0
        
        if len(bboxes) > 0:
            for bbox in bboxes:
                tmp[max(0, bbox[1] - obj_size[0]): bbox[3], max(0, bbox[0] - obj_size[1]): bbox[2]] = 0
        
        # If enough space is available, place object
        free_idxs = tmp.nonzero()
        if free_idxs[0].shape[0] > 0:
            rnd_int = np.random.randint(len(free_idxs[0]) - 1)
            x0, y0 = free_idxs[1][rnd_int], free_idxs[0][rnd_int]
            x1, y1 = x0 + obj_size[1], y0 + obj_size[0]
            
            bboxes.append((x0, y0, x1, y1))
            placed_obj_idxs.append(i)
    
    if plot:
        vis = Image.new('RGB', area_size)
        draw = ImageDraw.Draw(vis)
        for bbox in bboxes:
            draw.rectangle(bbox, fill=(255, 255, 255), width=2)
        
        vis.show()
    
    return bboxes, placed_obj_idxs


def augment_image(img: Image, augs: List[str]) -> Image:
    """
    Performs a list of string respresentations of torchvision.transforms functions and applies them onto an image
    :param img: Image to augment
    :param augs: List of string representation for each augmentation
    :return: Augmented image
    """
    
    tfs = []
    for aug in augs:
        name, params = aug.strip()[:-1].split('(')
        params = [f"'{param.split('=')[0]}':{param.split('=')[1]}" for param in params.split(',')]
        tfs.append(getattr(torchvision.transforms, name)(**ast.literal_eval(f"{{{','.join(params)}}}")))
    
    img = torchvision.transforms.Compose(tfs)(img)
    
    return img


def generate_annotations(imgs: List, bboxess: List[List[Tuple[int, int, int, int]]], labelss: List[str], save_dir: str):
    """
    Saves images and annotations in COCO format
    :param imgs: List of Pil images
    :param bboxess: List of lists containing bounding boxes
    :param labelss: List of lists containing labels
    :param save_dir: Directory to save images to
    """
    
    cats = [{'id': i, 'name': cat, 'supercategory': cat} for i, cat in enumerate(sorted(set([i for s in labelss for i in s])))]
    
    img_anns, obj_anns = [], []
    for i, (img, bboxes, labels) in enumerate(zip(imgs, bboxess, labelss)):
        img.save(os.path.join(save_dir, f'generated_{i}.png'))
        img_anns.append(
            {"id": i, "width": img.size[0], "height": img.size[1], "file_name": f'generated_{i}.png', "license": 0, "flickr_url": '', "coco_url": '',
             "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        
        for bbox, label in zip(bboxes, labels):
            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            area = width * height
            
            obj_anns.append(
                {"id": len(obj_anns), "image_id": i, "category_id": [cat['id'] for cat in cats if cat['name'] == label][0], "segmentation": [],
                 "area": int(area),
                 "bbox": [int(x0), int(y0), int(width), int(height)], "iscrowd": 0})
    
    with open(os.path.join(save_dir, 'annotations.json'), 'w') as f:
        json.dump({'images': img_anns, 'annotations': obj_anns, 'categories': cats}, f)


def main():
    """
    DONE - specify image size
    DONE - use images as background
    DONE - use random colors if no random background images are given
    DONE - do augmentations based on torchvision transform
    (- specify radius for spacing of objects (if possible, if they fit in the image))
    - specify if only bounding boxes/segmentations or both should be outputted
    """
    np.random.seed(1)
    
    args = parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    bg_imgs = get_backround_images(args.num_imgs, args.img_size, args.bgs_dir)
    
    paths = glob(os.path.join(args.objs_dir, '*', '*.png'))
    obj_imgs = [Image.open(path) for path in paths]
    labels = [os.path.basename(os.path.dirname(path)) for path in paths]
    
    bboxess, labelss = [], []
    # blend background and foreground images
    for bg_img in tqdm(bg_imgs, desc='Blending images'):
        obj_imgs_aug = [augment_image(obj_img, args.augs) for obj_img in obj_imgs] if args.augs is not None else obj_imgs
        obj_sizes = [obj_img.size[::-1] for obj_img in obj_imgs_aug]  # reverse to have height, width form
        
        bboxes, placed_obj_idxs = place_objects_in_area(obj_sizes, args.img_size)
        bboxess.append(bboxes)
        labelss.append([labels[idx] for idx in placed_obj_idxs])
        
        for bbox, idx in zip(bboxes, placed_obj_idxs):
            bg_img.paste(obj_imgs[idx], (bbox[0], bbox[1]), obj_imgs[idx])
    
    generate_annotations(bg_imgs, bboxess, labelss, args.save_dir)
    
    print()


if __name__ == '__main__':
    main()

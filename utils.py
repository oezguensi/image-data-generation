import ast
import json
import os
import datetime
import zipfile
from glob import glob
from shutil import copyfile
from typing import Tuple, List

import cv2
import numpy as np
import torchvision
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm


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


def get_segmentations(img: np.ndarray):
    """
    The COCO format for contours/polygons/segmentations is structured as follows:
    [[first object], [second object]] --> [[x1, y1, x2, y2, x3, y3, x4, y4, ...]]
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    segmentations = [cnt.flatten().tolist() for cnt in contours]
    
    return segmentations


def get_bounding_box(img: np.ndarray, threshold: int = 0, contours=None) -> Tuple[int, int, int, int]:
    """
    Gets the bounding box of a transparent image containing a single non-transparent object
    :param img: Either a transparent image with 4 dimensions where the last dimension is the alpha channel, or a binary mask
    :param threshold: Threshold for maximum alpha value to consider transparent. Necessary for the transparent case
    :return: Bounding box in the format x0, y0, x1, y1
    """
    
    if img.shape[-1] == 4:
        idxs = np.argwhere(img[:, :, -1] > threshold)
        y, x = idxs[:, 0], idxs[:, 1]
        x0, y0, x1, y1 = [np.min(x), np.min(y), np.max(x), np.max(y)]
    else:
        # rows = np.any(img, axis=1)
        # cols = np.any(img, axis=0)
        # x0, x1 = np.where(rows)[0][[0, -1]]
        # y0, y1 = np.where(cols)[0][[0, -1]]
        if contours is None:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
        x0, y0, w, h = cv2.boundingRect(biggest_contour)
        x1 = x0 + w
        y1 = y0 + h
    
    return x0, y0, x1, y1


def place_objects_in_area(obj_imgs: List, labels: List, area_size: Tuple[int, int], plot: bool = False) -> Tuple[List[Tuple[int, int, int, int]], List, List]:
    """
    Randomly places objects with corresponding bounding boxes in a fixed sized area
    :param obj_imgs: Images of objects to place in area
    :param labels: Label for each object
    :param area_size: Size (height, width) of the area
    :param plot: Whether to plot the positioned objects
    :return: Upper left corner (x, y) of the objects
    """
    
    bboxes, used_obj_imgs, used_labels = [], [], []
    for obj_img, label in zip(obj_imgs, labels):
        tmp = np.ones(area_size)
        tmp[-obj_img.size[1]:, :] = 0
        tmp[:, -obj_img.size[0]:] = 0
        
        for bbox in bboxes:
            tmp[max(0, bbox[1] - obj_img.size[1]): bbox[3], max(0, bbox[0] - obj_img.size[0]): bbox[2]] = 0
        
        # If enough space is available, place object
        free_idxs = tmp.nonzero()
        if free_idxs[0].shape[0] > 0:
            rnd_int = np.random.randint(len(free_idxs[0]) - 1)
            x0, y0 = free_idxs[1][rnd_int], free_idxs[0][rnd_int]
            x1, y1 = x0 + obj_img.size[0], y0 + obj_img.size[1]
            
            bboxes.append((x0, y0, x1, y1))
            used_obj_imgs.append(obj_img)
            used_labels.append(label)
    
    if plot:
        vis = Image.new('RGB', area_size)
        draw = ImageDraw.Draw(vis)
        for bbox in bboxes:
            draw.rectangle(bbox, fill=(255, 255, 255), width=2)
        
        vis.show()
    
    return bboxes, used_obj_imgs, used_labels


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


def generate_annotations(labelss: List[List[str]], save_dir: str, imgs: List = None, paths: List = None,
                         img_sizes: List[Tuple] = None, bboxess: List[List[Tuple[int, int, int, int]]] = None, segmentationss=None):
    """
    Saves images and annotations in COCO format
    :param bboxess: List of lists containing bounding boxes
    :param segmentationss: List of lists containing contours
    :param labelss: List of lists containing labels
    :param save_dir: Directory to save images to
    :param imgs: List of PIL images
    :param paths: List of paths to images
    :param img_sizes: Sizes of the PIL images
    """
    
    try:
        imgs = [None] * len(paths) if imgs is None else imgs
        paths = [None] * len(imgs) if paths is None else paths
    except Exception as e:
        print(f'Need to at least pass in real images or paths: {e}')
    
    img_sizes = [None] * len(paths if imgs is None else imgs) if img_sizes is None else img_sizes
    
    try:
        bboxess = [None] * len(segmentationss) if bboxess is None else bboxess
        segmentationss = [None] * len(bboxess) if segmentationss is None else segmentationss
    except Exception as e:
        print(f'Need to at least pass in bounding boxes or segmentations for each image: {e}')
    
    cats = [{'id': i, 'name': cat, 'supercategory': cat} for i, cat in enumerate(sorted(set([i for s in labelss for i in s])))]
    
    img_anns, obj_anns = [], []
    for i, (img, path, bboxes, segmentations, labels, img_size) in tqdm(enumerate(zip(imgs, paths, bboxess, segmentationss, labelss, img_sizes)),
                                                                        desc='Creating annotations', total=len(imgs)):
        if path is not None:
            copyfile(path, os.path.join(save_dir, f'generated_{i}.png'))
        else:
            img.save(os.path.join(save_dir, f'generated_{i}.png'))
        
        size = (Image.open(path).size if img_size is None else img_size) if img is None else img.size
        
        img_anns.append({"id": i, "width": size[0], "height": size[1], "file_name": f'generated_{i}.png', "license": 0, "flickr_url": '', "coco_url": '',
                         "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        
        segmentations = [None] * len(bboxes) if segmentations is None else segmentations
        bboxes = [None] * len(segmentations) if bboxes is None else bboxes
        
        for bbox, segmentation, label in zip(bboxes, segmentations, labels):
            if bbox is None:
                pass  # TODO get bbox from segmentation
            
            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            area = width * height
            
            obj_anns.append({"id": len(obj_anns), "image_id": i, "category_id": [cat['id'] for cat in cats if cat['name'] == label][0],
                             "segmentation": [] if segmentation is None else segmentation, "area": int(area),
                             "bbox": [int(x0), int(y0), int(width), int(height)], "iscrowd": 0})
    
    with open(os.path.join(save_dir, 'annotations.json'), 'w') as f:
        json.dump({'images': img_anns, 'annotations': obj_anns, 'categories': cats}, f)

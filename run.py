import argparse
import ast
import datetime
import json
import os
import random
from glob import glob
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs-dir', type=str, required=True,
                        help='Directory containing subdirectories with label names which each contain masked out images of the objects')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save images and annotations to')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256], help='Output image size. Specify height and width seperated by whitespace')
    parser.add_argument('--num-imgs', type=int, default=100, help='Number of unaugmented images to produce')
    parser.add_argument('--bgs-dir', type=str, default=None, help='Directory containing random background images')
    parser.add_argument('--augs', type=str, nargs='*', default=None,
                        help='`torchvision.transforms` augmentations to perform on the objects, in given order, seperated by whitespace. E.g. RandomHorizontalFlip(p=0.5) RandomVerticalFlip(p=0.5)')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for maximum alpha value to consider transparent')
    parser.add_argument('--max-num-objs', type=int, default=5, help='Maximum number of objects in a resulting image')
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


def crop_image(img: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Crops out single object in transparent image
    :param img: Image with 4 dimensions where the last dimension is the alpha channel
    :param threshold: Threshold for maximum alpha value to consider transparent
    :return: Cropped image
    """
    
    idxs = np.argwhere(img[:, :, -1] > threshold)
    y, x = idxs[:, 0], idxs[:, 1]
    cropped_img = img[np.min(y):np.max(y), np.min(x):np.max(x)]
    
    return cropped_img


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
    for i, (img, bboxes, labels) in tqdm(enumerate(zip(imgs, bboxess, labelss)), desc='Creating annotations', total=len(imgs)):
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
    - Do not specify image size but automatically allocate space for objects
    """
    # Set seeds for reproducibility
    np.random.seed(1)
    random.seed(1)
    
    args = parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.objs_dir), f'{os.path.basename(args.objs_dir)}_results')
    
    created_dir = False
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        created_dir = True
    
    try:
        paths = glob(os.path.join(args.objs_dir, '*', '*.png'))
        
        obj_imgs = [Image.open(path) for path in paths]
        obj_imgs = [Image.fromarray(crop_image(np.array(img), threshold=args.threshold)) for img in tqdm(obj_imgs, desc='Cropping images')]
        
        labels = [os.path.basename(os.path.dirname(path)) for path in paths]
        
        data = list(zip(obj_imgs, labels))
        
        bg_imgs = get_backround_images(args.num_imgs, args.img_size, args.bgs_dir)
        
        bboxess, labelss = [], []
        # blend background and foreground images
        for i, bg_img in tqdm(enumerate(bg_imgs), desc='Blending images', total=len(bg_imgs)):
            rnd_data = random.sample(data, k=min(len(obj_imgs), np.random.randint(1, args.max_num_objs)))
            rnd_obj_imgs, rnd_labels = [[d[i] for d in rnd_data] for i in range(len(rnd_data[0]))]
            rnd_obj_imgs = [augment_image(obj_img, args.augs) for obj_img in rnd_obj_imgs] if args.augs is not None else rnd_obj_imgs
            
            # reverse shape to have height, width form
            bboxes, used_obj_imgs, used_labels = place_objects_in_area(rnd_obj_imgs, labels, args.img_size)
            bboxess.append(bboxes)
            labelss.append(used_labels)
            
            for bbox, used_obj_img in zip(bboxes, used_obj_imgs):
                bg_img.paste(used_obj_img, (bbox[0], bbox[1]), used_obj_img)
            
            print()
        
        generate_annotations(bg_imgs, bboxess, labelss, args.save_dir)
    
    except Exception as e:
        print(e)
        if created_dir:
            os.remove(args.save_dir)


if __name__ == '__main__':
    main()

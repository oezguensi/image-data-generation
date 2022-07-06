import argparse
import os
import random
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_bounding_box, generate_annotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs-dir', type=str, required=True,
                        help='Directory containing subdirectories with label names which each contain masked out images of the objects')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save images and annotations to')
    parser.add_argument('--img-type', type=str, default='png', help='Type of images')
    args = parser.parse_args()
    
    return args


def main():
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
        img_paths = glob(os.path.join(args.objs_dir, '*', 'images', f'*.{args.img_type}'))
        mask_paths = [os.path.join(os.path.dirname(os.path.dirname(img_path)), 'masks', os.path.basename(img_path)) for img_path in img_paths]
        
        if len(mask_paths) == 0 or len(img_paths) == 0:
            raise ValueError('Could not find any images')
        
        masks = [Image.open(path) for path in mask_paths]
        img_sizes = [img.size for img in masks]
        bboxess = [[get_bounding_box(np.array(mask))] for mask in tqdm(masks, desc='Getting bounding boxes')]
        
        labelss = [[os.path.basename(os.path.dirname(os.path.dirname(path)))] for path in img_paths]
        
        generate_annotations(bboxess, labelss, args.save_dir, paths=img_paths, img_sizes=img_sizes)
    
    except Exception as e:
        print(e)
        if created_dir:
            os.remove(args.save_dir)


if __name__ == '__main__':
    main()

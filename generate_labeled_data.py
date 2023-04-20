import argparse
import os
import random
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_bounding_box, generate_annotations, get_segmentations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs-dir', type=str, required=True,
                        help='Directory containing subdirectories with label names which each contain masked out images of the objects')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save images and annotations to')
    parser.add_argument('--img-type', type=str, default='png', help='Type of images')
    parser.add_argument('--mask-type', type=str, default='png', help='Type of masks')
    args = parser.parse_args()
    
    return args


def main():
    # Set seeds for reproducibility
    np.random.seed(1)
    random.seed(1)
    
    args = parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.objs_dir), f'{os.path.basename(args.objs_dir)}_results')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
        img_paths = glob(os.path.join(args.objs_dir, '*', 'images', f'*.{args.img_type}'))
        mask_paths = [os.path.join(os.path.dirname(os.path.dirname(img_path)), 'masks', f"{os.path.basename(img_path).split('.')[0]}.{args.mask_type}") for
                      img_path in img_paths]
        
        if len(mask_paths) == 0 or len(img_paths) == 0:
            raise ValueError('Could not find any images')
        
        bboxess, segmentationss, img_sizes, labelss, success_paths = [], [], [], [], []
        for mask_path, img_path in tqdm(zip(mask_paths, img_paths), total=len(mask_paths)):
            try:
                mask = Image.open(mask_path).convert('L')
                
                segmentationss.append([get_segmentations(np.array(mask))])
                bboxess.append([get_bounding_box(np.array(mask))])
                img_sizes.append(mask.size)
                labelss.append([os.path.basename(os.path.dirname(os.path.dirname(img_path)))])
                success_paths.append(img_path)
            except Exception as e:
                print(f'Problem with image: {img_path}: {e}')
                
                num_success = min(len(segmentationss), len(bboxess), len(img_sizes), len(labelss))
                segmentationss, bboxess, img_sizes, labelss = segmentationss[:num_success], bboxess[:num_success], img_sizes[:num_success], labelss[
                                                                                                                                            :num_success]
        generate_annotations(labelss, args.save_dir, paths=success_paths, img_sizes=img_sizes, bboxess=bboxess, segmentationss=segmentationss)
        
        os.system(f'zip -r {os.path.join(args.save_dir, os.path.basename(args.objs_dir))}.zip {args.save_dir} -x ".*" -x "__MACOSX"')


if __name__ == '__main__':
    main()

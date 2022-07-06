import argparse
import os
import random
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_bounding_box, get_backround_images, augment_image, place_objects_in_area, generate_annotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs-dir', type=str, required=True,
                        help='Directory containing subdirectories with label names which each contain masked out images of the objects')
    parser.add_argument('--img-type', type=str, default='png', help='Type of images')
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
        img_paths = glob(os.path.join(args.objs_dir, '*', f'*.{args.img_type}'))
        
        imgs = [Image.open(path) for path in img_paths]
        bboxes = [get_bounding_box(np.array(img), threshold=args.threshold) for img in tqdm(imgs, desc='Getting bounding boxes')]
        imgs = [img.crop(bbox) for img, bbox in zip(imgs, bboxes)]
        
        labels = [os.path.basename(os.path.dirname(path)) for path in img_paths]
        
        data = list(zip(imgs, labels))
        
        bg_imgs = get_backround_images(args.num_imgs, args.img_size, args.bgs_dir)
        
        bboxess, labelss = [], []
        # blend background and foreground images
        for i, bg_img in tqdm(enumerate(bg_imgs), desc='Blending images', total=len(bg_imgs)):
            if args.max_num_objs == 1:
                rnd_data = [data[i % len(data)]]
            else:
                rnd_data = random.sample(data, k=min(len(imgs), np.random.randint(1, args.max_num_objs)))
            
            rnd_obj_imgs, rnd_labels = [[d[i] for d in rnd_data] for i in range(len(rnd_data[0]))]
            rnd_obj_imgs = [augment_image(obj_img, args.augs) for obj_img in rnd_obj_imgs] if args.augs is not None else rnd_obj_imgs
            
            # reverse shape to have height, width form
            bboxes, used_obj_imgs, used_labels = place_objects_in_area(rnd_obj_imgs, rnd_labels, args.img_size)
            bboxess.append(bboxes)
            labelss.append(used_labels)
            
            for bbox, used_obj_img in zip(bboxes, used_obj_imgs):
                bg_img.paste(used_obj_img, (bbox[0], bbox[1]), used_obj_img)
        
        generate_annotations(bboxess, labelss, args.save_dir, imgs=bg_imgs)
    
    except Exception as e:
        print(e)
        if created_dir:
            os.remove(args.save_dir)


if __name__ == '__main__':
    main()

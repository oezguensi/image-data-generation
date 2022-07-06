# Introduction
This project creates a labeled Object Detection dataset (COCO format).
You can either create a labeled synthetic dataset using segmented objects (`generate_synthetic_data.py`) or just create annotations using binary masks with the corresponding images (`generate_labeled_data.py`).

In the first case of creating a synthetic dataset the `generate_synthetic_data.py` script takes transparent images of segmented objects and puts them onto colored background or given background images.
To automatically segment out objects in images one can use e.g. pretrained Salient Object Detection models such as [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net). 

In the seoncd case of creating just the annotation file the `generate_labeled_data.py` script takes binary masks to specify bounding boxes as well as the original images.

# Installation
The project was tested using `python=3.7` and the packages listed under `requirements.txt`.
To install it, create a virtual environment and run `pip install -r requirements.txt`.

# Usage
For synthetic data generation (`generate_synthetic_data.py`) the folder structure must look as follows:
 ```
 .
 ├── ...
 ├── objects_dir
 │   ├── classA
 │   │   └── image1.png
 │   ├── classB
 │   │   └── image2.png
 │   └── ...
 ├── background_dir (Optional)
 │   ├── background1.jpg
 │   ├── background2.jpg
 │   └── ...
 ```
where the image files and directory names can be arbitrary and the background directory is optional.

For just creating a labeled dataset (`generate_labeled_data.py`) the folder structure must look as follows:
```
.
├── ...
├── objects_dir
│   ├── classA
│   │   └── images
│   │   │    └── image1.png
│   │   └── masks
│   │   │    └── image1.png
│   ├── classB
│   │   └── images
│   │   │    └── image2.png
│   │   └── masks
│   │   │    └── image2.png
│   └── ...
```
where the image filenames must equal the mask filenames.
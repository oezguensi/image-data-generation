# Introduction
This project creates a labeled Object Detection dataset (COCO format) using masked out objects.
It takes transparent images of segmented objects and puts them onto colored background or given background images.
To automatically segment out objects in images one can use e.g. pretrained Salient Object Detection models such as [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net). 

# Installation
The project was tested using `python=3.7` and the packages listed under `requirements.txt`.
To install it, create a virtual environment and run `pip install -r requirements.txt`.

# Usage
1. The objects which will be pasted onto background images have to be in the `png` format and transparent.
2. The folder structure must look as follows:
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
    └── ...
    ```
    where the image files and directory names can be arbitrary and the background directory is optional.
3. Execute `python run.py --help` to get the full documentation of the specific arguments that can be passed in.
4. Execute `python run.py` with the given flags to generate the data.
# Introduction
This project creates a labeled dataset (COCO format) of images using segmented objects

# Installation
The project was tested using `python=3.7` and the packages under `requirements.txt`.
To install it create a virtual environment and run `pip install -r requirements.txt`.

# Usage
1. The objects which will be pasted onto background images have to be in the `png` format and transparent.
2. The folder structure must look as follows:
    ```
    .
    ├── ...
    ├── objects_dir
    │   ├── class1
    │   │   └── image1_1.png
    │   ├── class2
    │   │   └── image2_1.png
    │   └── ...
    ├── background_dir
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
    ```
    where the image filenames can be arbitrary and the background directory is optional.
3. Execute `python run.py --help` to get the full documentation of the specific arguments that can be passed in.
4. Execute `python run.py` with the given flags to generate the data.
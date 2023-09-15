
# Face Detection and Clustering

This Python script performs face detection and clustering on a collection of images using OpenCV, NumPy, scikit-learn, and face_recognition libraries.

## Overview

This script contains two main functions:

1. **detect_faces(input_path: str) -> dict**: This function takes the path to a directory containing image files and performs face detection using a Haar cascade classifier. It returns a list of dictionaries, where each dictionary represents an image and the coordinates of the detected faces within that image.

2. **cluster_faces(input_path: str, K: int) -> dict**: This function takes the path to a directory containing image files and the desired number of clusters (K). It performs face detection, extracts facial features, and applies K-means clustering to group similar faces. It returns a dictionary that maps cluster numbers to the names of the images in each cluster.

![portfolio-7](https://github.com/mrunmayee17/Face-Detection-in-the-Wild/assets/48186569/f82c313b-bffd-4374-9710-ce31801335f3)


## How to Use

1. Make sure you have the required libraries installed, including OpenCV, NumPy, scikit-learn, and face_recognition.

2. Import the `detect_faces` and `cluster_faces` functions into your project.

3. Use the `detect_faces` function to detect faces in a directory of images and obtain their coordinates.

```python
from your_module import detect_faces

input_path = "path_to_image_directory"
results = detect_faces(input_path)

# Example output:
# [{'iname': 'image1.jpg', 'bbox': [x1, y1, width1, height1]}, {'iname': 'image2.jpg', 'bbox': [x2, y2, width2, height2]}, ...]

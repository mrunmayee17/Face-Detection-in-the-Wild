'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import glob
import re
from sklearn.cluster import KMeans
import face_recognition


'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
# Defining Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# finding all *.jpg images in faceCluster_5 and sorting them according to image file name numbers using glob and regex
    numbers = re.compile(r'(\d+)')

    img_dict = {int(numbers.split(k)[1]): k for k in glob.glob(f"{input_path}/*.jpg")}
    sorted_img_dict = dict(sorted(img_dict.items()))

# Reading Images in sorted order, Gray scaling them and using detect multiscale on Haar Cascade module to obtain x, y, width and height of detected faces
    for key, path in sorted_img_dict.items():
        img_name = path.split('/')
        tdc = {}
        read_image = cv2.imread(path)
        gray_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        # face_rec = face_cascade.detectMultiScale(gray_image,1.1,21)
        face_rec = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20), flags=0)
# converting face_rec values in to numpy array and storing it in a list
        face_rec_list = np.array(face_rec).tolist()
# if one and more than one face is detected then storing the path of image and (x,y,w,h) values in a dictionary
        if len(face_rec_list) >= 1:
            for i in range(len(face_rec_list)):
                tdc["iname"] = img_name[2]
                tdc["bbox"] = face_rec_list[i]
                result_list.append(tdc)
                tdc = {}
        # else:
        #     tdc["iname"] = img_name[2]
        #     tdc["bbox"] = face_rec_list
        #     result_list.append(tdc)

    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
# Defining Haar cascade classifier
    face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# finding all *.jpg images in faceCluster_5 and sorting them according to image file name numbers using glob and regex
    num1 = re.compile(r'(\d+)')
    img_dict1 = {int(num1.split(k)[3]): k for k in glob.glob(f"{input_path}/*.jpg")}
    sorted_img_dict1 = dict(sorted(img_dict1.items()))
    rsmap = []
    rs = []
    f_encodings = []
# Reading Images in sorted order, Gray scaling them and using detect multiscale on Haar Cascade module to obtain x, y, width and height of detected faces
    for key1, path1 in sorted_img_dict1.items():
        img_name1 = path1.split('/')
        tdc1 = {}
        read_image1 = cv2.imread(path1)
        gray_image1 = cv2.cvtColor(read_image1, cv2.COLOR_BGR2GRAY)

        face_rec = face_cascade1.detectMultiScale(
            gray_image1,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=0
        )
# converting them in to numpy array and storing it in a list
        face_rec_list1 = np.array(face_rec).tolist()

# if one and more than one face is detected then storing the path of image and (x,y,w,h) values in a dictionary
        if len(face_rec_list1) >= 1:
            for i in range(len(face_rec_list1)):
                tdc1["iname"] = img_name1[1]
                tdc1["bbox"] = face_rec_list1[i]
                rsmap.append(tdc1)
                tdc1 = {}
# creating seprate crop images of the face detected and applying face_recognition.face_encodings on the cropimages and there (x,y,w,h) values for obtaining features of the detected faces which is of an array of size 128
        for x, y, w, h in face_rec:
            cropimages = read_image1[y:y + h, x:x + w]
            f_encodings.append(face_recognition.face_encodings(cropimages, [(y, x + w, y + h, x)]))
# reducing the dimensions of f_encodings in order to fit training data for clustering
    sqfenc = np.array(f_encodings).squeeze()


# Applying K means cluster on face encodings obtained from crop images of detected faces
    kmean = KMeans(n_clusters=int(K), n_init = 1000, random_state=0)
    kmean.fit(sqfenc)

    labels = kmean.predict(sqfenc)
    labels = labels.tolist()

    # labels1 = kmean.labels_
    # Adding cluster value for each image in rsmap (dictionary)
    for idx, res in enumerate(rsmap):
        res['cluster'] = labels[idx]
    # creating dictionary of cluster number and its images in that cluster
    clstrs = {}
    for idxrs in rsmap:
        if idxrs['cluster'] not in clstrs:
            clstrs[idxrs['cluster']] = []
            clstrs[idxrs['cluster']].append(idxrs['iname'])
        elif idxrs['cluster'] in clstrs and idxrs['iname'] not in clstrs[idxrs['cluster']]:
            clstrs[idxrs['cluster']].append(idxrs['iname'])
# Sorting the dictionary in order according to cluster number and storing it in a variable
    sorted_clstrs = dict(sorted(clstrs.items()))

# formatting sorted cluster and storing it in a given format
    x = []
    y = []

    for i, m in sorted_clstrs.items():
        x.append(m)
        y.append(i)
    for i in range(len(x)):
        results1 = {}
        results1['cluster_no'] = y[i]
        results1['elements'] = x[i]
        result_list.append(results1)


    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""

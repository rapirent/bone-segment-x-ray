#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import json, codecs

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_ROOT, 'X_Ray_Data_set/Knee2Dto3D_120XRrays/Knee2Dto3D_120XRrays')

def detect_hand_and_fingers(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3),(1,1))
    img = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    a = img.copy()
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,91, -9)
    dilate_sz = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_sz, 2 * dilate_sz),(dilate_sz, dilate_sz))
    img = cv2.dilate(img, element,iterations = 1)
    return img

def detect_hand_silhoutte(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7),(3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,191, 9)
    erode_sz = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * erode_sz + 1, 2 * erode_sz + 1),(erode_sz, erode_sz))
    erosion = cv2.erode(img,kernel,iterations = 1)
    img = cv2.erode(img, element)
    dilate_sz = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * dilate_sz + 1, 2 * dilate_sz + 1),(dilate_sz, dilate_sz))
    img = cv2.dilate(img, element, iterations = 1)

    cv2.bitwise_not(img, img)
    return img

print('IN ' + DATA_PATH)

for dirpath, dirnames, filenames in os.walk(DATA_PATH):
    for index ,filename in enumerate(filenames):
        name, file_extension = os.path.splitext(os.path.join(dirpath,filename))
        if file_extension != '.jpg':
            continue
        print('LOAD ' + os.path.join(dirpath,filename))
        img = cv2.imread(os.path.join(dirpath,filename), 0)
        cv2.fastNlMeansDenoising(img, img,3, 7, 21)
        img2 = img.copy()

        height, width = img.shape[:2]
        start_row, start_col = int(0), int(0)
        end_row, end_col = int(height), int(width*0.5)
        left_img = img[start_row:end_row , start_col:end_col]
        left_img2 = left_img.copy()

        start_row, start_col = int(0), int(width*0.5)
        end_row, end_col = int(height), int(width)
        right_img = img[start_row:end_row , start_col:end_col]
        right_img2 = right_img.copy()

        gray_l = detect_hand_and_fingers(left_img)
        gray_silhouette_l = detect_hand_silhoutte(left_img2)
        gray_r = detect_hand_and_fingers(right_img)
        gray_silhouette_r = detect_hand_silhoutte(right_img2)

        bones_l = gray_l - gray_silhouette_l
        bones_r = gray_r - gray_silhouette_r


        _ , contours_l, hierarchy = cv2.findContours(bones_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _ , contours_r, hierarchy = cv2.findContours(bones_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_r = []
        contour_l = []
        contour_r = [_ for sublist in [a.tolist() for a in contours_r] for item in sublist for _ in item]
        contour_l = [_ for sublist in [a.tolist() for a in contours_l] for item in sublist for _ in item]

        json_file = str(index) + '_' + filename + '_l.json'
        print(json_file)
        json.dump(contour_l, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)
        json_file = str(index) + '_' + filename + '_r.json'
        json.dump(contour_r, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)

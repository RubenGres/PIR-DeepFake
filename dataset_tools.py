import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras as k
import keras.layers as l
import random

def get_pics(path, lim = 1000): 
    pics = []
    for i in range(1,len(os.listdir(path))):
        gray = cv.cvtColor(cv.imread(path + str(i) + ".bmp"), cv.COLOR_BGR2GRAY)
        pics.append(gray)
        
        if i == lim:break
    return pics

def resize_pics(pics, w, h):
    resize = []
    for pic in pics:
        resize.append(cv.resize(pic, (w, h)))
    return np.array(resize)

def mat2flat(pics, w, h):
    return pics.reshape((len(pics), w*h))

def flat2mat(pics, w, h):
    return pics.reshape((len(pics), w, h))

def format_images(pics_a, pics_b, w, h) :
        Xa = resize_pics(pics_a, w, h) / 255
        Xb = resize_pics(pics_b, w, h) / 255

        Xa = mat2flat(Xa, w, h)
        Xb = mat2flat(Xb, w, h)
        
        return Xa, Xb
    
def separate_dataset(Xa, ratio = 0.9):
    return Xa[0:int(len(Xa)*ratio)], Xa[int(len(Xa)*ratio):]
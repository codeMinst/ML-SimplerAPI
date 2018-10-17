import  base64
from PIL import Image
import io
import cv2
import numpy
import os

def bas64ToRGB(message):
    img= base64.b64decode(message.split(',')[1])
    img = Image.open(io.BytesIO(img)).convert('RGB')

    #return  cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2RGB)
    return img

def makeIndexToLabelFromDir(dirPath):
    idx2label = {}
    i = 0
    for dir in os.listdir(dirPath):
        idx2label[i] = dir
        i = i + 1
    #label2index = dict((v, k) for k, v in label2index.items())

    return idx2label

def makeLabelToIndexFromDir(dirPath):
    idx2label = {}
    i = 0
    for dir in os.listdir(dirPath):
        idx2label[dir] = i
        i = i + 1

    return idx2label


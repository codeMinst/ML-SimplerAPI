import  base64
from PIL import Image
import dlib
import numpy
from PIL import Image
from keras.utils import np_utils
import io
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

class withDlib:
    def __init__(self, predictor_path = 'shape_predictor_68_face_landmarks.dat',
                 face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'):
        self.shapePredictor = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.detector = dlib.get_frontal_face_detector()

    #디텍션 값 없이 이미지만 받아 디텍션과 피쳐추출을 한꺼번에 진행하여 디텍션된 만큼의 피쳐를 추출해도 되지만
    #디텍션 값의 크기로 피쳐 추출을 진행 할지의 여부에 따라 핑요없는 피쳐추출이 있을 수 있으므로 디텍션값을 받아서 처리
    def extractFeatureFromImg(self, dets, img):
        dlibFeatures = []
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = self.shapePredictor(img, d)

            dlibFeature = self.facerec.compute_face_descriptor(img, shape)
            dlibFeatures.append(numpy.array(dlibFeature))

        return dlibFeatures

    def generateXYFromDir(self, path):
        win = dlib.image_window()
        xFeatures = []
        yLabel = []
        label2index = makeLabelToIndexFromDir(path)
        # Now process all the images
        for root, dirs, files in os.walk(path):
            for file in files:
                dirName = root.split('\\')[1]
                filePath = path + '/' + dirName + '/' + file
                print(filePath)

                print("Processing file: {}".format(filePath))
                img = Image.open(filePath).convert('RGB')
                img = numpy.array(img)
                win.set_image(img)

                dets = self.detector(img, 1)
                print("Number of faces detected: {}".format(len(dets)))
                if len(dets) == 1:
                    xFeature = self.extractFeatureFromImg(dets=dets, img=img)
                    xFeatures.append(xFeature[0])
                    yLabel.append(label2index[root.split('\\')[1]])
        #onehot
        yLabel = np_utils.to_categorical(yLabel)

        return xFeatures, yLabel

'''
win = dlib.image_window()

self.win.clear_overlay()
self.win.set_image(img)

self.win.clear_overlay()
self.win.add_overlay(d)
win.add_overlay(shape)
'''
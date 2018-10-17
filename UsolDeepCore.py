# -*- coding: utf-8 -*-

from usol import usolUtil
import numpy as np
import dlib
import pickle
from keras_vggface.vggface import VGGFace
from keras.models import load_model
from PIL import Image
import json
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping

### Work around - CUDA_ERROR_OUT_OF_MEMORY
import keras.backend as K
from pathlib import Path

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
########################################

if __name__ == "__main__":
    print("core 직접 실행")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("core import")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_DETECT = 4
PROB_WORK_BENCHMARK = 0.93  # 출퇴근 해당값보다 작을 때 unknown 처리
L2_WORK_BENCHMARK = 3.5  # 출퇴근 해당 값보다 클 때 unknown 처리
PROB_LOG_BENCHMARK = 0.7  # 로깅 해당값보다 작을 때 unknown 처리
L2_LOG_BENCHMARK = 4.0  # 로깅 해당 값보다 클 때 unknown 처리
UPLOAD_DIR = './upload_img/'  # 프론트에서 업로드한 원본 이미지 저장
PEOPLE_DIR = './people/'  # 224 사이즈로 리사이즈한 이미지 저장
DATA_XY_FILE = 'dataXY.npz'
MODEL_NAME = 'hs_model.h5'  # 모델명
MODEL_LABEL = 'hs_model_label.pkl'  # 레이블 리스트

global graph
graph = tf.get_default_graph()

detector = dlib.get_frontal_face_detector()
resnet_vgg = VGGFace(model='resnet50', include_top=False,
                     input_shape=(224, 224, 3), pooling='max', weights='vggface')

# unknown 판별 모델
with open('hs_svm_work_unknown.pkl', 'rb') as f:
    work_svm = pickle.load(f)
with open('hs_svm_log_unknown.pkl', 'rb') as f:
    log_svm = pickle.load(f)

myModel = None
LabelDic = None
svm = None


###############################################################################
# Private functions
###############################################################################
# Action:
#   모델, 레이블이 업데이트 되었는지 체크 후 모델, 레이블 로드
def loadModel():
    _model = None
    _labeldic = None

    if os.path.isfile(MODEL_NAME):
        _model = load_model(MODEL_NAME)
        with open(MODEL_LABEL, 'rb') as f:
            _labeldic = pickle.load(f)

    return _model, _labeldic


# Actions:
#   boundingBox 잘라내기 및 이미지 리사이즈
# Params:
#   img - Image 객첵
#   d - dector 객체
#   x,y - 리사이즈할 사이즈
def imgCropResize(image, d, x, y):
    img = image.crop((d.left(), d.top(), d.right(), d.bottom()))
    img = img.resize((x, y), Image.ANTIALIAS)
    return img


# Actions:
#   resnet_vgg를 이용하여 피처 추출 후 피처값 Nomalize
# Params:
#   img - Image 객첵
# Return value:
#   Normalization된 피처값
def getFeatureByResnetVgg(img):
    npImg = np.array(img)
    npImg = npImg.reshape(1, 224, 224, 3)  # 추후, 이미지로부터 shape를 뽑아내서 사용

    with graph.as_default():
        feature = resnet_vgg.predict(npImg)
        # normalization. because training feature coverted normalization
        feature = (feature - feature.min()) / feature.max() - feature.min()
    return feature


# Actions:
#   리사이즈된 이미지 저장 디렉토리(name) 생성 및 이미지 리사이즈 & 저장
def imgResize(name):
    if os.path.exists(PEOPLE_DIR + name):
        print('Error: ' + name + ' dir alreay exists')
        return 0
    else:
        os.makedirs(PEOPLE_DIR + name)
        img_list = os.listdir(UPLOAD_DIR + name)

        for i, img_one in enumerate(img_list):
            try:
                org_img = Image.open(UPLOAD_DIR + name + '/' + img_one)

                dets = detector(np.array(org_img), 1)
                if len(dets) != 1:
                    print("Only 1 Face is accepted")
                    continue
                else:
                    img = imgCropResize(org_img, dets[0], 224, 224)
                    img.save(PEOPLE_DIR + name + '/' + img_one, 'JPEG')
            except:
                print('Some Error Occured... ')
        return (i + 1)

    # Actions:


#   name 디렉토리 이미지 리사이즈 후 피처 추출
def getXY(name):
    img_list = os.listdir(UPLOAD_DIR + name)
    n = len(img_list)

    i = 0
    X = []
    Y = []
    for img_one in img_list:
        try:
            org_img = Image.open(UPLOAD_DIR + name + '/' + img_one)

            dets = detector(np.array(org_img), 1)
            if len(dets) == 1:
                img = imgCropResize(org_img, dets[0], 224, 224)
                feature = getFeatureByResnetVgg(img)
                X.append(feature)
                Y.append(name)
                i += 1
            else:
                print('BB Detector length Err: ', len(dets))
        except Exception as ex:
            print('Some Error Occured...: ', ex)

    return i, X, Y


# Actions:
#   기존 기존 피처 X,Y에 새로운 피처 X,Y 업데이트
def updateXY(X, Y):
    if os.path.isfile(DATA_XY_FILE):
        dataXY = np.load(DATA_XY_FILE)
        xx, yy = dataXY['x'], dataXY['y']
        X = np.vstack((X, xx))
        Y.extend(yy)
        print(type(X), len(X), X.shape)
        print(type(Y), len(Y), Y)
    return X, Y


# Action:
#    문자열의 레이블을 onehot encoding된 레이블로 변환
# Return value: Number of Classes
def getEncodedYY(y):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    label_encoder.fit(y)
    y_ = label_encoder.transform(y)  # 숫자화
    y_ = y_.reshape(len(y_), 1)
    yy = onehot_encoder.fit_transform(y_)

    with open(MODEL_LABEL, 'wb') as f:
        pickle.dump(label_encoder.classes_, f)
    print('Model Classes....: ', label_encoder.classes_)

    return len(label_encoder.classes_), label_encoder.classes_, yy


# Action:
#    x,y 데이타 셔플
def dataShuffle(x, y):
    xy = list(zip(x, y))
    random.shuffle(xy)
    xx, yy = zip(*xy)
    xx = np.array(xx)
    yy = np.array(yy)
    return xx, yy


# Action:
#    모델을 만들고 트레이닝 및 파일로 저장
def getModel(numClasses, train_features, train_labels, validation_features, validation_labels):
    with graph.as_default():
        print('getModel()..........numClasses: {}'.format(numClasses))
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_dim=1 * 1 * 2048))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(numClasses, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      # optimizer=optimizers.RMSprop(lr=2e-4),
                      optimizer=optimizers.sgd(),
                      metrics=['acc'])

        early_stopping = EarlyStopping(patience=15, mode='auto', monitor='val_loss')
        history = model.fit(train_features,
                            train_labels,
                            epochs=500,
                            batch_size=200,
                            validation_data=(validation_features, validation_labels),
                            callbacks=[early_stopping])

        model.save(MODEL_NAME)
        return model


##############################################################################

# 1. mode 0: 출근 1: 입실
def facePridict(mode, strImg):
    recImg = usolUtil.bas64ToRGB(strImg)
    dets = detector(np.array(recImg), 1)
    print("Number of faces detected: {}".format(len(dets)))

    result = {}
    if len(dets) < 1:
        result = {'result': '0', 'msg': "can't find a face"}
    elif len(dets) > MAX_DETECT:
        result = {'result': '0', 'msg': "found too many faces"}
    else:
        features = []
        people = []
        if os.path.isfile(DATA_XY_FILE):
            dataXY = np.load(DATA_XY_FILE)
            xx, yy = dataXY['x'], dataXY['y']

        # with graph.as_default():
        for k, d in enumerate(dets):
            img = imgCropResize(recImg, d, 224, 224)
            feature = getFeatureByResnetVgg(img)
            features.append(feature)
        print("-----------------dnn predict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")
        for f in features:
            with graph.as_default():
                predictions = myModel.predict_classes(f.reshape(1, -1))
                prob = myModel.predict(f.reshape(1, -1))
                name, score = LabelDic[predictions[0]], str(np.amax(prob[0]))

                if (mode == "0"):
                    svm = work_svm
                    result = svm.predict(float(score))
                else:
                    svm = log_svm
                    # idx = np.where(np.argmax(yy, axis=1) == predictions[0])[0]
                    idx = np.where(yy == name)[0]
                    l2dists = []
                    for k in idx:
                        dist = np.linalg.norm(f.reshape(-1) - xx[k], axis=None, ord=None)
                        l2dists.append(dist)
                    result = svm.predict([float(score), np.min(l2dists)])

                print(name + '////prob--' + score + '////--' + str(result))
                if 0 == int(result):
                    name, score = 'unknown', '0'
                people.append({'name': name, 'prob': score})

        result = {'result': '1', 'people': people}

    jsonString = json.dumps(result)

    return jsonString


# test function
#    img_path : '../visitor/pre_over_100/people/test/ParkMinjeong/51.jpg'
def test_facePridict(img_path):
    PROB_BENCHMARK = PROB_WORK_BENCHMARK
    L2_BENCHMARK = L2_WORK_BENCHMARK
    img = Image.open(img_path)
    dets = detector(np.array(img), 1)
    print("Number of faces detected: {}".format(len(dets)))

    result = {}
    if len(dets) < 1:
        result = {'result': '0', 'msg': "can't find a face"}
    elif len(dets) > MAX_DETECT:
        result = {'result': '0', 'msg': "found too many faces"}
    else:
        features = []
        people = []

        for k, d in enumerate(dets):
            img = imgCropResize(img, d, 224, 224)
            feature = getFeatureByResnetVgg(img)
            features.append(feature)
        print("-----------------dnn predict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")
        for f in features:
            predictions = myModel.predict_classes(f.reshape(1, -1))

            with graph.as_default():
                prob = myModel.predict(f.reshape(1, -1))
                print("Predict result: ", LabelDic[predictions[0]], np.max(prob[0]), ", PROB_BENCHMARK(",
                      PROB_BENCHMARK, ")")
                if (np.amax(prob[0]) > PROB_BENCHMARK):
                    dnnPridict = LabelDic[predictions[0]]
                    people.append({'name': dnnPridict, 'prob': str(np.amax(prob[0]))})
                else:
                    people.append({'name': 'unknown', 'prob': 0})
        result = {'result': '1', 'people': people}

        jsonString = json.dumps(result)
        print("JSON Result: ", jsonString)

    return


# https://<<domain>>:<<port>>/api/request_training
# Params:
#   name - 모델에 추가할 클래스(사람) 이름
# Return values:
#   1 - 디렉토리 생성 성공
#   0 - 디렉토리 생성 실패
# Actions:
#   name 디렉토리 생성
def faceRegister(name):
    result = {}
    if not os.path.exists(UPLOAD_DIR + name):
        os.makedirs(UPLOAD_DIR + name)
        result = {'result': '1', 'msg': "OK"}
    else:
        result = {'result': '0', 'msg': "Name(" + name + ") already exists"}

    jsonString = json.dumps(result)
    return jsonString


# https://<<domain>>:<<port>>/api/register_face
# Params:
#   name - 모델에 추가할 클래스(사람) 이름
# Return values:
#   1 - 트레이닝 성공
#   0 - 트레이닝 실패
# Actions:
#   name 디렉토리 이미지 리사이징
#   name 클래스의 피처 추출 및 저장
#   model 만들기 및 테스트
def faceTraining(name):
    num, X, Y = getXY(name)

    X, Y = updateXY(X, Y)
    print(type(X), type(Y))
    np.savez(DATA_XY_FILE, x=X, y=Y)
    print('Total dataset size. X: ', len(X), ', Y: ', len(Y))

    X, Y = dataShuffle(X, Y)
    numClasses, dicClassess, enY = getEncodedYY(Y)

    if numClasses == 1:
        # 클래스가 1개인 경우 무조건 ok
        print('Number of Clases is 1!')
        result = {'result': '1', 'msg': "OK"}
        jsonString = json.dumps(result)
        return jsonString

    X = X.reshape(len(X), 2048)
    totalNum = len(X)
    print('After shuffle....')
    print(len(X), len(Y), len(enY), type(X), type(Y), type(enY), X.shape, Y.shape, enY.shape)

    # 데이타셋 나누기 train:validation:test = 80:15:5
    num_train = int(totalNum * 0.8)
    num_validation = int(totalNum * 0.15) + num_train
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        X[:num_train], enY[:num_train], X[num_train:num_validation], enY[num_train:num_validation], X[
                                                                                                    num_validation:], Y[
                                                                                                                      num_validation:]
    print('Total data is {}, train({}), validation({}), test({})'.format(totalNum, len(train_features),
                                                                         len(validation_features), len(test_features)))

    new_model = getModel(numClasses, train_features, train_labels, validation_features, validation_labels)

    # test
    with graph.as_default():
        predictions = new_model.predict_classes(test_features)
        prob = new_model.predict(test_features)
        print(predictions)
        print(prob)

    rightPred = 0
    for i, pred in enumerate(predictions):
        print('Real value: ', test_labels[i], '..... Predict value: ', dicClassess[pred], np.max(prob[i]))
        if test_labels[i] == dicClassess[pred]:
            rightPred += 1
    if rightPred == len(predictions):
        print('Model is good!!!!!')

    # 새 모델로 변수 업데이트
    global myModel, LabelDic
    myModel = new_model
    LabelDic = dicClassess

    result = {'result': '1', 'msg': "OK"}
    jsonString = json.dumps(result)

    # Path('./UsolDeepCore.py').touch() ## Work around - for updating global variables

    return jsonString


print('Start............................')
# 모델과 label dictionary가 존재하면 로딩....
myModel, LabelDic = loadModel()
if myModel is None:
    print("Model doesn't eixst....Do train first!!!!")
print('End............................')

########################## TEST ################################
# faceTraining('ParkSunhee')

# print('\n\ntest_facePridict()............\n')
# test_facePridict('../visitor/pre_over_100/people/test/ParkMinjeong/98.jpg')
########################## TEST ################################

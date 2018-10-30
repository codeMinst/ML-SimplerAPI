import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
import pickle
from keras_vggface.vggface import VGGFace

'''
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
vgg_conv.summary()
'''
resnet_v2 = VGGFace(model='resnet50', include_top=False,
                    input_shape=(224, 224, 3), pooling='max', weights='vggface')
resnet_v2.summary()
'''
resnet_v2 = InceptionResNetV2(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
'''
#os.path.abspath(a)
data_dir =  '../dataset/pre_over_100/people/train'
test_dir = '../dataset/pre_over_100/people/test'
labelCount = 0
day = '_resnet50vgg_max_0511'

#datagen = ImageDataGenerator(rescale=1. / 255)
datagen = ImageDataGenerator()
batch_size = 100

def makeFeature(generator, n):
    _features = np.zeros(shape=(n, 2048))
    _labels = np.zeros(shape=(n, labelCount))

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = resnet_v2.predict(inputs_batch)
        features_batch = (features_batch - features_batch.min()) / features_batch.max() - features_batch.min()
        _features[i * batch_size: (i + 1) * batch_size] = features_batch
        _labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        print('extract feature -------------------> ' + str((i + 1) * batch_size))
        i += 1
        if i * batch_size >= n:
            break

    # _features = np.reshape(_features, (n, 1*1*2048))
    return _features, _labels


data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

labelCount = data_generator.class_indices.__len__()
if os.path.isfile("dataXY" + day + ".npz"):
    print("-----------------load feature!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")
    data, test = np.load('dataXY' + day +'.npz'), np.load('testXY' + day+ '.npz')
    train_features, train_labels = data['x'][:3500], data['y'][:3500]
    validation_features, validation_labels = data['x'][3500:], data['y'][3500:]
    testFeatures, testLabel = test['x'], test['y']
else:
    print("-----------------save feature!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")
    features, labels = makeFeature(data_generator, data_generator.classes.size)
    train_features, train_labels, validation_features, validation_labels = features[:3500], labels[:3500], features[3500:], labels[3500:]
    np.savez('dataXY' + day, x=features, y=labels)
    testFeatures, testLabels = makeFeature(test_generator, test_generator.classes.size)
    np.savez('testXY' + day, x=testFeatures, y=testLabels)

'''
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_features, test_labels = makeFeature(test_generator, test_generator.classes.size)
'''

from sklearn.neighbors import KNeighborsClassifier

knn =None
if not os.path.isfile('knn_model' + day + '.pkl'):
    print("-----------------knn fit & start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")
    knn = KNeighborsClassifier(n_neighbors=labelCount)
    knn.fit(train_features, train_labels)
    print("-----------------knn fit & save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------")

    model_score = knn.score(train_features, train_labels)
    print(model_score)
    with open('knn_model' + day + '.pkl', 'wb') as f:
        pickle.dump(knn, f)

from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping

if not os.path.isfile('my_model' + day + '.h5'):
    #라벨 해쉬값 뒤바꾸기
    label2index = test_generator.class_indices
    idx2label = dict((v,k) for k,v in label2index.items())
    with open('idx2label' + day + '.pkl', 'wb') as f:
        pickle.dump(idx2label, f)

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_dim=1*1*2048))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(labelCount, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  #optimizer=optimizers.RMSprop(lr=2e-4),
                  optimizer=optimizers.sgd(),
                  metrics=['acc'])

    early_stopping = EarlyStopping(patience=15, mode='auto', monitor='val_loss')
    history = model.fit(train_features,
                        train_labels,
                        epochs=500,
                        batch_size=200,
                        validation_data=(validation_features,validation_labels),
                        callbacks=[early_stopping])

    model.save('hs_model' + day + '.h5')
    myModel = load_model('hs_model' + day + '.h5')


    fnames = test_generator.filenames
    ground_truth = test_generator.classes

    predictions = myModel.predict_classes(testFeatures)
    prob = myModel.predict(testFeatures)

    errors = np.where(predictions != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),test_generator.classes.size))

    i=0
    for i in range(len(errors)):
        pred_class = np.argmax(prob[errors[i]])
        pred_label = idx2label[pred_class]

        print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            prob[errors[i]][pred_class]))

        original = load_img('{}/{}'.format(test_dir, fnames[errors[i]]))
        plt.imshow(original)
        plt.show()


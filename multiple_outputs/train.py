# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from multiple_outputs.multicnn_net import MultiCnnNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# 실행인자 setting
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--categorybin", required=True, help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True, help="path to output color label binarizer")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 200
IMAGE_DIMS = (96, 96, 3)

# grab the data paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels
data = []
categoryLabels = []
colorLabels = []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    data.append(image)

    # dataset의 이미지 디렉토리명이 color_category로 구성되어 있음을 전제로 각 라벨 생성
    (color, cat) = imagePath.split(os.path.sep)[-2].split("_")
    categoryLabels.append(cat)
    colorLabels.append(color)

# scale the raw pixel intensities to the range [0, 1] and convert to
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

# binarize both sets of labels(one hot encoding)
# sklearn 전처리(https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/)
print("[INFO] binarizing labels...")
categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, categoryLabels, colorLabels, test_size=0.2, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = split
print("[INFO] complete data preprocessing...")

# initialize our MultiCnnNet multi-output network
model = MultiCnnNet.build(IMAGE_DIMS[1],
                           IMAGE_DIMS[0],
                           numCategories=len(categoryLB.classes_),
                           numColors=len(colorLB.classes_),
                           finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that specifies the weight per loss
# to balance the contribution of the different losses
losses = {
    "category_output": "categorical_crossentropy",
    "color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(trainX,
    {"category_output": trainCategoryY, "color_output": trainColorY},
    validation_data=(testX,
    {"category_output": testCategoryY, "color_output": testColorY}),
    epochs=EPOCHS,
    batch_size=BS,
    verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open(args["categorybin"], "wb")
f.write(pickle.dumps(categoryLB))
f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()

# plot the total loss, category loss, and color loss
lossNames = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
ax[i].legend()

# save the losses figure
plt.tight_layout()
plt.savefig("output_losses.png")
plt.close()

# create a new figure for the accuracies
accuracyNames = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("output_accs.png")
plt.close()
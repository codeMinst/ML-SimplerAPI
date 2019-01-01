# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


class MultiCnnNet:
    @staticmethod
    def build_category_branch(inputs, numCategories, finalAct="softmax", chanDim=-1):
        """이미지 분류를 위한 sub-network 생성

            # Arguments
                inputs:input받을 tensor ex)inputs = Input(shape=(height, width, 3))
                numCategories:종류숫자
                finalAct:최종 활성화 함수
                chanDim:추출된 conv들을 BatchNormalization 할 때의 기준축(axis) default=-1
                -1은 각 conv들 기준으로 함을 의미한다
                (axis개념 http://taewan.kim/post/numpy_sum_axis/)

            # Returns
               the category prediction sub-network
        """

        # keras lambda는 keras에서 제공하지 않는 커스텀 레이어를 추가 할 수 있게 해주는 기능
        # 특징점 추출에는 rgb가 아닌 1ch grayscale로도 가능하여 grayscale로 변환하는 레이어 추가
        # (다차원 독립 변수 벡터가 있을 때 각 벡터 원소들의 상대적 크기만 중요한 경우에 사용된다)
        x = Lambda(lambda c:tf.image.rgb_to_grayscale(c))(inputs)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        # 배치별 평균과 분산이 일정하도록 제한. 따라서 각 레이어가 앞쪽 층과의 관계를 약화시키고,
        # 각 레이어가 앞레이어의 영향을 많이 받지않고 학습된다. vanishing gradient, exploding gradient 해결
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # fully network
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numCategories)(x)
        x = Activation(finalAct, name="category_output")(x)

        return x

    @staticmethod
    def build_color_branch(inputs, numColors, finalAct="softmax", chanDim=-1):
        """색 분류를 위한 sub-network 생성

            # Arguments
                inputs:input받을 tensor ex)inputs = Input(shape=(height, width, 3))
                numColors:색상숫자
                finalAct:최종 활성화 함수
                chanDim:추출된 conv들을 BatchNormalization 할 때의 기준축(axis) default=-1
                -1은 각 conv들 기준으로 함을 의미한다
                (axis개념 http://taewan.kim/post/numpy_sum_axis/)

            # Returns
               the color prediction sub-network
        """

        # CONV => RELU => POOL
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # fully network
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numColors)(x)
        x = Activation(finalAct, name="color_output")(x)

        return x

    @staticmethod
    def build(width, height, numCategories, numColors, finalAct="softmax"):
        """input shape 초기화 설정 및 construct both the "category" and "color" sub-networks

            # Arguments
                width:인풋 이미지 횡길이
                height:인풋 이미지 종길이
                numCategories:물체종류 숫자
                numColors:색상숫자
                finalAct:최종 활성화 함수

            # Returns
               the constructed network architecture
        """

        #input shape 초기화 설정
        inputShape = (height, width, 3)
        chanDim = -1

        # construct both the "category" and "color" sub-networks
        inputs = Input(shape=inputShape)
        categoryBranch = MultiCnnNet.build_category_branch(inputs, numCategories, finalAct=finalAct, chanDim=chanDim)
        colorBranch = MultiCnnNet.build_color_branch(inputs, numColors, finalAct=finalAct, chanDim=chanDim)

        # 하나의 인풋값으로 branch 두개의 각각 출력
        model = Model(
            inputs=inputs,
            outputs=[categoryBranch, colorBranch],
            name="fashionnet")

        return model
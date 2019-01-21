from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, LambdaCallback
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')

x = model.output
x = Dense(1024, name='fully', init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(3, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)#rgb값 reduce
train_generator = train_datagen.flow_from_directory(
        './dataset/furuit/train',
        target_size=(224, 224),
        batch_size=30,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        './dataset/furuit/validation',
        target_size=(224, 224),
        batch_size=30,
        class_mode='categorical')

model.compile(loss='categorical_crossentropy',
                  #optimizer=optimizers.RMSprop(lr=2e-4),
                  optimizer=optimizers.adam(),
                  metrics=['acc'])

print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))
early_stopping = EarlyStopping(patience=15, mode='auto', monitor='val_loss')
history = model.fit_generator(train_generator,
                              steps_per_epoch=25,
                              epochs=100,
                              validation_data=val_generator,
                              validation_steps=5,
                              callbacks=[early_stopping, print_weights])

#모델 평가
print("-- Evaluate --")
scores = model.evaluate_generator(val_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


"""
# Store the fully connected layers
fc1 = model.layers[-3]
fc2 = model.layers[-2]
predictions = model.layers[-1]

# Create the dropout layers
dropout1 = Dropout(0.85)
dropout2 = Dropout(0.85)

# Reconnect the layers
x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
predictors = predictions(x)

# Create a new model
model2 = Model(inputs=model.input, outputs=predictors)
model2.summary()
"""
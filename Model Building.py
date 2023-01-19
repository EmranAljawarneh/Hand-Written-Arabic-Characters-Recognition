# import model layers.
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.utils import np_utils

model = Sequential()
# First feature extraction layer.
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1),padding='same'))
print("Convolution layer shape: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer shape: ",model.output_shape)
model.add(Dropout(0.25))
print("Dropout layer shape: ",model.output_shape)

# Second feature extraction layer.
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1),padding='same'))
print("Convolution layer shape: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer shape: ",model.output_shape)
model.add(Dropout(0.25))
print("Dropout layer shape: ",model.output_shape)

# Third feature extraction layer.
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1),padding='same'))
print("Convolution layer shape: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer shape: ",model.output_shape)
model.add(Dropout(0.25))
print("Dropout layer shape: ",model.output_shape)

# Fourth feature extraction layer.
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1),padding='same'))
print("Convolution layer shape: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer shape: ",model.output_shape)
model.add(Dropout(0.25))
print("Dropout layer shape: ",model.output_shape)

# Fifth feature extraction layer.
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1),padding='same'))
print("Convolution layer shape: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer shape: ",model.output_shape)
model.add(Dropout(0.25))
print("Dropout layer shape: ",model.output_shape)

model.add(Flatten())
print(model.output_shape)
model.add(Dense(128, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
print(model.output_shape)
model.add(Dense(28, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model.output_shape)

batch_size = 32
#batch_size = 16
#batch_size = 8
epoch = 20

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_image_reshape, train_label_encoded, train_size=0.7)

train_score = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)

test_score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical # convert to one-hot-encoding

# read train 
train_image = pd.read_csv("csvTrainImages 13440x1024.csv", header=None)
print(train_image.shape)

# read test 
test_image = pd.read_csv("csvTestImages 3360x1024.csv", header=None)
print(test_image.shape)

label_train_image = pd.read_csv('csvTrainLabel 13440x1.csv', header=None)
print(label_train_image.shape)

label_test_image = pd.read_csv('csvTestLabel 3360x1.csv', header=None)
print(label_test_image.shape)

# Normalize the data
train_image = train_image.values.astype('float32') / 255.0
label_train_image = label_train_image.values.astype('int32')
test_image = test_image.values.astype('float32') / 255.0
label_test_image = label_test_image.values.astype('int32')
print("train shape: ", train_image.shape)
print("test shape: ", test_image.shape)

# Label Encoding 

train_label_encoded = to_categorical(label_train_image -1, num_classes=28)
test_label_encoded = to_categorical(label_test_image -1, num_classes=28)
print(train_label_encoded.shape)
print(test_label_encoded.shape)

# reshape input letter images to 32x32x1
train_image_reshape = train_image.reshape([-1,32,32,1])
test_image_reshape = test_image.reshape([-1,32,32,1])
print(train_image_reshape.shape, train_label_encoded.shape, test_image_reshape.shape, test_label_encoded.shape)
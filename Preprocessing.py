# Normalize the data
train_image = train_image.values.astype('float32') / 255.0
label_train_image = label_train_image.values.astype('int32')
test_image = test_image.values.astype('float32') / 255.0
label_test_image = label_test_image.values.astype('int32')
print("train shape: ", train_image.shape)
print("test shape: ", test_image.shape)

# Label Encoding 
from keras.utils import to_categorical # convert to one-hot-encoding
train_label_encoded = to_categorical(label_train_image -1, num_classes=28)
test_label_encoded = to_categorical(label_test_image -1, num_classes=28)
print(train_label_encoded.shape)
print(test_label_encoded.shape)

# reshape input letter images to 32x32x1
train_image_reshape = train_image.reshape([-1,32,32,1])
test_image_reshape = test_image.reshape([-1,32,32,1])
print(train_image_reshape.shape, train_label_encoded.shape, test_image_reshape.shape, test_label_encoded.shape)
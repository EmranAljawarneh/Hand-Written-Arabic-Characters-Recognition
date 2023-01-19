# import packages
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt

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
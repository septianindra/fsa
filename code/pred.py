from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from cv2 import imread, createCLAHE 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = load_model('./cxr_reg_weights.best.hdf5')

print(model.summary())

im = cv2.resize(cv2.imread('./raw_data/test/dcom/1.jpg'),(256*2,256*2))[:,:,0]
predictions = model.predict(im)

plt.figure(figsize=(20,10))
plt.imshow(predictions)
plt.xlabel("Pridiction")
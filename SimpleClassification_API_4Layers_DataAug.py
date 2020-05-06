# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:44:03 2019

@author: Lory
"""
# PRIMA DI LANCIARE PYTHON SU CUDA:
# export THEANO_FLAGS="device=cuda,floatX=float32"
# /home/luca/anaconda2/bin/python


# 07/08/2019: Training set - 01: 28; 02: 28; 03: 18; 04: 22
# Epoch=10, Batch size=4
# DATA AUGMENTATION
# NO BRAIN EXTRACTION -> possibilita' di ridurre le informazioni nei dati e i bias
# TUTTA IMMAGINE


import os
import numpy as np
from nibabel import load as load_nii
from nibabel import save as save_nii
from matplotlib import pyplot
from glob import glob
import cPickle as pickle

from keras.layers import Convolution3D
from keras.layers import MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

# NEW
from keras.models import Model
# Replaced Concatenate with merge (for old version of keras with theano backend)
from keras.layers import Input, merge, Flatten
from keras import metrics
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.ndimage import rotate, zoom


# LOAD TRAINING AND TEST DATA
# Qui va inserito il percorso alla cartella con le immagini di training (tutte assieme)
TRAIN_path='/kitty/home2/mara/DEEP_LEARNING/Training_set'

all_images_T1=sorted(glob(os.path.join(TRAIN_path,'*_t1.nii')))
all_images_T2=sorted(glob(os.path.join(TRAIN_path,'*_t2.nii')))

datatype=np.float32
#images_T1=[load_nii(name).get_data() for name in all_images_T1]
rotation=30
all_images=[]
for name in all_images_T1:
        images_T1=load_nii(name).get_data()
        all_images.append(images_T1)
        all_images.append(rotate(images_T1, rotation, reshape =False))


images_norm_T1=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in all_images]
train_data_T1=np.array(images_norm_T1)
x_train_T1 = train_data_T1.reshape(train_data_T1.shape[0], train_data_T1.shape[1], train_data_T1.shape[2], train_data_T1.shape[3], 1)
input_shape = (train_data_T1.shape[1], train_data_T1.shape[2], train_data_T1.shape[3], 1)

# Stessa cosa con altra modalità di MRI (T2w)
all_images2=[]
for name in all_images_T2:
        images_T2=load_nii(name).get_data()
        all_images2.append(images_T2)
        all_images2.append(rotate(images_T2, rotation, reshape =False))


images_norm_T2=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in all_images2]
train_data_T2=np.array(images_norm_T2)
x_train_T2 = train_data_T2.reshape(train_data_T2.shape[0], train_data_T2.shape[1], train_data_T2.shape[2], train_data_T2.shape[3], 1)

# Devo creare un array con i labels per ciascun soggetto incluso nel training
# a seconda della classe (patologia)
# train_labels.len() deve essere uguale a train_data.shape[0]
a = 0*(np.ones([76,1], dtype=int))
b = np.ones([96,1], dtype=int)
c = 2*(np.ones([96,1], dtype=int))
d = 3*(np.ones([72,1], dtype=int))
# concateno gli array
train_labels=np.concatenate([a,b,c,d], axis=0)

num_classes=4 # numero delle patologie da classificare
y_train = np_utils.to_categorical(train_labels, num_classes)


# Define two input layers
image_input1 = Input((input_shape)) # Modalità MRI 1
image_input2 = Input((input_shape)) # Modalità MRI 2



conv_layer = (Convolution3D(input_shape=input_shape, nb_filter = 28, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(image_input1)
Pool_one=MaxPooling3D(pool_size=(2, 2, 2))(conv_layer) # WARNING
Drop_one=Dropout(0.)(Pool_one)
second_layer=(Convolution3D(nb_filter = 24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_one)
Pool_two=MaxPooling3D(pool_size=(2, 2, 2))(second_layer)
Drop_two=Dropout(0.)(Pool_two)
third_layer = (Convolution3D(input_shape=input_shape, nb_filter = 12, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_two)
Pool_third=MaxPooling3D(pool_size=(2, 2, 2))(third_layer)
Drop_third=Dropout(0.)(Pool_third)
fourth_layer=(Convolution3D(nb_filter = 6, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_third)
Pool_four=MaxPooling3D(pool_size=(2, 2, 2))(fourth_layer)
Drop_fourth=Dropout(0.)(Pool_four)
flat_layer = Flatten()(Drop_fourth)


conv_layer1 = (Convolution3D(input_shape=input_shape, nb_filter = 28, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(image_input2)
Pool_one1=MaxPooling3D(pool_size=(2, 2, 2))(conv_layer1)
Drop_one1=Dropout(0.)(Pool_one1)
second_layer1=(Convolution3D(nb_filter = 24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_one1)
Pool_two1=MaxPooling3D(pool_size=(2, 2, 2))(second_layer1)
Drop_two1=Dropout(0.)(Pool_two1)
third_layer1 = (Convolution3D(input_shape=input_shape, nb_filter = 12, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_two1)
Pool_third1=MaxPooling3D(pool_size=(2, 2, 2))(third_layer1)
Drop_third1=Dropout(0.)(Pool_third1)
fourth_layer1=(Convolution3D(nb_filter = 6, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, border_mode='valid', activation='relu'))(Drop_third1)
Pool_four1=MaxPooling3D(pool_size=(2, 2, 2))(fourth_layer1)
Drop_fourth1=Dropout(0.)(Pool_four1)
flat_layer1 = Flatten()(Drop_fourth1)

concat_layer= merge([flat_layer, flat_layer1])
output = Dense(4)(concat_layer)

# define a model with a list of two inputs
model = Model([image_input1, image_input2], output)

adam=Adam(lr=0.001)
model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mae', 'accuracy'])

# -----------------------------------------------------------------------------------#
# LOAD VALIDATION SET

TEST_path='/kitty/home2/mara/DEEP_LEARNING/Test_set'


all_images_T1=sorted(glob(os.path.join(TEST_path,'*_t1.nii')))
all_images_T2=sorted(glob(os.path.join(TEST_path,'*_t2.nii')))

# DATA AUGMENTATION ON VALIDATION SET?

#datatype=np.float32
#rotation=30
#all_images=[]
#for name in all_images_T1:
#       images_T1=load_nii(name).get_data()
#       all_images.append(images_T1)
#       all_images.append(rotate(images_T1, rotation, reshape =False))

#images_norm_T1=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in all_images]
#test_data_T1=np.array(images_norm_T1)
#x_test_T1 = test_data_T1.reshape(test_data_T1.shape[0], test_data_T1.shape[1], test_data_T1.shape[2], test_data_T1.shape[3], 1)
#input_shape = (test_data_T1.shape[1], test_data_T1.shape[2], test_data_T1.shape[3], 1)


#datatype=np.float32
#all_images2=[]
#for name in all_images_T2:
#       images_T2=load_nii(name).get_data()
#       all_images2.append(images_T2)
#       all_images2.append(rotate(images_T2, rotation, reshape =False))

#images_norm_T2=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in all_images2]
#test_data_T2=np.array(images_norm_T2)
#x_test_T2 = test_data_T2.reshape(test_data_T2.shape[0], test_data_T2.shape[1], test_data_T2.shape[2], test_data_T2.shape[3], 1)



datatype=np.float32
images_T1=[load_nii(name).get_data() for name in all_images_T1]
images_norm_T1=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images_T1]
test_data_T1=np.array(images_norm_T1)
x_test_T1 = test_data_T1.reshape(test_data_T1.shape[0], test_data_T1.shape[1], test_data_T1.shape[2], test_data_T1.shape[3], 1)
input_shape = (test_data_T1.shape[1], test_data_T1.shape[2], test_data_T1.shape[3], 1)

# Stessa cosa con altra modalità di MRI (T2w)
images_T2=[load_nii(name).get_data() for name in all_images_T2]
images_norm_T2=[(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images_T2]
test_data_T2=np.array(images_norm_T2)
x_test_T2 = test_data_T2.reshape(test_data_T2.shape[0], test_data_T2.shape[1], test_data_T2.shape[2], test_data_T2.shape[3], 1)


a = 0*(np.ones([18,1], dtype=int))
b = np.ones([22,1], dtype=int)
c = 2*(np.ones([31,1], dtype=int))
d = 3*(np.ones([15,1], dtype=int))
#train_labels=np.array([[0],[1], []])
# concateno gli array
test_labels=np.concatenate([a,b,c,d], axis=0)

num_classes=4 # numero delle patologie da classificare
y_test = np_utils.to_categorical(test_labels, num_classes)

filepath='/kitty/home2/mara/DEEP_LEARNING/MODEL/Model_4Layers_BestPerfomance_DataAug.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model_check = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
history = model.fit([x_train_T1, x_train_T2], y_train, validation_data=([x_test_T1, x_test_T2], y_test), nb_epoch=10, batch_size=1, verbose=1, callbacks=[early_stopping, model_check])


# PREDICTION

P=model.predict([x_test_T1, x_test_T2], batch_size=1, verbose=1)
A= [np.where(P[p] == np.amax(P[p])) for p in range(0,len(P))]

# SAVE PREDICTIONS
np.save('Predictions', A)

##############################################################################################


# SALVATAGGIO DEL MODELLO



save_dir='/kitty/home2/mara/DEEP_LEARNING/CODE/Training_models/4Layers_batch1_epoch10'
model_name_h5='Training_model_4Layers_batch1.h5'
model_name_json='Training_model_4Layers_batch1.json'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path_h5 = os.path.join(save_dir, model_name_h5)
model.save_weights(model_path_h5)

# serialize model to JSON
model_path_json = os.path.join(save_dir, model_name_json)
model_json = model.to_json()
with open(model_path_json, "w") as json_file:
    json_file.write(model_json)

print('Saved trained model at %s ' % model_path_json)

model_name_dict='Training_model_4Layers_batch1_dict'
model_path_dict = os.path.join(save_dir, model_name_dict)
with open(model_path_dict, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# LOAD SCORES
pickle_out = open("trainHistoryDict","rb")
pickle.load(pickle_out)

#score = model.evaluate([x_test, x_test], y_test, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#np.save('/kitty/home2/mara/DEEP_LEARNING/CODE/Training_prova/prova',score)
#PROVA=np.load('/kitty/home2/mara/DEEP_LEARNING/CODE/Training_prova/prova.npy')


# TEST THE MODEL

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# model.predict(x, batch_size=None, verbose=1, steps=None, callbacks=None)


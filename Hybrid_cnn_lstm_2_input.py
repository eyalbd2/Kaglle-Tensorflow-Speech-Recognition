import pickle
import keras as ke
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Permute
from keras.layers import BatchNormalization,  Reshape, LSTM, GRU
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import add,average, multiply

## extra imports to set GPU options
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#%%

with open('aug_trainSetMFCC_DB64_with_silence.dat', "rb") as load_file:
    trainSetData1 = pickle.load(load_file)
with open('aug_trainSetClasses64_with_silence.dat', "rb") as load_file:
    trainSetClass = pickle.load(load_file)


with open('valSetMFCC_DB64_with_silence.dat', "rb") as load_file:
    valSetData1 = pickle.load(load_file)

with open('valSetClasses64_with_silence.dat', "rb") as load_file:
    valSetClass = pickle.load(load_file)

with open('aug_trainSetMEL_DB64_with_silence.dat', "rb") as load_file:
    trainSetData2 = pickle.load(load_file)

with open('valSetMEL_DB64_with_silence.dat', "rb") as load_file:
    valSetData2 = pickle.load(load_file)


# # show labels probabillity in train dataset
# plt.hist(trainSetClass, bins = [0,1,2,3,4,5,6,7,8,9,10,11])
# plt.title("histogram of Train set")
# plt.show()
#
# # show labels probabillity in valid dataset
# plt.hist(valSetClass, bins = [0,1,2,3,4,5,6,7,8,9,10,11])
# plt.title("histogram of Validation set")
# plt.show()




#%%
# getting the data's dimensions
l_train = len(trainSetData1)
# h_data, w_data = trainSetData[0].shape
h_data, w_data = (64, 64)
# centering the data
X_train_cen1 = []
X_train_cen2 = []
train_sum1 = np.zeros((h_data, w_data))
train_sum2 = np.zeros((h_data, w_data))
for i in range(0, l_train):
    trainSetData1[i] = trainSetData1[i][:, 0:64]
    trainSetData2[i] = trainSetData2[i][:, 0:64]
    train_sum1 = trainSetData1[i] + train_sum1
    train_sum2 = trainSetData2[i] + train_sum2
train_mean1 = train_sum1 / l_train
train_mean2 = train_sum2 / l_train
for mel in trainSetData2:
    X_train_cen2.append(mel-train_mean2)
for mel in trainSetData1:
    X_train_cen1.append(mel - train_mean1)

del trainSetData1, trainSetData2


# transforming the data into the right shape for training
train_mat1 = np.zeros((np.int(1*l_train), h_data, w_data, 1))
for i in range(0, l_train):
    train_mat1[i, :, :, 0] = X_train_cen1[i]
train_mat2 = np.zeros((np.int(1 * l_train), h_data, w_data, 1))
for i in range(0, l_train):
    train_mat2[i, :, :, 0] = X_train_cen2[i]

# creating label matrix
train_label = ke.utils.to_categorical(np.asarray(trainSetClass[0:l_train]), num_classes=12)
del trainSetClass, X_train_cen1, X_train_cen2
# centering the validation set
X_val_cen1 = []
X_val_cen2 = []
l_val = len(valSetData1)
for mel in valSetData1:
    mel = mel[:, 0:64]
    X_val_cen1.append(mel-train_mean1)
for mel in valSetData2:
    mel = mel[:, 0:64]
    X_val_cen2.append(mel - train_mean2)

del valSetData2, valSetData1,

#creating the labeled verification mat + reshaping
val_mat1 = np.zeros((l_val, h_data, w_data, 1))
for i in range(0, l_val):
    val_mat1[i, :, :, 0] = X_val_cen1[i]
val_mat2 = np.zeros((l_val, h_data, w_data, 1))
for j in range(0, l_val):
    val_mat2[j, :, :, 0] = X_val_cen2[j]

val_label = ke.utils.to_categorical(np.asarray(valSetClass), num_classes=12)

del X_val_cen1, X_val_cen2, valSetClass, train_sum1, train_sum2, load_file, mel

#%%

def hybrid():
    model1 = Sequential()

    model1.add(((BatchNormalization(input_shape=(64, 64, 1)))))
    model1.add((Conv2D(16, (3, 3), padding='same', activation='elu')))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(16, (3, 3), padding='same', activation='elu')))
    model1.add((MaxPooling2D()))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model1.add((Dropout(0.2)))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model1.add((MaxPooling2D()))
    model1.add(Dropout(0.3))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(96, (3, 3), padding='same', activation='elu')))
    model1.add((Dropout(0.2)))
    model1.add(((BatchNormalization())))
    model1.add((Conv2D(96, (3, 3), padding='same', activation='tanh')))
    model1.add((Dropout(0.55)))
    model1.add(Permute((3, 2, 1)))

    model2 = Sequential()
    model2.add(((BatchNormalization(input_shape=(64, 64, 1)))))
    model2.add((Conv2D(16, (3, 3), padding='same', activation='elu')))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(16, (3, 3), padding='same', activation='elu')))
    model2.add((MaxPooling2D()))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model2.add((Dropout(0.2)))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model2.add((MaxPooling2D()))
    model2.add(Dropout(0.3))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(96, (3, 3), padding='same', activation='elu')))
    model2.add((Dropout(0.2)))
    model2.add(((BatchNormalization())))
    model2.add((Conv2D(96, (3, 3), padding='same', activation='elu')))
    model2.add((Dropout(0.55)))
    model2.add(Permute((3, 2, 1)))

    merge = multiply([model1.output, model1.output])
    lstm1 = TimeDistributed(Bidirectional(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.2)))(merge)
    flt = Flatten()(lstm1)
    fc1 = Dense(256, activation='elu')(flt)
    drop1 = Dropout(0.55)(fc1)
    fc2 = Dense(96, activation='elu')(drop1)
    drop2 = Dropout(0.55)(fc2)
    bn = BatchNormalization()(drop2)
    fc3 = Dense(12, activation='softmax')(bn)

    model = Model(inputs=[model1.input, model2.input], outputs=fc3)

    return model

filepath = "64x64_model_hybrid_with_silence_2_input_v1_mul.h5"
model = hybrid()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
model.summary()
#%%
checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkPoint]
model.fit([train_mat1,train_mat2], train_label, validation_data=([val_mat1,val_mat2], val_label), batch_size=320, callbacks=callbacks_list, epochs=150,
          shuffle=True, verbose=2)
#%%
score = model.evaluate([val_mat1,val_mat2], val_label, batch_size=32)
#%%
print(score)

# model = ke.models.load_model('64x64_model_v5_high_dim_strides.h5')
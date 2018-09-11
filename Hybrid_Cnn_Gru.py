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
## extra imports to set GPU options
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#%%
# trainSetClass = np.load('aug_trainSetClasses64_alldata.npy')
# trainSetData = np.load('aug_trainSetMEL_DB64_alldata.npy')

# with open('aug_trainSetMEL_DB128_with_silence.dat', "rb") as load_file:
#     trainSetData = pickle.load(load_file)
# with open('aug_trainSetClasses128_with_silence.dat', "rb") as load_file:
#     trainSetClass = pickle.load(load_file)


# with open('valSetMEL_DB128_with_silence.dat', "rb") as load_file:
#     valSetData = pickle.load(load_file)
#
# with open('valSetClasses128_with_silence.dat', "rb") as load_file:
#     valSetClass = pickle.load(load_file)

# with open('aug_trainSetChroma_DB128_with_silence.dat', "rb") as load_file:
#     trainSetData = pickle.load(load_file)
# with open('aug_trainSetChroma_Classes128_with_silence.dat', "rb") as load_file:
#     trainSetClass = pickle.load(load_file)
#
#
# with open('valSetChroma_DB128_with_silence.dat', "rb") as load_file:
#     valSetData = pickle.load(load_file)
#
# with open('valSetChroma_Classes128_with_silence.dat', "rb") as load_file:
#     valSetClass = pickle.load(load_file)

with open('aug_trainSetMFCC_DB128_with_silence.dat', "rb") as load_file:
    trainSetData = pickle.load(load_file)
with open('aug_trainSetClasses128_with_silence.dat', "rb") as load_file:
    trainSetClass = pickle.load(load_file)


with open('valSetMFCC_DB128_with_silence.dat', "rb") as load_file:
    valSetData = pickle.load(load_file)

with open('valSetClasses128_with_silence.dat', "rb") as load_file:
    valSetClass = pickle.load(load_file)

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
l_train = len(trainSetData)
# h_data, w_data = trainSetData[0].shape
h_data, w_data = (64, 128)
# centering the data
X_train_cen = []
train_sum = np.zeros((h_data, w_data))
for i in range(0, l_train):
    trainSetData[i] = trainSetData[i][:, 0:128]
    train_sum = trainSetData[i] + train_sum
train_mean = train_sum / l_train
with open('./train_mean_mfcc', 'wb') as f:
    pickle.dump(train_mean, f)
for mel in trainSetData:
    X_train_cen.append(mel-train_mean)

# transforming the data into the right shape for training
train_mat = np.zeros((np.int(1*l_train), h_data, w_data, 1))
for i in range(0, l_train):
    train_mat[i, :, :, 0] = X_train_cen[i]

# creating label matrix
train_label = ke.utils.to_categorical(np.asarray(trainSetClass[0:l_train]), num_classes=12)

# centering the validation set
X_val_cen = []
l_val = len(valSetData)
for mel in valSetData:
    mel = mel[:, 0:128]
    X_val_cen.append(mel-train_mean)

#creating the labeled verification mat + reshaping
val_mat = np.zeros((l_val, h_data, w_data, 1))
for i in range(0, l_val):
    val_mat[i, :, :, 0] = X_val_cen[i]
val_label = ke.utils.to_categorical(np.asarray(valSetClass), num_classes=12)

del X_val_cen, X_train_cen, train_mean, valSetClass, valSetData, trainSetClass, trainSetData, train_sum, load_file, mel

#%%

def hybrid():
    model = Sequential()

    model.add(((BatchNormalization(input_shape=(64, 128, 1)))))
    model.add((Conv2D(16, (5, 5), padding='same', activation='elu')))
    model.add(((BatchNormalization())))
    model.add((Conv2D(16, (5, 5), padding='same', activation='elu')))
    # model.add(((BatchNormalization())))
    # model.add((Conv2D(16, (3, 3), padding='same', activation='elu')))
    model.add((MaxPooling2D()))

    # model.add(Dropout(0.3))
    model.add(((BatchNormalization())))
    model.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model.add((Dropout(0.2)))
    model.add(((BatchNormalization())))
    model.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    # model.add(((BatchNormalization())))
    # model.add((Conv2D(32, (3, 3), padding='same', activation='elu')))
    model.add((MaxPooling2D()))

    # model.add(Dropout(0.3))
    model.add(((BatchNormalization())))
    model.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model.add(((BatchNormalization())))
    model.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    # model.add(((BatchNormalization())))
    # model.add((Conv2D(64, (3, 3), padding='same', activation='elu')))
    model.add((MaxPooling2D()))

    model.add(((BatchNormalization())))
    model.add((Conv2D(96, (3, 3), padding='same', activation='elu')))
    model.add((Dropout(0.2)))
    model.add(((BatchNormalization())))
    model.add((Conv2D(96, (3, 3), padding='same', activation='elu')))
    # model.add((Dropout(0.2)))
    # model.add((MaxPooling2D())) # larger vector inputs to lstm seq of 16 with each input vector 8
    model.add((Dropout(0.4)))


    model.add(((BatchNormalization())))
    model.add(Permute((3, 2, 1)))
    model.add(TimeDistributed(Bidirectional(LSTM(42, return_sequences=False, dropout=0.4, recurrent_dropout=0.1))))

    model.add((Flatten()))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.45))

    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.45))


    model.add(((BatchNormalization())))
    model.add((Dense(12, activation='softmax')))

    return model


epochs = 75
filepath = "64x128_model_hybrid_with_silence_v17(MFCC)_120.h5"
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
history = model.fit(train_mat, train_label, validation_data=(val_mat, val_label), batch_size=256, callbacks=callbacks_list, epochs=epochs,
          shuffle=True, verbose=2)
#%%
import matplotlib.pyplot as plt
his = history.history
x = list(range(epochs))
y_1 = his['val_loss']
y_2 = his['loss']
plt.plot(x,y_1)
plt.plot(x,y_2)
plt.legend(['validation loss','training_loss'])
plt.savefig('MFCC_training_loss_v2.png')


score = model.evaluate(val_mat, val_label, batch_size=32)
#%%
print(score)

# model = ke.models.load_model('64x64_model_v5_high_dim_strides.h5')
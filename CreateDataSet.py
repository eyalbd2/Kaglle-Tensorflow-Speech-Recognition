##
DATADIR = './train'  # unzipped train and test data
#OUTDIR = './model-k' # just a random name

# Data Loading
import os
import numpy as np
import re
from glob import glob
import librosa

from scipy.io import wavfile


POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

# function for stretching sounds
def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    print(all_files)
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        # r = re.match(pattern, entry)
        # if r:
        #     valset.add(r.group(3))
        temp = entry.split()
        valset.add(temp[0])
    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = entry.split('/')
        label, uid = r[4], r[4]+'/'+r[5]
        if label == '_background_noise_':
            label = 'silence'
        if label not in possible:
            label = 'unknown'

        label_id = name2id[label]

        sample = (label_id, uid, entry)
        if uid in valset:
            val.append(sample)
        else:
            train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val


trainset, valset = load_data(DATADIR)


#%%
trainSetMEL_DB = []
trainSetMFCC = []
# valSetMEL_DB = []
valSetMFCC = []
valSetClasses = []
trainSetClasses = []

idx_min = 0
idx_max = 0
np.random.seed(seed=56)
for index in range(0, len(trainset)):
    if trainset[index][0] == 10:
        sample_rate, samples = wavfile.read(trainset[index][2])
        for idx in range(1500):
            print('Creating Silence')
            start_point = np.int((900000)*(np.random.rand(1)))
            end_point = start_point + 16000
            cur_samples = samples[start_point:end_point-1]
            power_factor = 0.5 + np.random.rand(1)
            cur_samples = cur_samples * power_factor
            S = librosa.feature.melspectrogram(cur_samples.astype(float), sr=sample_rate, n_mels=64, hop_length=250,
                                               n_fft=480, fmin=20, fmax=4000)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=64)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            # choose if 'silence' is going to train or validation
            if np.random.binomial(1, 1400/1500):
                # trainSetMEL_DB.append(log_S)
                trainSetMFCC.append(delta2_mfcc)
                trainSetClasses.append(trainset[index][0])
            else:
                # valSetMEL_DB.append(log_S)
                valSetMFCC.append(delta2_mfcc)
                valSetClasses.append(10)

    else:
        if (trainset[index][0] == 11) & (np.random.binomial(1, 0.80)):
            continue
        sample_rate, samples = wavfile.read(trainset[index][2])
        if len(samples) <16000:
            padlen = 16000-len(samples)
            samples = np.concatenate((samples.astype(float), np.zeros((padlen,))))
        if len(samples) > 16000:
            samples = samples[0:16000-1]
        # original data
        S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=64, hop_length=250, n_fft=480, fmin=20, fmax=4000)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=64)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        # trainSetMEL_DB.append(log_S)
        trainSetMFCC.append(delta2_mfcc)
        trainSetClasses.append(trainset[index][0])
        if not(trainset[index][0] == 11):
            # shifted data
            shift_factor = np.round(np.random.uniform(-2000,2000,1))
            data_roll = np.roll(samples, int(shift_factor))
            S_shift = librosa.feature.melspectrogram(data_roll.astype(float), sr=sample_rate, n_mels=64, hop_length=250, n_fft=480,fmin=20, fmax=4000)
            log_S_shift = librosa.power_to_db(S_shift, ref=np.max)
            mfcc_shift = librosa.feature.mfcc(S=log_S_shift, n_mfcc=64)
            delta2_mfcc_shift = librosa.feature.delta(mfcc_shift, order=2)
            # trainSetMEL_DB.append(log_S_shift)
            trainSetMFCC.append(delta2_mfcc_shift)
            trainSetClasses.append(trainset[index][0])

            # streched data
            strech_factor = np.random.randn(1)/10+1
            if strech_factor>1.4:
                strech_factor = 1.4
            if strech_factor<0.6:
                strech_factor = 0.6
            data_strech = stretch(samples.astype(float), float(strech_factor))
            S_stretch =  librosa.feature.melspectrogram(data_strech.astype(float), sr=sample_rate, n_mels=64, hop_length=250, n_fft=480,fmin=20, fmax=4000)
            log_S_stretch = librosa.power_to_db(S_stretch, ref=np.max)
            mfcc_strech = librosa.feature.mfcc(S=log_S_stretch, n_mfcc=64)
            delta2_mfcc_strech = librosa.feature.delta(mfcc_strech, order=2)
            # trainSetMEL_DB.append(log_S_stretch)
            trainSetMFCC.append(delta2_mfcc_strech)
            trainSetClasses.append(trainset[index][0])

            # shifted and streched data
            strech_factor = np.random.randn(1) / 10 + 1
            if strech_factor > 1.4:
                strech_factor = 1.4
            if strech_factor < 0.6:
                strech_factor = 0.6
            shift_factor = np.round(np.random.uniform(-2000, 2000, 1))
            data_mixed = stretch(samples.astype(float), float(strech_factor))
            data_mixed = np.roll(data_mixed, int(shift_factor))
            S_mixed = librosa.feature.melspectrogram(data_mixed.astype(float), sr=sample_rate, n_mels=64, hop_length=250, n_fft=480,fmin=20, fmax=4000)
            log_S_mixed = librosa.power_to_db(S_mixed, ref=np.max)
            mfcc_mixed = librosa.feature.mfcc(S=log_S_mixed, n_mfcc=64)
            delta2_mfcc_mixed = librosa.feature.delta(mfcc_mixed, order=2)
            # trainSetMEL_DB.append(log_S_mixed)
            trainSetMFCC.append(delta2_mfcc_mixed)
            trainSetClasses.append(trainset[index][0])
    print(index)


## validation set

for index in range(0, len(valset)):
    print(index)
    sample_rate, samples = wavfile.read(valset[index][2])
    if len(samples) < 16000:
        padlen = 16000 - len(samples)
        samples = np.concatenate((samples.astype(float), np.zeros((padlen,))))
    if len(samples) > 16000:
        samples = samples[0:16000 - 1]
    S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=64, hop_length=250, n_fft=480, fmin=20, fmax=4000)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=64)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # valSetMEL_DB.append(log_S)
    valSetMFCC.append(delta2_mfcc)
    valSetClasses.append(valset[index][0])



#%%
print('finished preprocecessing')
import pickle

# with open('./valSetMEL_DB64_with_silence.dat', 'wb') as f:
    # pickle.dump(valSetMEL_DB, f)
with open('./valSetMFCC_DB64_with_silence.dat', 'wb') as f:
    pickle.dump(valSetMFCC, f)
with open('./valSetClasses64_with_silence.dat', 'wb') as f:
    pickle.dump(valSetClasses, f)



# with open('./aug_trainSetMEL_DB64_with_silence.dat', 'wb') as f:
#     pickle.dump(trainSetMEL_DB, f)
with open('./aug_trainSetMFCC_DB64_with_silence.dat', 'wb') as f:
    pickle.dump(trainSetMFCC, f)
with open('./aug_trainSetClasses64_with_silence.dat', 'wb') as f:
    pickle.dump(trainSetClasses, f)

##
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(trainSetMEL_DB[5500],y_axis='mel', fmax=8000,x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
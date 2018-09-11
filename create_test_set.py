# ##
# #OUTDIR = './model-k' # just a random name
#
# # Data Loading
# import os
# import numpy as np
# import re
# from glob import glob
# import librosa
#
# from scipy.io import wavfile
#
#
#
#
# test_dir = os.path.join(DATADIR, 'test/audio/*wav')
# all_files = glob(test_dir)
#
#
# #%%
# test_set = []
#
# for index in range(0, len(all_files)):
#     sample_rate, samples = wavfile.read(all_files[index])
#     if len(samples) <16000:
#         padlen = 16000-len(samples)
#         samples = np.concatenate((samples.astype(float), np.zeros((padlen,))))
#     if len(samples) > 16000:
#         samples = samples[0:16000-1]
#     # original data
#     S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=64, hop_length=125, n_fft=480, fmin=20, fmax=4000)
#     log_S = librosa.power_to_db(S, ref=np.max)
#     test_set.append(log_S)
#
#
# #%%
# print('finished preprocecessing')
# import pickle
# #
# with open('./testSetMEL128.dat', 'wb') as f:
#     pickle.dump(test_set, f)
#
# np.save('testSetMEL_128.npy',test_set)

##
DATADIR = './test'  # unzipped train and test data

# Data Loading
import os
import re
from glob import glob
import numpy as np
from scipy.io import wavfile


def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    test_dir = os.path.join(DATADIR, 'test/audio/*wav')
    all_files = glob(test_dir)
    print(all_files)
    test = []
    for entry in all_files:
        r = entry.split('/')
        uid = r[4]
        print('uid')
        print(uid)
        print('entry')
        print(entry)
        sample = (uid, entry)
        test.append(sample)

    print('There are {} test'.format(len(test)))
    return test


testset = load_data(DATADIR)

np.save('testSet_128.npy',testset)

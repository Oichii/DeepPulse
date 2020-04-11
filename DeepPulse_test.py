import cv2
from tensorflow.keras.models import load_model
import os, glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from DeepPulse_train import NormalisedDifferenceLayer


def gen_test_sequence(sequence, img_height=36, img_width=36, batch_size=1):
    file_path = 'PURE'
    sequence_dir = os.path.join(file_path, sequence)  # folder zawierający klatki sekwencji

    frames = glob.glob(sequence_dir + '/cropped/*.png')  # lista klatek sekwencji
    frames.sort()
    reference = json.load(open(sequence_dir + '.JSON', 'r'))

    ref = []
    seq = []
    seq2 = []

    for sample in reference['/FullPackage']:
        ref.append(sample['Value']['waveform'])
    ref = np.array(ref)
    ref_resample = []
    clean_ref_resample = resample(ref, len(frames))
    for s in range(len(clean_ref_resample) - 1):
        ref_resample.append(clean_ref_resample[s + 1] - clean_ref_resample[s])

    for j in range(len(frames)-1):
        print(j)
        ct = np.array(cv2.imread(frames[j]), dtype='float64')/255
        ct = cv2.resize(ct, (img_height, img_width), interpolation=cv2.INTER_CUBIC)

        # cv2.imwrite('sequence/ct00_{}.png'.format(j), ct)

        ct1 = np.array(cv2.imread(frames[j + 1]), dtype='float64')/255
        ct1 = cv2.resize(ct1, (img_height, img_width), interpolation=cv2.INTER_CUBIC)

        seq.append(ct)
        seq2.append(ct1)
        # print(ref)
        # print(normalized_difference)
    norm_ref = np.array(ref_resample)
    mean = norm_ref.mean()
    std = norm_ref.std()

    # norm_ref = np.clip(norm_ref, mean - std, mean + std)

    norm_ref = (norm_ref - mean) / std
    # norm_ref = 2*((norm_ref - np.amin(norm_ref)) / (np.amax(norm_ref)-np.amin(norm_ref)))-1

    # ref = np.array(ref)
    # plt.plot(range(len(norm_ref)), norm_ref)
    # plt.show()
    seq = np.array(seq)
    seq2 = np.array(seq2)

    seq = (seq-seq.mean())/seq.std()
    seq2 = (seq2-seq2.mean())/seq2.std()

    return seq, seq2, norm_ref


model = load_model('models/set_PURE/model_001.hdf5', custom_objects={'NormalisedDifferenceLayer': NormalisedDifferenceLayer()})


seq, seq2, ref_out = gen_test_sequence('04-01')

# print(ref_out)
# plt.plot(range(len(ref_out)), ref_out)
# plt.show()
# print(len(seq))
out = []
for f in range(len(seq)):
    ct = seq[f][np.newaxis, ...]
    ct1 = seq2[f][np.newaxis, ...]

    nd = np.array(NormalisedDifferenceLayer()([ct, ct1]))
    print(nd[0, ...].shape)
    # cv2.imshow("nd", nd[0, ...])
    # cv2.waitKey()
    print(np.amin(ct), np.amax(ct))
    x = model.predict([ct, ct1], verbose=True)
    print(x, ref_out[f])
    out.append(x)

print(out)
out_signal = []

for i in out:

    j = i[0][0]
    print(i, j)
    out_signal.append(j)

plt.plot(range(len(out_signal)), out_signal, 'g',  range(len(ref_out)), ref_out, 'b--',)
plt.legend(['output', 'reference'])
plt.show()

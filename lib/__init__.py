import torch
import scipy.io
import os
import librosa
import string
import math
import mne
from sklearn.preprocessing import RobustScaler

import pdb
import pandas as pd
import soundfile as sf
import numpy as np
from textgrids import TextGrid

import matplotlib.pyplot as plt
from data_utils import TextTransform

# fmt: off
phoneme_inventory = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']
# fmt: on


def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9) / 9.0
    v = np.convolve(x, f, mode="same")
    w = np.convolve(v, f, mode="same")
    return w


def get_semg_feats_orig(
    eeg_data, hop_length=6, frame_length=16, stft=False, debug=False
):
    xs = eeg_data - eeg_data.mean(axis=1, keepdims=True)
    frame_features = []
    for i in range(eeg_data.shape[0]):
        x = xs[i, :]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(
            w, frame_length=frame_length, hop_length=hop_length
        ).mean(axis=0)
        p_w = librosa.feature.rms(
            y=w, frame_length=frame_length, hop_length=hop_length, center=False
        )
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(
            y=r, frame_length=frame_length, hop_length=hop_length, center=False
        )
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(
            p, frame_length=frame_length, hop_length=hop_length, center=False
        )
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(
            r, frame_length=frame_length, hop_length=hop_length
        ).mean(axis=0)

        if stft:
            s = abs(
                librosa.stft(
                    np.ascontiguousarray(x),
                    n_fft=frame_length,
                    hop_length=hop_length,
                    center=False,
                )
            )

        if debug:
            plt.subplot(7, 1, 1)
            plt.plot(x)
            plt.subplot(7, 1, 2)
            plt.plot(w_h)
            plt.subplot(7, 1, 3)
            plt.plot(p_w)
            plt.subplot(7, 1, 4)
            plt.plot(p_r)
            plt.subplot(7, 1, 5)
            plt.plot(z_p)
            plt.subplot(7, 1, 6)
            plt.plot(r_h)

            plt.subplot(7, 1, 7)
            plt.imshow(s, origin="lower", aspect="auto", interpolation="nearest")

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        if stft:
            frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, "highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 5):  # (1,8)
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def get_usable_datasets(fname):
    mat = scipy.io.loadmat(fname)
    usable = [x for x in mat["use"][0]]  # file listing
    usable = [x[0] for x in usable]  # fnames only
    usable = [x.split(".")[0] for x in usable]  # dataset ids only
    return [x for x in usable]


def load_eeg(fname):
    # raw_eeg = scipy.io.loadmat(fname)
    # raw_eeg = raw_eeg["raw"][0][0][3][0][0]
    raw = mne.io.read_raw_brainvision(fname, preload=True)
    raw_eeg = raw.get_data()
    return raw_eeg


def get_audio_feats(
    audio,
    n_mel_channels=128,
    filter_length=512,
    win_length=432,
    hop_length=160,
    r=16_000,
):
    audio_features = librosa.feature.melspectrogram(
        y=audio,
        sr=r,
        n_mels=n_mel_channels,
        center=False,
        n_fft=filter_length,
        win_length=win_length,
        hop_length=hop_length,
    ).T
    audio_features = np.log(audio_features + 1e-5)
    return audio_features.astype(np.float32)


def load_audio(fname):
    audio, r = sf.read(fname)
    if r != 16_000:
        audio = librosa.resample(audio, orig_sr=r, target_sr=16_000)
        r = 16_000
    assert r == 16_000

    return audio


def load_phonemes(textgrid_fname, audio_feats, phoneme_dict):
    tg = TextGrid(textgrid_fname)
    phone_intervals = tg["phones"]

    seq_len = audio_feats.shape[0]
    phone_ids = np.zeros(seq_len, dtype=np.int64)
    phone_ids[phone_ids == 0] = -1

    for interval in phone_intervals:
        xmin = interval.xmin
        xmax = interval.xmax

        phone = interval.text.lower()
        if phone in ["", "sp", "spn"]:
            phone = "sil"
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_dict.index(phone)

        phone_win_start = int(xmin * 100)
        phone_duration = xmax - xmin
        phone_win_duration = int(math.ceil(phone_duration * 100))
        phone_win_end = phone_win_start + phone_win_duration

        phone_ids[phone_win_start:phone_win_end] = ph_id

    ii = np.where(phone_ids == -1)[0]

    assert (phone_ids >= 0).all(), "missing aligned phones"
    return phone_ids


def load_phoneme_dict(phoneme_dict_path):
    with open(phoneme_dict_path) as f:
        content = [l.split(":")[1].strip() for l in f.read().split("\n")]
    return content


"""
def get_phone_idxs(audio_start,
                   audio_end,
                   audio_feats,
                   phoneme_intervals,
                   phoneme_dict,
                   debug=False):
    # max_ms_dist := how close phoneme needs to be to be considered start of phoneme range
    seq_len   = audio_feats.shape[0]
    phone_ids = np.zeros(seq_len, dtype=np.int64)
    phone_ids[phone_ids == 0] = -1
    # max_dist = max_ms_dist / 1000
    
    # ---
    phone = interval.text.lower()
    # Convert silent phoneme token and strip digits
    if phone in ["", "sp", "spn"]:
        phone = "sil"
    if phone[-1] in string.digits:
        phone = phone[:-1]
    ph_id = phoneme_dict.index(phone)
    # ---

    # Contains the expanded phoneme classes with the entire overlapping phoneme classes
    # print("seq len:", seq_len)
    # print("phoneme_dict:", phoneme_dict)

    rel_phoneme_count = 0 # Phonemes found within desired segment window

    for interval in phoneme_intervals:
        xmin = interval.xmin
        xmax = interval.xmax
        if debug:
            print("interval:", interval, xmin, xmax, audio_start, audio_end)
        if xmin >= audio_start and xmin <= audio_end or \
           xmax >= audio_start and xmax <= audio_end:
            phone = interval.text.lower()
            if phone in ["", "sp", "spn"]:
                phone = "sil"
            if phone[-1] in string.digits:
                phone = phone[:-1]
            ph_id = phoneme_dict.index(phone)

            phone_duration  = (interval.xmax - interval.xmin) #- phone_overspill
            phone_win_start_d = int((xmin - audio_start) * 100)
            
            phone_win_start = phone_win_start_d

            if phone_win_start_d >= 0:
                phone_win_start = max(rel_phoneme_count, phone_win_start_d)
            else:
                phone_win_start = 0

            phone_win_duration = int(math.ceil(phone_duration * 100))
            phone_win_end   = phone_win_start + phone_win_duration

            phone_ids[phone_win_start:phone_win_end] = ph_id

            rel_phoneme_count += 1

    # print(segment_phonemes[0], segment_phonemes[-1])
    # print("phone_ids:", phone_ids, phone_ids.shape)
    
    if debug:
        print([phoneme_dict[phone] for phone in list(phone_ids)])
    assert (phone_ids >= 0).all(), 'missing aligned phones' # f'missing aligned phones: {phone_ids}"
    return phone_ids
"""


def preprocess_eeg(eeg_data):  # based on Meta
    # Remove last two channels
    eeg_data = eeg_data[:-2, :]

    # Apply baseline correction
    baseline = np.mean(eeg_data[:, : int(0.5 * 120)], axis=1)
    eeg_data -= baseline[:, None]

    # Robust scaling using scikit-learn
    scaler = RobustScaler()
    eeg_data = scaler.fit_transform(eeg_data.T).T

    # Clipping the outliers below 5th percentile and above 95th percentile
    eeg_data = np.clip(
        eeg_data, np.percentile(eeg_data, 5), np.percentile(eeg_data, 95)
    )

    # Clamping values greater than 20 standard deviations
    std = np.std(eeg_data)
    mean = np.mean(eeg_data)
    eeg_data = np.clip(eeg_data, mean - 20 * std, mean + 20 * std)

    # Standard normalization for EEG
    eeg_mean = eeg_data.mean()
    eeg_std = eeg_data.std()
    eeg_data = (eeg_data - eeg_mean) / eeg_std

    return eeg_data


class BrennanDataset(torch.utils.data.Dataset):
    num_features = 60 * 5
    num_mels = 128
    audio_hz = 16_000

    def __init__(
        self,
        text_transform,
        root_dir,
        phoneme_dir,
        idx,
        split: str = "train",
        max_items=0,
        phoneme_dict_path="./phoneme_dict.txt",
        debug=False,
    ):
        self.root_dir = root_dir
        self.idx = idx
        self.debug = debug
        self.text_transform = text_transform

        # Metadata
        metadata_fi = os.path.join(root_dir, "AliceChapterOne-EEG.csv")
        self.metadata = pd.read_csv(metadata_fi)

        proc = scipy.io.loadmat(os.path.join(root_dir, f"proc/{idx}.mat"))
        self.eeg_segments = eeg_segments = proc["proc"][0][0][4]
        self.order_idx_s = [seg[-1] for seg in eeg_segments]

        # # Labels
        segment_labels = self.metadata[self.metadata["Order"].isin(self.order_idx_s)]
        self.labels = segment_labels["Word"]

        # # Modify metadata handling to group by sentences, and do some data cleaning on the sentences
        # self.sentences = segment_labels.groupby("Sentence")["Word"].apply(
        #     lambda words: " ".join(word.replace("\x1a", "'").upper() for word in words)
        # )
        # self.sentence_groups = segment_labels.groupby("Sentence")

        # # Store order indices grouped by sentence
        # self.sentence_order_idxs = []
        # for sentence_id in self.sentences.index:
        #     sentence_data = self.sentence_groups.get_group(sentence_id)
        #     self.sentence_order_idxs.append(list(sentence_data["Order"]))

        # if max_items > 0:
        #     self.sentence_order_idxs = self.sentence_order_idxs[:max_items]
        #     self.sentences = self.sentences[:max_items]

        # EEG
        eeg_path = os.path.join(root_dir, f"{idx}.vhdr")
        eeg_data = load_eeg(eeg_path)
        self.eeg_data = eeg_data

    def __getitem__(self, i):

        # # Get all order indices for the sentence
        # sentence_order_idxs = self.sentence_order_idxs[i]
        # sentence = self.sentences.iloc[i]
        label = self.labels[i]

        # EEG Segment
        order_idx = self.order_idx_s[i]
        powerline_freq = 60  # Assumption based on US recordings
        cur_eeg_segment = [seg for seg in self.eeg_segments if seg[-1] == order_idx][0]
        brain_shift = 150  # Mental response time to stimuli
        eeg_start_idx = int(cur_eeg_segment[0]) + brain_shift
        eeg_end_idx = int(cur_eeg_segment[1]) + brain_shift
        eeg_x = self.eeg_data[:, eeg_start_idx:eeg_end_idx]
        eeg_x = notch_harmonics(eeg_x, powerline_freq, 500)
        eeg_x = remove_drift(eeg_x, 500)
        # Meta's preprocessing
        eeg_x = preprocess_eeg(eeg_x)
        eeg_feats = get_semg_feats_orig(eeg_x, hop_length=4, stft=False)
        eeg_raw = apply_to_all(subsample, eeg_x.T, 400, 500)

        # Get label int
        label_int = self.text_transform.text_to_int(label)

        # Dict Segment
        data = {
            "label": label,
            "label_int": label_int,
            "eeg_raw": torch.from_numpy(eeg_raw).pin_memory(),
            "eeg_feats": torch.from_numpy(eeg_feats).pin_memory(),
        }

        if self.debug:
            print(i, label)

        return data

    def __len__(self):
        return len(self.order_idx_s)

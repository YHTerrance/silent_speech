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
        self.phoneme_dir = phoneme_dir
        self.phoneme_dict_path = phoneme_dict_path
        self.debug = debug
        self.text_transform = text_transform

        # Metadata
        metadata_fi = os.path.join(root_dir, "AliceChapterOne-EEG.csv")
        self.metadata = metadata = pd.read_csv(metadata_fi)

        proc = scipy.io.loadmat(os.path.join(root_dir, f"proc/{idx}.mat"))
        self.eeg_segments = eeg_segments = proc["proc"][0][0][4]
        self.order_idx_s = order_idx_s = [seg[-1] for seg in eeg_segments]

        # Labels
        segment_labels = metadata[metadata["Order"].isin(order_idx_s)]
        # self.labels = segment_labels["Word"]

        # Modify metadata handling to group by sentences, and do some data cleaning on the sentences
        self.sentences = segment_labels.groupby("Sentence")["Word"].apply(
            lambda words: " ".join(word.replace("\x1a", "h").upper() for word in words)
        )
        self.sentence_groups = segment_labels.groupby("Sentence")

        # Store order indices grouped by sentence
        self.sentence_order_idxs = []
        for sentence_id in self.sentences.index:
            sentence_data = self.sentence_groups.get_group(sentence_id)
            self.sentence_order_idxs.append(list(sentence_data["Order"]))

        if max_items > 0:
            self.sentence_order_idxs = self.sentence_order_idxs[:max_items]
            self.sentences = self.sentences[:max_items]

        # EEG
        eeg_path = os.path.join(root_dir, f"{idx}.vhdr")
        eeg_data = load_eeg(eeg_path)
        self.eeg_data = eeg_data

        # Audio
        audio_dir = os.path.join(root_dir, "audio")
        audio_idx_range = range(1, 12 + 1 if not self.debug else 3 + 1)
        self.audio_raw_s = [
            load_audio(
                os.path.join(
                    audio_dir, f"DownTheRabbitHoleFinal_SoundFile{audio_idx}.wav"
                )
            )
            for audio_idx in audio_idx_range
        ]
        self.audio_feat_s = [
            get_audio_feats(
                audio_raw,
                hop_length=int(self.audio_hz / 100),
                n_mel_channels=self.num_mels,
            )
            for audio_raw in self.audio_raw_s
        ]

        # Phoneme Dictionary
        phoneme_dict = load_phoneme_dict(phoneme_dict_path)
        phoneme_dict = [phone.lower() for phone in phoneme_dict]
        phoneme_dict[0] = "sil"
        for sil_tok in ["sp", "spn"]:  # silence tokens
            if sil_tok in phoneme_dict:
                phoneme_dict.remove(sil_tok)
        for i in range(len(phoneme_dict)):
            if phoneme_dict[i][-1] in string.digits:
                phoneme_dict[i] = phoneme_dict[i][:-1]
        phoneme_dict = list(dict.fromkeys(phoneme_dict))
        self.phoneme_dict = phoneme_dict

        # Phonemes
        phoneme_fis = os.listdir(phoneme_dir)
        phoneme_idx_range = range(1, 12 + 1 if not self.debug else 3 + 1)
        self.phoneme_s = [
            load_phonemes(
                os.path.join(
                    phoneme_dir,
                    f"DownTheRabbitHoleFinal_SoundFile{phoneme_idx_range[i]}.TextGrid",
                ),
                audio_feats,
                phoneme_dict,
            )
            for i, audio_feats in enumerate(self.audio_feat_s)
        ]

    def __getitem__(self, i):
        # Target Audio Hz
        audio_hz = 16_000

        # Get all order indices for the sentence
        sentence_order_idxs = self.sentence_order_idxs[i]
        sentence = self.sentences.iloc[i]

        audio_segments = {}
        for order_idx in sentence_order_idxs:
            metadata_entry = self.metadata[self.metadata["Order"] == order_idx].iloc[0]
            audio_segment = (
                metadata_entry["Segment"] - 1
            )  # 0-index, not original 1-index
            if audio_segment not in audio_segments:
                audio_segments[audio_segment] = []
            audio_segments[audio_segment].append(order_idx)

        # Process each segment and concatenate
        audio_raw_parts = []
        audio_feats_parts = []
        phonemes_parts = []
        for segment, segment_order_idxs in audio_segments.items():
            start_order_idx = segment_order_idxs[0]
            end_order_idx = segment_order_idxs[-1]

            # Get audio boundaries for this segment
            audio_len = self.audio_raw_s[segment].shape[0] / audio_hz
            start_metadata_entry = self.metadata[
                self.metadata["Order"] == start_order_idx
            ].iloc[0]
            end_metadata_entry = self.metadata[
                self.metadata["Order"] == end_order_idx
            ].iloc[0]

            start_audio_onset = start_metadata_entry["onset"]
            end_audio_onset = end_metadata_entry["onset"]
            audio_start = max(start_audio_onset - 0.3, 0)
            audio_end = min(end_audio_onset + 1.0, audio_len)

            audio_start_idx = int(audio_start * audio_hz)
            audio_end_idx = int(audio_end * audio_hz)
            audio_raw_parts.append(
                self.audio_raw_s[segment][audio_start_idx:audio_end_idx]
            )

            # Get corresponding audio features
            audio_start_win = int(audio_start * 100)
            audio_end_win = int(audio_end * 100)
            audio_feats_parts.append(
                self.audio_feat_s[segment][audio_start_win:audio_end_win]
            )

            # Phoneme Segment
            phonemes_parts.append(
                self.phoneme_s[segment][audio_start_win:audio_end_win]
            )

        # Concatenate all parts
        audio_raw = np.concatenate(audio_raw_parts)
        audio_feats = np.concatenate(audio_feats_parts)
        phonemes = np.concatenate(phonemes_parts)
        # EEG Segment
        powerline_freq = 60  # Assumption based on US recordings

        start_cur_eeg_segment = [
            seg for seg in self.eeg_segments if seg[-1] == start_order_idx
        ][0]

        end_cur_eeg_segment = [
            seg for seg in self.eeg_segments if seg[-1] == end_order_idx
        ][0]

        brain_shift = 150  # Mental response time to stimuli
        eeg_start_idx = int(start_cur_eeg_segment[0]) + brain_shift
        eeg_end_idx = int(end_cur_eeg_segment[1]) + brain_shift
        eeg_x = self.eeg_data[:, eeg_start_idx:eeg_end_idx]
        eeg_x = notch_harmonics(eeg_x, powerline_freq, 500)
        eeg_x = remove_drift(eeg_x, 500)
        # Meta's preprocessing
        eeg_x = preprocess_eeg(eeg_x)
        eeg_feats = get_semg_feats_orig(eeg_x, hop_length=4, stft=False)
        eeg_raw = apply_to_all(subsample, eeg_x.T, 400, 500)

        label_int = np.array(self.text_transform.text_to_int(sentence))

        # Dict Segment
        data = {
            "label": sentence,
            "label_int": torch.from_numpy(label_int).pin_memory(),
            "audio_feats": torch.from_numpy(audio_feats).pin_memory(),
            "audio_raw": torch.from_numpy(audio_raw).pin_memory(),
            "eeg_raw": torch.from_numpy(eeg_raw).pin_memory(),
            "eeg_feats": torch.from_numpy(eeg_feats).pin_memory(),
            "phonemes": torch.from_numpy(phonemes).pin_memory(),
        }

        if self.debug:
            print(i, sentence)

        return data

    def __len__(self):
        return len(self.sentences)

    # def get_label_idxs(self, target_label):
    #     labels = list(self.labels)
    #     # print(labels)
    #     if self.max_items:
    #         labels = labels[0: self.max_items]

    #     if target_label in labels:
    #         matches = [i for i in range(
    #             len(labels)) if target_label == labels[i]]
    #         return matches
    #     else:
    #         ValueError(
    #             f"Attempting to get label {target_label} which does not exist.")


class BrennanSeqDataset(torch.utils.data.Dataset):
    num_features = 60 * 5
    num_mels = 80
    audio_hz = 16_000

    def __init__(
        self,
        root_dir,
        phoneme_dir,
        idx,
        max_items=0,
        phoneme_dict_path="./phoneme_dict.txt",
        seq_len=5,  # Add sequence length as a parameter
        debug=False,
    ):
        self.root_dir = root_dir
        self.idx = idx
        self.phoneme_dir = phoneme_dir
        self.phoneme_dict_path = phoneme_dict_path
        self.debug = debug
        self.seq_len = seq_len  # Sequence length for grouping items

        # Metadata
        metadata_fi = os.path.join(root_dir, "AliceChapterOne-EEG.csv")
        self.metadata = metadata = pd.read_csv(metadata_fi)
        proc = scipy.io.loadmat(os.path.join(root_dir, f"proc/{idx}.mat"))
        self.eeg_segments = eeg_segments = proc["proc"][0][0][4]
        self.order_idx_s = order_idx_s = [seg[-1] for seg in eeg_segments]
        if max_items > 0:
            self.order_idx_s = self.order_idx_s[:max_items]
        self.max_items = max_items

        # Labels
        segment_labels = metadata[metadata["Order"].isin(order_idx_s)]
        self.labels = list(
            segment_labels["Word"]
        )  # Convert to list for sequence slicing

        # EEG
        eeg_path = os.path.join(root_dir, f"{idx}.vhdr")
        eeg_data = load_eeg(eeg_path)
        self.eeg_data = eeg_data

        # Audio
        audio_dir = os.path.join(root_dir, "audio")
        audio_idx_range = range(1, 12 + 1 if not self.debug else 3 + 1)
        self.audio_raw_s = [
            load_audio(
                os.path.join(
                    audio_dir, f"DownTheRabbitHoleFinal_SoundFile{audio_idx}.wav"
                )
            )
            for audio_idx in audio_idx_range
        ]
        self.audio_feat_s = [
            get_audio_feats(
                audio_raw,
                hop_length=int(self.audio_hz / 100),
                n_mel_channels=self.num_mels,
            )
            for audio_raw in self.audio_raw_s
        ]

        # Phoneme Dictionary
        phoneme_dict = load_phoneme_dict(phoneme_dict_path)
        phoneme_dict = [phone.lower() for phone in phoneme_dict]
        phoneme_dict[0] = "sil"
        for sil_tok in ["sp", "spn"]:  # silence tokens
            if sil_tok in phoneme_dict:
                phoneme_dict.remove(sil_tok)
        for i in range(len(phoneme_dict)):
            if phoneme_dict[i][-1] in string.digits:
                phoneme_dict[i] = phoneme_dict[i][:-1]
        phoneme_dict = list(dict.fromkeys(phoneme_dict))
        self.phoneme_dict = phoneme_dict

        # Phonemes
        phoneme_fis = os.listdir(phoneme_dir)
        phoneme_idx_range = range(1, 12 + 1 if not self.debug else 3 + 1)
        self.phoneme_s = [
            load_phonemes(
                os.path.join(
                    phoneme_dir,
                    f"DownTheRabbitHoleFinal_SoundFile{phoneme_idx_range[i]}.TextGrid",
                ),
                audio_feats,
                phoneme_dict,
            )
            for i, audio_feats in enumerate(self.audio_feat_s)
        ]

    def __getitem__(self, i):
        # Ensure the index `i` is valid for the desired sequence length
        if i > len(self):
            raise IndexError("Index exceeds dataset length for the desired sequence.")

        # Initialize lists to store sequences
        seq_labels = []
        seq_audio_feats = []
        seq_audio_raw = []
        seq_eeg_feats = []
        seq_eeg_raw = []
        seq_phonemes = []

        # Loop through the sequence length
        for j in range(self.seq_len):
            idx = i * self.seq_len + j  # Current index in the sequence

            # Target Audio Hz
            audio_hz = 16_000

            # Metadata Segment
            order_idx = self.order_idx_s[idx]  # EEG
            label = self.labels[idx]  # Text Label
            metadata_entry = self.metadata[self.metadata["Order"] == order_idx]
            audio_segment = (
                metadata_entry.iloc[0]["Segment"] - 1
            )  # 0-index, not original 1-index

            # Audio Segment
            audio_len = self.audio_raw_s[audio_segment].shape[0] / audio_hz
            audio_onset = metadata_entry.iloc[0]["onset"]
            audio_start = max(audio_onset - 0.3, 0)
            audio_end = min(audio_onset + 1.0, audio_len)
            audio_start_idx = int(audio_start * audio_hz)
            audio_end_idx = int(audio_end * audio_hz)
            audio_raw = self.audio_raw_s[audio_segment][audio_start_idx:audio_end_idx]

            # Audio Feats Segment
            audio_start_win = int(audio_start * 100)
            audio_end_win = int(audio_end * 100)
            audio_feats = self.audio_feat_s[audio_segment][
                audio_start_win:audio_end_win
            ]

            # EEG Segment
            powerline_freq = 60  # Assumption based on US recordings
            cur_eeg_segment = [
                seg for seg in self.eeg_segments if seg[-1] == order_idx
            ][0]
            brain_shift = 150  # Mental response time to stimuli
            eeg_start_idx = int(cur_eeg_segment[0]) + brain_shift
            eeg_end_idx = int(cur_eeg_segment[1]) + brain_shift
            eeg_x = self.eeg_data[:, eeg_start_idx:eeg_end_idx]
            eeg_x = notch_harmonics(eeg_x, powerline_freq, 500)
            eeg_x = remove_drift(eeg_x, 500)
            eeg_x = preprocess_eeg(eeg_x)
            eeg_feats = get_semg_feats_orig(eeg_x, hop_length=4, stft=False)
            eeg_raw = apply_to_all(subsample, eeg_x.T, 400, 500)

            # Phoneme Segment
            phonemes = self.phoneme_s[audio_segment][audio_start_win:audio_end_win]

            # Append to sequence lists
            seq_labels.append(label)
            seq_audio_feats.append(audio_feats)
            seq_audio_raw.append(audio_raw)
            seq_eeg_feats.append(eeg_feats)
            seq_eeg_raw.append(eeg_raw)
            seq_phonemes.append(phonemes)
        # Concatenate the sequence into a single array
        seq_labels = " ".join(seq_labels)
        seq_audio_feats = np.concatenate(seq_audio_feats, axis=0)
        seq_audio_raw = np.concatenate(seq_audio_raw, axis=0)
        seq_eeg_feats = np.concatenate(seq_eeg_feats, axis=0)
        seq_eeg_raw = np.concatenate(seq_eeg_raw, axis=0)
        seq_phonemes = np.concatenate(seq_phonemes, axis=0)

        # Combine the sequence into a dictionary
        data = {
            "labels": seq_labels,
            "audio_feats": seq_audio_feats,
            "audio_raw": seq_audio_raw,
            "eeg_raw": seq_eeg_raw,
            "eeg_feats": seq_eeg_feats,
            "phonemes": seq_phonemes,
        }

        if self.debug:
            print(i, seq_labels)

        return data

    # Returns the number of sequences in the dataset
    def __len__(self):
        return len(self.order_idx_s) // self.seq_len

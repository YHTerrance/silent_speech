import pdb
import os
import logging
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Literal
from lib import BrennanDataset, BrennanSentenceDataset
from data_utils import TextTransformOrig, WordTransform
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
from collections import defaultdict

base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")

class EEGDataset(ConcatDataset):
    def __init__(
        self,
        datasets,
        text_transform,
        num_features,
        time_mask_param=20,
        freq_mask_param=10,
    ):
        super().__init__(datasets)
        self.text_transform = text_transform
        self.num_features = num_features
        self.time_masking = None
        self.freq_masking = None
        if time_mask_param:
            self.time_masking = T.TimeMasking(
                time_mask_param=time_mask_param,
                iid_masks=True,
                p=0.5,
            )
        if freq_mask_param:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    @classmethod
    def from_subjects_word_cls(
        cls,
        subjects,
        base_dir,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        time_mask_param=50,
        freq_mask_param=10,
    ):
        text_transform = WordTransform()
        preload_dataset = {}
        word2sub = defaultdict(list)  # word_id: [subject_ids]

        for subIdx, subject in enumerate(subjects):
            dataset = BrennanDataset(
                text_transform=text_transform,
                root_dir=base_dir,
                idx=subject,
            )
            preload_dataset[subIdx] = dataset

            for wordId in range(len(dataset)):
                word2sub[wordId].append(subIdx)

        subs2word_train = defaultdict(list)  # sub: [word_ids]
        subs2word_dev = defaultdict(list)  # sub: [word_ids]
        subs2word_test = defaultdict(list)  # sub: [word_ids]

        for wordId, subIdx in word2sub.items():
            # stratify by sentences
            if len(subIdx) < 10:
                continue

            # Subject IDs
            train_idx, test_idx = train_test_split(subIdx, train_size=train_ratio)
            dev_idx, test_idx = train_test_split(
                test_idx, train_size=dev_ratio / (dev_ratio + test_ratio)
            )
            for sub in train_idx:
                subs2word_train[sub].append(wordId)
            for sub in dev_idx:
                subs2word_dev[sub].append(wordId)
            for sub in test_idx:
                subs2word_test[sub].append(wordId)

        trainsets, devsets, testsets = [], [], []
        for subject_id, subject in enumerate(subjects):
            trainsets.append(
                Subset(preload_dataset[subject_id], subs2word_train[subject_id])
            )
            devsets.append(
                Subset(preload_dataset[subject_id], subs2word_dev[subject_id])
            )
            testsets.append(
                Subset(preload_dataset[subject_id], subs2word_test[subject_id])
            )
        sample_eeg = trainsets[0][0]["eeg_raw"]

        # Concatenate all datasets
        return (
            cls(
                trainsets,
                text_transform,
                num_features=sample_eeg.shape[1],
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
            ),
            cls(
                devsets,
                text_transform,
                num_features=sample_eeg.shape[1],
            ),
            cls(
                testsets,
                text_transform,
                num_features=sample_eeg.shape[1],
            ),
        )
        
    @classmethod
    def from_subjects_seq2seq(
        cls,
        subjects,
        base_dir,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        time_mask_param=50,
        freq_mask_param=10,
    ):
        text_transform = TextTransformOrig()
        preload_dataset = {}
        sen2sub = defaultdict(list)  # sentence_id: [subs]

        for subIdx, subject in enumerate(subjects):
            dataset = BrennanSentenceDataset(
                text_transform=text_transform,
                root_dir=base_dir,
                idx=subject,
            )

            preload_dataset[subIdx] = dataset
            for sentId in range(len(dataset)):
                sen2sub[sentId].append(subIdx)

        subs2sen_train = defaultdict(list)  # sub: [sentences]
        subs2sen_dev = defaultdict(list)  # sub: [sentences]
        subs2sen_test = defaultdict(list)  # sub: [sentences]

        for sentId, subIdx in sen2sub.items():
            # stratify by sentences
            if len(subIdx) < 10:
                continue

            # Subject IDs
            train_idx, test_idx = train_test_split(subIdx, train_size=train_ratio)
            dev_idx, test_idx = train_test_split(
                test_idx, train_size=dev_ratio / (dev_ratio + test_ratio)
            )
            for sub in train_idx:
                subs2sen_train[sub].append(sentId)
            for sub in dev_idx:
                subs2sen_dev[sub].append(sentId)
            for sub in test_idx:
                subs2sen_test[sub].append(sentId)

        trainsets, devsets, testsets = [], [], []
        for subject_id, subject in enumerate(subjects):
            trainsets.append(
                Subset(preload_dataset[subject_id], subs2sen_train[subject_id])
            )
            devsets.append(
                Subset(preload_dataset[subject_id], subs2sen_dev[subject_id])
            )
            testsets.append(
                Subset(preload_dataset[subject_id], subs2sen_test[subject_id])
            )
        sample_eeg = trainsets[0][0]["eeg_raw"]

        # Concatenate all datasets
        return (
            cls(
                trainsets,
                text_transform,
                num_features=sample_eeg.shape[1],
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
            ),
            cls(
                devsets,
                text_transform,
                num_features=sample_eeg.shape[1],
                time_mask_param=0,
                freq_mask_param=0,
            ),
            cls(
                testsets,
                text_transform,
                num_features=sample_eeg.shape[1],
                time_mask_param=0,
                freq_mask_param=0,
            ),
        )

    def collate_seq2seq(self, batch):
        batch_size = len(batch)
        eeg_raw = [ex["eeg_raw"] for ex in batch]  # B x T x C
        lengths = [ex["eeg_raw"].shape[0] for ex in batch]
        sentence_ints = [ex["sentence_int"] for ex in batch]
        sentence_lengths = [ex["sentence_int"].shape[0] for ex in batch]
        sentences = [ex["sentence"] for ex in batch]
        if self.time_masking or self.freq_masking:
            for i in range(batch_size):
                eeg_raw[i] = eeg_raw[i].unsqueeze(0)
                eeg_raw[i] = eeg_raw[i].transpose(1, 2)
                if self.time_masking:
                    eeg_raw[i] = self.time_masking(eeg_raw[i])
                if self.freq_masking:
                    eeg_raw[i] = self.freq_masking(eeg_raw[i])
                eeg_raw[i] = eeg_raw[i].transpose(1, 2)
                eeg_raw[i] = eeg_raw[i].squeeze(0)

        result = {
            "eeg_raw": eeg_raw,
            "sentences": sentences,
            "lengths": lengths,
            "sentence_int": sentence_ints,
            "sentence_lengths": sentence_lengths,
        }
        return result

    def collate_word_cls(self, batch):
        batch_size = len(batch)
        lengths = [ex["eeg_raw"].shape[0] for ex in batch]
        word_ints = [ex["word_int"] for ex in batch]
        max_seq_len = 520
        eeg_raw = []
        for i in range(batch_size):
            eeg = batch[i]["eeg_raw"]  # Tensor of shape [T, C]
            if eeg.shape[0] < max_seq_len:  # Pad shorter sequences
                pad_length = max_seq_len - eeg.shape[0]
                eeg_raw.append(
                    torch.cat([eeg, torch.zeros(pad_length, eeg.shape[1])], dim=0)
                )  # Shape: [max_seq_len, C]
            else:  # Truncate longer sequences
                eeg_raw.append(eeg[:max_seq_len])
        words = [ex["word"] for ex in batch]
        
        if self.time_masking or self.freq_masking:
            for i in range(batch_size):
                eeg_raw[i] = eeg_raw[i].unsqueeze(0)
                eeg_raw[i] = eeg_raw[i].transpose(1, 2)
                if self.time_masking:
                    eeg_raw[i] = self.time_masking(eeg_raw[i])
                if self.freq_masking:
                    eeg_raw[i] = self.freq_masking(eeg_raw[i])
                eeg_raw[i] = eeg_raw[i].transpose(1, 2)
                eeg_raw[i] = eeg_raw[i].squeeze(0)

        result = {
            "eeg_raw": eeg_raw,
            "lengths": lengths,
            "words": words,
            "word_ints": word_ints,
        }
        return result

    def verify_dataset(self):
        """
        Verify the EEGDataset by checking:
        1. All samples have valid EEG data and labels
        2. Dimensions are consistent
        3. No NaN or infinite values
        """
        print("Verifying dataset...")

        # Track dimensions of first sample for comparison
        first_sample = self[0]
        eeg_dims = first_sample["eeg_raw"].shape[1]

        # Track max sequence length
        max_seq_len = 0

        for i, sample in enumerate(self):
            # Check that required keys exist
            assert "eeg_raw" in sample, f"Sample {i} missing EEG data"
            assert "label_int" in sample, f"Sample {i} missing label"

            # Check dimensions
            assert (
                sample["eeg_raw"].shape[1] == eeg_dims
            ), f"Sample {i} has inconsistent EEG dimensions: {sample['eeg_raw'].shape[1]} vs {eeg_dims}"

            # Check for invalid values
            assert torch.isfinite(
                sample["eeg_raw"]
            ).all(), f"Sample {i} contains NaN or infinite values in EEG data"
            assert torch.isfinite(
                sample["label_int"]
            ).all(), f"Sample {i} contains NaN or infinite values in labels"

            # Check that lengths are non-zero
            assert (
                sample["eeg_raw"].shape[0] > 0
            ), f"Sample {i} has zero-length EEG data"
            assert len(sample["label_int"]) > 0, f"Sample {i} has zero-length label"

            # Update max sequence length
            max_seq_len = max(max_seq_len, sample["eeg_raw"].shape[0])

        print(f"Dataset verification complete. {len(self)} samples checked.")
        print(f"EEG feature dimensions: {eeg_dims}")
        print(f"Longest sequence length: {max_seq_len}")

        return max_seq_len


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = EEGDataset.from_subjects(
        subjects=["S01", "S03", "S04"],
        base_dir=base_dir,
    )
    print(
        f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}"
    )

    print(train_dataset[0])

    train_dataloder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        collate_fn=EEGDataset.collate_raw,
    )

    test_dataloder = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        collate_fn=EEGDataset.collate_raw,
    )

def load_datasets(subjects = [], dataset_type: Literal["seq2seq", "word_cls"] = "seq2seq"):

    base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")

    if dataset_type == "seq2seq":
        datasets_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice/datasets/seq2seq")
    elif dataset_type == "word_cls":
        datasets_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice/datasets/word_cls")
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    # check if trainset.pkl, devset.pkl, testset.pkl exist in the path, if not create them and pickel them
    if (
            not os.path.exists(datasets_dir / "trainset.pkl")
            or not os.path.exists(datasets_dir / "devset.pkl")
            or not os.path.exists(datasets_dir / "testset.pkl")
        ):

        if dataset_type == "seq2seq":
            trainset, devset, testset = EEGDataset.from_subjects_seq2seq(
                subjects=subjects,
                base_dir=base_dir,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
            )
        elif dataset_type == "word_cls":
            trainset, devset, testset = EEGDataset.from_subjects_word_cls(
                subjects=subjects,
                base_dir=base_dir,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
            )
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # train_max_seq_len = trainset.verify_dataset()
        # dev_max_seq_len = devset.verify_dataset()
        # test_max_seq_len = testset.verify_dataset()

        # max_seq_len = max(train_max_seq_len, dev_max_seq_len, test_max_seq_len)
        # logging.info("max sequence length: %d", max_seq_len)

        logging.info(
            "train / dev / test split: %d %d %d",
            len(trainset),
            len(devset),
            len(testset),
        )

        with open(datasets_dir / "trainset.pkl", "wb") as f:
            pickle.dump(trainset, f)
        with open(datasets_dir / "devset.pkl", "wb") as f:
            pickle.dump(devset, f)
        with open(datasets_dir / "testset.pkl", "wb") as f:
            pickle.dump(testset, f)
    else:
        with open(datasets_dir / "trainset.pkl", "rb") as f:
            trainset = pickle.load(f)
        with open(datasets_dir / "devset.pkl", "rb") as f:
            devset = pickle.load(f)
        with open(datasets_dir / "testset.pkl", "rb") as f:
            testset = pickle.load(f)
                
    return trainset, devset, testset
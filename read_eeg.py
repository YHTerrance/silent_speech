import pdb
import pandas as pd
import numpy as np
from lib import BrennanDataset, BrennanSeqDataset
import torch
from data_utils import TextTransform, TextTransformOrig
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T

base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")
# phoneme_dir = "/ocean/projects/cis240129p/shared/data/eeg_alice/phonemes"
# phoneme_dict_path = "/ocean/projects/cis240129p/shared/data/eeg_alice/phoneme_dict.txt"
subjects_used = [
    "S01",
    "S03",
    "S04",
    "S13",
    "S18",
    "S19",
    "S37",
    # "S38",
    "S41",
    "S42",
    "S44",
    "S48",
]  # exclude 'S05' - less channels / other good channels: "S13", "S19"


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
    def from_subjects(
        cls,
        subjects,
        base_dir,
        train_ratio=0.7,
        dev_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    ):

        trainsets = []
        devsets = []
        testsets = []

        text_transform = TextTransformOrig()

        for subject in subjects:

            dataset = BrennanDataset(
                text_transform=text_transform,
                root_dir=base_dir,
                phoneme_dir=base_dir / "phonemes",
                idx=subject,
                phoneme_dict_path=base_dir / "phoneme_dict.txt",
            )
            num_data_points = len(dataset)
            indices = np.arange(num_data_points)

            # First split: separate train and temp (val + test)
            train_indices, temp_indices = train_test_split(
                indices, train_size=train_ratio, random_state=random_state, shuffle=True
            )

            # Second split: separate val and test from temp
            relative_dev_ratio = dev_ratio / (dev_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=relative_dev_ratio,
                random_state=random_state,
                shuffle=True,
            )

            # Create Subset datasets using indices
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            # Append to respective lists
            trainsets.append(train_dataset)
            devsets.append(val_dataset)
            testsets.append(test_dataset)

            # Print split sizes for verification
            print(f"Subject {subject} splits:")
            print(
                f"  Train: {len(train_indices)} ({len(train_indices)/num_data_points:.1%})"
            )
            print(f"  Val: {len(val_indices)} ({len(val_indices)/num_data_points:.1%})")
            print(
                f"  Test: {len(test_indices)} ({len(test_indices)/num_data_points:.1%})"
            )

        for sample in dataset:
            sample_eeg = sample["eeg_raw"]
            break

        # Concatenate all datasets
        return (
            cls(
                trainsets,
                text_transform,
                num_features=sample_eeg.shape[1],
                time_mask_param=50,
                freq_mask_param=10,
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

    # @staticmethod
    def collate_raw(self, batch):
        batch_size = len(batch)
        eeg_raw = [ex["eeg_raw"] for ex in batch]  # B x T x C
        lengths = [ex["eeg_raw"].shape[0] for ex in batch]
        text_ints = [ex["label_int"] for ex in batch]
        text_lengths = [ex["label_int"].shape[0] for ex in batch]
        labels = [ex["label"] for ex in batch]
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
            "labels": labels,
            # "eeg_generated": eeg_generated,
            "lengths": lengths,
            "text_int": text_ints,
            "text_int_lengths": text_lengths,
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
        subjects=["S04"],
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

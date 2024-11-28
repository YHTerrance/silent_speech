import pdb
import pandas as pd
import numpy as np
from lib import BrennanDataset
import torch
from data_utils import TextTransform
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")
# phoneme_dir = "/ocean/projects/cis240129p/shared/data/eeg_alice/phonemes"
# phoneme_dict_path = "/ocean/projects/cis240129p/shared/data/eeg_alice/phoneme_dict.txt"
subjects_used = [
    "S04"
]  # exclude 'S05' - less channels / other good channels: "S13", "S19"


class EEGDataset(ConcatDataset):
    def __init__(self, datasets, text_transform, num_features):
        super().__init__(datasets)
        self.text_transform = text_transform
        self.num_features = num_features

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    @classmethod
    def from_subjects(cls, subjects, base_dir, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, random_state=42):

        trainsets = []
        devsets = []
        testsets = []

        text_transform = TextTransform()

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
                indices,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
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
            print(
                f"  Val: {len(val_indices)} ({len(val_indices)/num_data_points:.1%})")
            print(
                f"  Test: {len(test_indices)} ({len(test_indices)/num_data_points:.1%})")

        for sample in dataset:
            sample_eeg = sample["eeg_raw"]
            break

        # Concatenate all datasets
        return (
            cls(trainsets, text_transform, num_features=sample_eeg.shape[1]),
            cls(devsets, text_transform, num_features=sample_eeg.shape[1]),
            cls(testsets, text_transform, num_features=sample_eeg.shape[1]),
        )

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        eeg_raw = [ex["eeg_raw"] for ex in batch]
        lengths = [ex["eeg_raw"].shape[0] for ex in batch]
        text_ints = [ex["label_int"] for ex in batch]
        text_lengths = [ex["label_int"].shape[0] for ex in batch]

        result = {
            "eeg_raw": eeg_raw,
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
            assert sample["eeg_raw"].shape[1] == eeg_dims, \
                f"Sample {i} has inconsistent EEG dimensions: {sample['eeg_raw'].shape[1]} vs {eeg_dims}"
            
            # Check for invalid values
            assert torch.isfinite(sample["eeg_raw"]).all(), \
                f"Sample {i} contains NaN or infinite values in EEG data"
            assert torch.isfinite(sample["label_int"]).all(), \
                f"Sample {i} contains NaN or infinite values in labels"
            
            # Check that lengths are non-zero
            assert sample["eeg_raw"].shape[0] > 0, f"Sample {i} has zero-length EEG data"
            assert len(sample["label_int"]) > 0, f"Sample {i} has zero-length label"
            
            # Update max sequence length
            max_seq_len = max(max_seq_len, sample["eeg_raw"].shape[0])
            
        print(f"Dataset verification complete. {len(self)} samples checked.")
        print(f"EEG feature dimensions: {eeg_dims}")
        print(f"Longest sequence length: {max_seq_len}")

        return max_seq_len



    # def collate_fn(batch):
    #     """
    #     A custom collate function that handles different types of data in a batch.
    #     It dynamically creates batches by converting arrays or lists to tensors and
    #     applies padding to variable-length sequences.
    #     """
    #     batch_dict = {}
    #     for key in batch[0].keys():
    #         batch_items = [item[key] for item in batch]
    #         if isinstance(batch_items[0], np.ndarray) or isinstance(
    #             batch_items[0], torch.Tensor
    #         ):
    #             if isinstance(batch_items[0], np.ndarray):
    #                 batch_items = [torch.tensor(b) for b in batch_items]
    #             if len(batch_items[0].shape) > 0:
    #                 batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
    #                     batch_items, batch_first=True  # pad with zeros
    #                 )
    #             else:
    #                 batch_dict[key] = torch.stack(batch_items)
    #         else:
    #             batch_dict[key] = batch_items

    #     lengths = [a.shape[0] for a in batch_dict["eeg_raw"]]
    #     batch_dict["lengths"] = torch.tensor(lengths)

    #     return batch_dict


# if __name__ == "__main__":
#     train_dataset, val_dataset, test_dataset = create_datasets(
#         subjects_used, base_dir)
#     print(
#         f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}"
#     )

#     print(train_dataset[0])

#     train_dataloder = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=2,
#         num_workers=1,
#         shuffle=True,
#         collate_fn=collate_fn,
#     )

#     test_dataloder = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=2,
#         num_workers=1,
#         shuffle=False,
#         collate_fn=collate_fn,
#     )

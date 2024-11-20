import pandas as pd
import numpy as np
from lib import BrennanDataset
import torch
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")
# phoneme_dir = "/ocean/projects/cis240129p/shared/data/eeg_alice/phonemes"
# phoneme_dict_path = "/ocean/projects/cis240129p/shared/data/eeg_alice/phoneme_dict.txt"
subjects_used = [
    "S04"
]  # exclude 'S05' - less channels / other good channels: "S13", "S19"


def create_datasets(
    subjects, base_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Create train, validation, and test datasets from multiple subjects.

    Args:
        subjects: List of subject identifiers
        base_dir: Base directory containing the dataset
        train_size: Proportion of data for training (default: 0.7)
        val_size: Proportion of data for validation (default: 0.15)
        test_size: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split proportions must sum to 1")

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for subject in subjects:
        dataset = BrennanDataset(
            root_dir=base_dir,
            phoneme_dir=base_dir / "phonemes",
            idx=subject,
            phoneme_dict_path=base_dir / "phoneme_dict.txt",
        )
        num_data_points = len(dataset)
        indices = np.arange(num_data_points)

        # First split: separate train and temp (val + test)
        train_indices, temp_indices = train_test_split(
            indices, train_size=train_size, random_state=random_state, shuffle=True
        )

        # Second split: separate val and test from temp
        relative_val_size = val_size / (val_size + test_size)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=relative_val_size,
            random_state=random_state,
            shuffle=True,
        )

        # Create Subset datasets using indices
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Append to respective lists
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

        # Print split sizes for verification
        print(f"Subject {subject} splits:")
        print(
            f"  Train: {len(train_indices)} ({len(train_indices)/num_data_points:.1%})"
        )
        print(f"  Val: {len(val_indices)} ({len(val_indices)/num_data_points:.1%})")
        print(f"  Test: {len(test_indices)} ({len(test_indices)/num_data_points:.1%})")

    # Concatenate all datasets
    return (
        ConcatDataset(train_datasets),
        ConcatDataset(val_datasets),
        ConcatDataset(test_datasets),
    )


def collate_fn(batch):
    """
    A custom collate function that handles different types of data in a batch.
    It dynamically creates batches by converting arrays or lists to tensors and
    applies padding to variable-length sequences.
    """
    batch_dict = {}
    lengths = [ex["audio_feats"].shape[0] for ex in batch]
    for key in batch[0].keys():
        batch_items = [item[key] for item in batch]
        if isinstance(batch_items[0], np.ndarray) or isinstance(
            batch_items[0], torch.Tensor
        ):
            if isinstance(batch_items[0], np.ndarray):
                batch_items = [torch.tensor(b) for b in batch_items]
            if len(batch_items[0].shape) > 0:
                batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                    batch_items, batch_first=True  # pad with zeros
                )
            else:
                batch_dict[key] = torch.stack(batch_items)
        else:
            batch_dict[key] = batch_items
    batch_dict["lengths"] = torch.tensor(lengths)
    return batch_dict


if __name__ == "__main__":
    train_ds, test_ds = create_datasets(subjects_used, base_dir)
    train_dataset = ConcatDataset(train_ds)
    test_dataset = ConcatDataset(test_ds)
    print(
        f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}"
    )

    train_dataloder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_dataloder = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

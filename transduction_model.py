import os
import sys
import numpy as np
import logging
import subprocess
from pathlib import Path

import soundfile as sf
import tqdm

import torch
import torch.nn.functional as F
import pickle
from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from align import align_from_distances
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from vocoder import Vocoder

from absl import flags

from read_eeg import create_datasets, collate_fn

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_integer("epochs", 5, "number of training epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("learning_rate_patience", 5, "learning rate decay patience")
flags.DEFINE_integer("learning_rate_warmup", 500, "steps of linear warmup")
flags.DEFINE_string("start_training_from", None, "start training from this model")
flags.DEFINE_float("data_size_fraction", 0.01, "fraction of training data to use")
flags.DEFINE_float(
    "phoneme_loss_weight", 0, "weight of auxiliary phoneme prediction loss"
)
flags.DEFINE_float("l2", 1e-7, "weight decay")
flags.DEFINE_string("output_directory", "output", "output directory")


def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=32, collate_fn=collate_fn
    )
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    seq_len = 200
    with torch.no_grad():
        for id, batch in tqdm.tqdm(enumerate(dataloader), "Validation", disable=None):
            # X = combine_fixed_length(
            #     [t.to(device, non_blocking=True) for t in batch["emg"]], seq_len
            # )
            # X_raw = combine_fixed_length(
            #     [t.to(device, non_blocking=True) for t in batch["eeg_raw"]], seq_len * 8
            # )
            # sess = combine_fixed_length(
            #     [t.to(device, non_blocking=True) for t in batch["session_ids"]], seq_len
            # )

            X_raw = batch["eeg_raw"].float().to(device)
            pred, phoneme_pred = model(X_raw)
            loss, phon_acc = dtw_loss(
                pred, phoneme_pred, batch, True, phoneme_confusion
            )
            losses.append(loss.item())

            accuracies.append(phon_acc)

    model.train()
    return (
        np.mean(losses),
        np.mean(accuracies),
        phoneme_confusion,
    )  # TODO size-weight average


def save_output(model, datapoint, filename, device, vocoder):  # audio_normalizer,
    model.eval()
    with torch.no_grad():
        X_raw = (
            torch.Tensor(datapoint["eeg_raw"])
            .to(dtype=torch.float32, device=device)
            .unsqueeze(0)
        )

        pred, pred_phoneme = model(X_raw)
        y = pred.squeeze(0)

        # y = audio_normalizer.inverse(y.cpu()).to(device)

        audio = vocoder(y).cpu().numpy()
    # save phoneme prediction as pickle
    with open(filename + ".phoneme", "wb") as f:
        pickle.dump(pred_phoneme.squeeze(0).cpu().numpy(), f)

    sf.write(filename, audio, 22050)

    model.train()


# Used key: lengths, audio_feats, phonemes, silent
def dtw_loss(
    predictions,
    phoneme_predictions,
    example,
    phoneme_eval=False,
    phoneme_confusion=None,
):
    device = predictions.device
    predictions = decollate_tensor(predictions, example["lengths"])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example["lengths"])

    audio_feats = [t.to(device, non_blocking=True) for t in example["audio_feats"]]

    phoneme_targets = example["phonemes"]

    losses = []
    correct_phones = 0
    total_length = 0
    for pred, y, pred_phone, y_phone in zip(
        predictions,
        audio_feats,
        phoneme_predictions,
        phoneme_targets,
        # example["silent"],
    ):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)

        if y.size(0) != pred.size(0):
            print(f"y: {y.size()}, pred: {pred.size()}")
            # print all pred's length
            for p in predictions:
                print(p.size())

        assert y.size(0) == pred.size(0), f"{y.size()} != {pred.size()}"
        dists = F.pairwise_distance(y, pred)  # audio_feats v.s. model output

        if not pred_phone is None:
            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction="sum")
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss
        else:
            loss = dists.sum()

        if phoneme_eval:
            pred_phone = pred_phone.argmax(-1)
            correct_phones += (pred_phone == y_phone).sum().item()

            for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y.size(0)

    return sum(losses) / total_length, correct_phones / total_length


def train_model(
    trainset,
    devset,
    device,
    num_features,
    num_speech_features,
    save_sound_outputs=False,
):
    n_epochs = FLAGS.epochs

    # if FLAGS.data_size_fraction >= 1:
    training_subset = trainset
    # else:
    # training_subset = trainset.subset(FLAGS.data_size_fraction)

    dataloader = torch.utils.data.DataLoader(
        training_subset,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        num_workers=0,
        batch_size=32,
        # batch_sampler=SizeAwareSampler(training_subset, 256000),
    )

    n_phones = len(phoneme_inventory)

    model = Model(num_features, num_speech_features, n_phones).to(device)
    # model = Model(num_features, num_speech_features, None).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    if save_sound_outputs:
        vocoder = Vocoder()

    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, "min", 0.5, patience=FLAGS.learning_rate_patience
    )

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    # seq_len = 200 TODO: Not used as we do not understood its purpose see below

    batch_idx = 0
    for epoch_idx in range(n_epochs):
        losses = []
        for batch in tqdm.tqdm(dataloader, "Train step", disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            X_raw = batch["eeg_raw"].float().to(device)

            pred, phoneme_pred = model(X_raw)
            loss, _ = dtw_loss(pred, phoneme_pred, batch)

            losses.append(loss.item())

            loss.backward()
            optim.step()

            batch_idx += 1

        train_loss = np.mean(losses)
        val, phoneme_acc, _ = test(model, devset, device)

        lr_sched.step(val)
        logging.info(
            f"finished epoch {epoch_idx+1} - validation loss: {val:.4f} training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc*100:.2f}"
        )
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, "model.pt"))
        if save_sound_outputs:
            save_output(
                model,
                devset[0],
                os.path.join(FLAGS.output_directory, f"epoch_{epoch_idx}_output.wav"),
                device,
                # devset.mfcc_norm,
                vocoder,
            )

    if save_sound_outputs:
        for i, datapoint in enumerate(devset):
            save_output(
                model,
                datapoint,
                os.path.join(FLAGS.output_directory, f"example_output_{i}.wav"),
                device,
                # devset.mfcc_norm,
                vocoder,
            )

        evaluate(devset, FLAGS.output_directory)

    return model


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, "log.txt"), "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        ).stdout
    )
    logging.info(
        subprocess.run(
            ["git", "diff"], stdout=subprocess.PIPE, universal_newlines=True
        ).stdout
    )

    logging.info(sys.argv)

    # trainset = EMGDataset(dev=False, test=False)
    # devset = EMGDataset(dev=True)

    base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")
    subjects_used = ["S04"]
    trainset, devset, _ = create_datasets(subjects_used, base_dir)
    sample_audio_feats = trainset[0]["audio_feats"]
    sample_raw_eeg = trainset[0]["eeg_raw"]
    num_features = sample_raw_eeg.shape[-1]
    num_speech_features = sample_audio_feats.shape[-1]

    # logging.info("output example: %s", devset.example_indices[0])
    logging.info("train / dev split: %d %d", len(trainset), len(devset))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_model(
        trainset,
        devset,
        device,
        num_features=num_features,
        num_speech_features=num_speech_features,
        save_sound_outputs=(FLAGS.hifigan_checkpoint is not None),
    )


if __name__ == "__main__":
    FLAGS(sys.argv)
    main()

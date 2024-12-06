import sys
import pdb
import torch
import os
import logging
import tqdm
import jiwer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import decollate_tensor, combine_fixed_length, save_model
from ctcdecode import CTCBeamDecoder
from read_eeg import EEGDataset, load_datasets
from eeg_architecture import EEGModel
from config import subjects
import gc
import json
import wandb
import pickle

from pathlib import Path
from absl import flags
import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string("output_directory", "output",
                    "where to save models and outputs")
flags.DEFINE_boolean("debug", False, "debug")
flags.DEFINE_string("start_training_from", None,
                    "start training from this model")
flags.DEFINE_float("l2", 0, "weight decay")
flags.DEFINE_integer("epochs", 100, "number of training epochs")
flags.DEFINE_integer("batch_size", 8, "training batch size")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_integer("learning_rate_warmup", 1000, "steps of linear warmup")
flags.DEFINE_integer("learning_rate_patience", 5,
                     "learning rate decay patience")
flags.DEFINE_string("evaluate_saved", None,
                    "run evaluation on given model file")
flags.DEFINE_string("wandb_name", "word_cls", "wandb run name")


def train_model(trainset, devset, device):
    torch.cuda.empty_cache()
    gc.collect()
    n_epochs = FLAGS.epochs
    dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        pin_memory=(device == "cuda"),
        num_workers=0,
        collate_fn=trainset.collate_raw,
        batch_size=FLAGS.batch_size,
    )
    chars = list(trainset.text_transform.chars.keys()) + ["_"]
    blank_id = n_chars = len(chars) - 1
    decoder = CTCBeamDecoder(
        chars,
        blank_id=blank_id,
        log_probs_input=True,
        model_path="lm.binary",
        alpha=1.5,
        beta=1.85,
        beam_width=30,
    )

    model = EEGModel(devset.num_features, n_chars + 1).to(device)
    wandb.login(key="1b4a08dec829dd8f2d99985b647c44f28c0e2b23", relogin=True)
    run = wandb.init(
        name=FLAGS.wandb_name,
        reinit=True,
        project="eeg-alice",
    )
    expt_root = run.dir
    os.makedirs(expt_root, exist_ok=True)
    model_arch = str(model)
    model_path = os.path.join(expt_root, "model_arch.txt")
    arch_file = open(model_path, "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()
    wandb.watch(model, log="all")

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(
            FLAGS.start_training_from, map_location=torch.device(device)
        )
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.AdamW(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2
    )
    # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[125, 150, 175], gamma=0.5
    )

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        losses = []
        wers = []
        curr_lr = float(optim.param_groups[0]["lr"])
        batch_idx = 0
        results = []
        for example in tqdm.tqdm(dataloader, "Train step", disable=None):
            schedule_lr(batch_idx)
            X = combine_fixed_length(
                example["eeg_raw"], 5000).float().to(device)
            pred = model(X)
            pred = F.log_softmax(pred, 2)

            pred_lengths = [l // 4 for l in example["lengths"]]
            pred = nn.utils.rnn.pad_sequence(
                decollate_tensor(pred, pred_lengths),
                batch_first=False,
                padding_value=n_chars - 1,
            )  # seq first, as required by ctc

            y = nn.utils.rnn.pad_sequence(
                example["text_int"],
                batch_first=True,
                padding_value=n_chars - 1,
            ).to(device)

            loss = F.ctc_loss(
                pred, y, pred_lengths, example["text_int_lengths"], blank=n_chars
            )
            losses.append(loss.item())
            loss.backward()
            pred = pred.permute(1, 0, 2)
            beam_results, _, _, out_lens = decoder.decode(
                # pred should be B x T x C
                pred, seq_lens=torch.tensor(pred_lengths)
            )

            # Calculate WER for each batch
            references = []
            predictions = []
            for i in range(len(y)):
                target_text = trainset.text_transform.int_to_text(
                    y[i].cpu().numpy())
                # target_text = target_text.replace(trainset.text_transform.pad_token, "")
                references.append(target_text.strip())
                if i < len(beam_results):
                    pred_int = beam_results[i, 0, : out_lens[i, 0]].tolist()
                    try:
                        pred_text = trainset.text_transform.int_to_text(
                            pred_int)
                        # pred_text = pred_text.replace(
                        #     trainset.text_transform.pad_token, ""
                        # )
                    except:
                        print(f"!!!ERROR!!! batch idx: {batch_idx}, i: {i}")
                        exit()
                    predictions.append(pred_text.strip())
                    results.append(
                        {"original": target_text, "predicted": pred_text})
            wer = jiwer.wer(references, predictions)
            wers.append(wer)
            if (batch_idx + 1) % 2 == 0:
                optim.step()
                optim.zero_grad()
            batch_idx += 1
            torch.cuda.empty_cache()

        train_loss = np.mean(losses)
        train_wer = np.mean(wers)
        val = test(model, devset, device, epoch_idx)
        lr_sched.step()
        logging.info(
            f"finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} training WER:{train_wer*100:.2f} validation WER: {val*100:.2f} lr: {curr_lr}"
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "train_wer": train_wer,
                "val_wer": val,
                "lr": curr_lr,
            }
        )
        save_model(
            model,
            optimizer=optim,
            scheduler=lr_sched,
            metric=("WER", val),
            epoch=epoch_idx,
            path=os.path.join(FLAGS.output_directory, "model.pt"),
        )
        if epoch_idx % 5 == 0:
            output_json_dir = os.path.join(FLAGS.output_directory, "text")
            os.makedirs(output_json_dir, exist_ok=True)
            output_json_path = os.path.join(
                output_json_dir, f"epoch_{epoch_idx}_results.json"
            )
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
    run.finish()
    return model


def test(model, testset, device, epoch_idx=0):
    model.eval()

    chars = list(testset.text_transform.chars.keys()) + ["_"]
    blank_id = len(chars) - 1
    decoder = CTCBeamDecoder(
        chars,
        blank_id=blank_id,
        log_probs_input=True,
        model_path="lm.binary",
        alpha=1.5,
        beta=1.85,
        beam_width=30,
    )
    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=1  # , collate_fn=EEGDataset.collate_raw, num_workers=0
    )

    references = []
    predictions = []
    results = []
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            X = example["eeg_raw"].float().to(device)
            pred = F.log_softmax(model(X), -1)
            # pred_lengths = [l // 4 for l in example["lengths"]]
            beam_results, _, _, out_lens = decoder.decode(
                pred  # , seq_lens=torch.tensor(pred_lengths)
            )
            pred_int = beam_results[0, 0, : out_lens[0, 0]].tolist()

            pred_text = testset.text_transform.int_to_text(pred_int)
            target_text = testset.text_transform.clean_text(
                example["label"][0])

            references.append(target_text)
            predictions.append(pred_text)
            results.append({"original": target_text, "predicted": pred_text})
    if epoch_idx % 5 == 0:
        output_json_dir = os.path.join(FLAGS.output_directory, "text_val")
        os.makedirs(output_json_dir, exist_ok=True)
        output_json_path = os.path.join(
            output_json_dir, f"epoch_{epoch_idx}_results.json"
        )
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
    model.train()
    return jiwer.wer(references, predictions)


def evaluate_saved():
    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    testset = EEGDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = EEGModel(testset.num_features, n_chars + 1).to(device)
    model.load_state_dict(
        torch.load(FLAGS.evaluate_saved, map_location=torch.device(device))
    )
    print("WER:", test(model, testset, device))


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(
                FLAGS.output_directory, "log.txt"), "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(sys.argv)

    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    print("device:", device)

    trainset, devset, testset = load_datasets(subjects, dataset_type="seq2seq")
    model = train_model(trainset, devset, device)
    test_wer = test(model, testset, device)
    logging.info("Test WER: %f", test_wer)


if __name__ == "__main__":
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()

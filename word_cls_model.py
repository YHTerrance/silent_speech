import sys
import pdb
import torch
import os
import logging
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from data_utils import save_model
from read_eeg import EEGDataset
from eeg_architecture import EEGModel
import gc
from pathlib import Path
from absl import flags
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("output_directory", "output", "where to save models and outputs")
flags.DEFINE_boolean("debug", False, "debug")
flags.DEFINE_string("start_training_from", None, "start training from this model")
flags.DEFINE_float("l2", 0, "weight decay")
flags.DEFINE_integer("epochs", 100, "number of training epochs")
flags.DEFINE_integer("batch_size", 8, "training batch size")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_integer("k", 10, "top k accuracy")
flags.DEFINE_string("evaluate_saved", None, "run evaluation on given model file")
flags.DEFINE_string("wandb_name", "word_cls", "wandb run name")


def train_model(trainset, devset, device, max_seq_len):
    wandb.login(key="1b4a08dec829dd8f2d99985b647c44f28c0e2b23", relogin=True)
    run = wandb.init(
        name=FLAGS.wandb_name,
        reinit=True,
        project="eeg-alice",
    )
    expt_root = run.dir
    os.makedirs(expt_root, exist_ok=True)
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
    n_chars = trainset.text_transform.vocabs_size
    model = EEGModel(devset.num_features, n_chars).to(device)
    model_arch = str(model)
    model_path = os.path.join(expt_root, "model_arch.txt")
    arch_file = open(model_path, "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()
    # save model_arch to wandb
    wandb.save(model_path)
    wandb.watch(model, log="all")
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(
            FLAGS.start_training_from, map_location=torch.device(device)
        )
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.AdamW(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=0.01
    )
    # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=5, factor=0.5, mode="max"
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    k = FLAGS.k
    best_topk = 0.0
    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        running_loss = 0.0
        correct = 0
        top_k_correct = 0
        total = 0
        model.train()
        curr_lr = float(optim.param_groups[0]["lr"])
        for example in tqdm.tqdm(dataloader, "Train step", disable=None):
            X = torch.stack(example["eeg_raw"]).float().to(device)  # [B x T x C]
            pred = model(X)
            y = torch.tensor(example["text_int"]).to(device)  # [B]
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            _, predicted = torch.max(pred, dim=1)
            correct += (predicted == y).sum().item()
            _, top_k_indices = torch.topk(pred, k, dim=1)
            y_expanded = y.view(-1, 1).expand_as(top_k_indices)
            top_k_correct += (top_k_indices == y_expanded).sum().item()
            total += y.size(0)
            torch.cuda.empty_cache()

        train_loss = running_loss / len(dataloader)
        accuracy = correct / total
        top_k_accuracy = top_k_correct / total if total > 0 else 0.0
        val_acc, val_topk = test(model, devset, device)
        lr_sched.step(val_acc)
        logging.info(
            f"epoch {epoch_idx+1} - training loss: {train_loss:.4f} acc:{accuracy*100:.2f} "
            f"top-{k} acc: {top_k_accuracy*100:.2f} "
            f"validation acc: {val_acc*100:.2f} top-{k} acc: {val_topk*100:.2f}"
            f"lr: {curr_lr}"
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": accuracy,
                "train_top_k_accuracy": top_k_accuracy,
                "val_accuracy": val_acc,
                "val_top_k_accuracy": val_topk,
                "lr": curr_lr,
            }
        )
        save_model(
            model,
            optimizer=optim,
            scheduler=lr_sched,
            metric=("top_k_accuracy", top_k_accuracy),
            epoch=epoch_idx,
            path=os.path.join(FLAGS.output_directory, "model.pt"),
        )
        if val_topk > best_topk:
            best_topk = val_topk
            save_model(
                model,
                optimizer=optim,
                scheduler=lr_sched,
                metric=("top_k_accuracy", top_k_accuracy),
                epoch=epoch_idx,
                path=os.path.join(FLAGS.output_directory, "best_model.pt"),
            )


def test(model, testset, device):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=1, collate_fn=testset.collate_raw, num_workers=0
    )
    correct = 0
    total = 0
    top_k_correct = 0
    k = FLAGS.k
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            X = torch.stack(example["eeg_raw"]).float().to(device)  # B x T x C
            pred = model(X)
            y = torch.tensor(example["text_int"]).to(device)
            _, predicted = torch.max(pred, dim=1)
            correct += (predicted == y).sum().item()
            _, top_k_indices = torch.topk(pred, k, dim=1)
            y_expanded = y.view(-1, 1).expand_as(top_k_indices)
            top_k_correct += (top_k_indices == y_expanded).sum().item()
            total += y.size(0)
    accuracy = correct / total if total > 0 else 0.0
    top_k_accuracy = top_k_correct / total if total > 0 else 0.0
    model.train()
    return accuracy, top_k_accuracy


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
            logging.FileHandler(os.path.join(FLAGS.output_directory, "log.txt"), "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(sys.argv)

    base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")

    subjects = [
        "S01",
        "S03",
        "S04",
        # "S05", missing one channel
        "S08",
        "S11",
        "S12",
        "S13",
        "S16",
        "S17",
        "S18",
        "S19",
        "S22",
        "S26",
        "S36",
        "S37",
        # "S38", missing one channel
        "S40",
        "S41",
        "S42",
        "S44",
        "S48",
    ]
    logging.info(f"subjects: {subjects}")
    # subjects = ["S04", "S13"]

    # Train on 2x, 3x, 4x... increase subjects in this list
    # generated_subjects = [""]

    trainset, devset, testset = EEGDataset.from_subjects(
        subjects=subjects,
        # generated_subjects=generated_subjects,
        base_dir=base_dir,
    )

    train_max_seq_len = trainset.verify_dataset()
    dev_max_seq_len = devset.verify_dataset()
    test_max_seq_len = testset.verify_dataset()

    max_seq_len = max(train_max_seq_len, dev_max_seq_len, test_max_seq_len)

    logging.info(
        "train / dev / test split: %d %d %d", len(trainset), len(devset), len(testset)
    )

    logging.info("max sequence length: %d", max_seq_len)

    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    print("device:", device)

    model = train_model(trainset, devset, device, max_seq_len)
    test_acc, test_topk = test(model, testset, device)
    logging.info(
        f"test accuracy: {test_acc*100:.2f} top-{FLAGS.k} acc: {test_topk*100:.2f}"
    )


if __name__ == "__main__":
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()

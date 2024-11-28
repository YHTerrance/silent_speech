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
from data_utils import decollate_tensor, combine_fixed_length
from ctcdecode import CTCBeamDecoder
from read_emg import EMGDataset
from read_eeg import EEGDataset
from eeg_architecture import EEGModel

from pathlib import Path
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('output_directory', 'output',
                    'where to save models and outputs')
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('start_training_from', None,
                    'start training from this model')
flags.DEFINE_float('l2', 0, 'weight decay')
flags.DEFINE_integer('batch_size', 2, 'training batch size')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_string('evaluate_saved', None, 'run evaluation on given model file')


def train_model(trainset, devset, device, max_seq_len, n_epochs=100):
    dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        pin_memory=(device == 'cuda'),
        num_workers=4,
        collate_fn=EEGDataset.collate_raw,
        batch_size=FLAGS.batch_size,
    )

    n_chars = len(devset.text_transform.chars)
    model = EEGModel(devset.num_features, n_chars+1).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from,
                                map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.AdamW(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[125, 150, 175], gamma=.5)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    batch_idx = 0

    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        losses = []
        for example in tqdm.tqdm(dataloader, 'Train step', disable=None):
            schedule_lr(batch_idx)

            X = combine_fixed_length(example['eeg_raw'], 5000).float().to(device)
            # X = nn.utils.rnn.pad_sequence(
                # example['eeg_raw'], batch_first=True).float().to(device)
            
            pred = model(X)
            pred = F.log_softmax(pred, 2)

            pred_lengths = [l // 4 for l in example['lengths']]
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(
                pred, pred_lengths), batch_first=False)  # seq first, as required by ctc
            # pred = nn.utils.rnn.pad_sequence(pred, batch_first=False)

            y = nn.utils.rnn.pad_sequence(
                example['text_int'], batch_first=True).to(device)
            
            loss = F.ctc_loss(
                pred, y, pred_lengths, example['text_int_lengths'], blank=n_chars)
            losses.append(loss.item())

            loss.backward()
            if (batch_idx+1) % 2 == 0:
                optim.step()
                optim.zero_grad()

            batch_idx += 1
        train_loss = np.mean(losses)
        val = test(model, devset, device)
        lr_sched.step()
        logging.info(
            f'finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}')
        torch.save(model.state_dict(), os.path.join(
            FLAGS.output_directory, 'model.pt'))


def test(model, testset, device):
    model.eval()

    blank_id = len(testset.text_transform.chars)
    decoder = CTCBeamDecoder(
        testset.text_transform.chars+'_',
        blank_id=blank_id,
        log_probs_input=True,
        model_path='lm.binary',
        alpha=1.5,
        beta=1.85
    )
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)

    references = []
    predictions = []
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, 'Evaluate', disable=None):
            X = example['eeg_raw'].float().to(device)
            # X_raw = example['raw_emg'].to(device)
            # sess = example['session_ids'].to(device)

            pred = F.log_softmax(model(X), -1)

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(
                pred)
            pred_int = beam_results[0, 0, :out_lens[0, 0]].tolist()

            pdb.set_trace()

            pred_text = testset.text_transform.int_to_text(pred_int)
            target_text = testset.text_transform.clean_text(example['label'][0])

            references.append(target_text)
            predictions.append(pred_text)

    model.train()
    return jiwer.wer(references, predictions)


def evaluate_saved():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = EEGModel(testset.num_features, n_chars+1).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved,
                          map_location=torch.device(device)))
    print('WER:', test(model, testset, device))


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(
            FLAGS.output_directory, 'log.txt'), 'w'),
        logging.StreamHandler()
    ], level=logging.INFO, format="%(message)s")

    logging.info(sys.argv)

    base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")

    subjects = ["S04"]
    # subjects = ["S04", "S13"]

    # Train on 2x, 3x, 4x... increase subjects in this list
    # generated_subjects = [""]

    trainset, devset, testset = EEGDataset.from_subjects(
        subjects=subjects,
        # generated_subjects=generated_subjects,
        base_dir=base_dir
    )

    train_max_seq_len = trainset.verify_dataset()
    dev_max_seq_len = devset.verify_dataset()
    test_max_seq_len = testset.verify_dataset()

    max_seq_len = max(train_max_seq_len, dev_max_seq_len, test_max_seq_len)

    logging.info("train / dev / test split: %d %d %d",
                 len(trainset), len(devset), len(testset))

    logging.info("max sequence length: %d", max_seq_len)    

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'

    model = train_model(trainset, devset, device, max_seq_len)


if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()

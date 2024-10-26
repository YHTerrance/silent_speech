##
# Conclusions so far:
# - more data is better, so let's train on everything
# - 1 epoch is worse than 3 epochs, as is 5 epochs
# - no easy may be slightly better

# Proposed final model:
# all data 3 epochs. no easy seems too risky for 0.1% improvement

# TODO: check 500b vs 500b-no-easy to validate
#    - ftjob-s1sm2NspgZdHSbuQtg4PGSRx 500b
#    - ftjob-bSolxPJBCSVw3wkP3qj8T49Y 500b-no-easy
# check 5 epochs vs 3 epochs to validate
#    - ft:gpt-3.5-turbo-1106:personal::8rjLa89Y 5 epochs 0::2
#    - compare to ft:gpt-3.5-turbo-1106:personal::8rg4EtAG 0::2
# MAYBE TODO:
# 3-fold cross-validation for 5 epochs, no easy..?

# TODO: evaluate the following job IDs:
# ftjob-dXkS39hLYGTiOdwR88gwztiI trained on first 500
# ftjob-wABfdutTpdL7jAOlp4EzeAaW trained on first 500 (no easy examples)
# ftjob-gC6obTMnQKJQjIJJqL5ypn3b trained on 0::2 for 1 epoch
import os, sys, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from helpers import (
    direct_LISA,
    batch_completions,
    DIRECT_SYS_MSG,
    in_notebook,
    calc_wer,
)
from data_utils import TextTransform
from openai import AsyncOpenAI
from glob import glob
from helpers import save_finetuning_dset

##
if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio

    nest_asyncio.apply()

##
preds = [[
    "he had money i’d be happy to be known",
    "i had money i’d be happy to be a no",
    "i had money i’d be safe to buy new",
    "i had money i’d be safe to buy new",
    "he had money i’d be safe to be a no",
    "he had money i’d be safe to buy a new one",
    "if i had money i’d be happy to be a no",
    "he had money i’d be safe to buy a known",
    "if i had money i’d be happy to be a no",
    "i had money i’d be safe to be a no",
]]

client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)
model = "gpt-3.5-turbo-16k-0613"
# lisa_predictions = batch_completions(client,
#     [s for s in preds], DIRECT_SYS_MSG, model=model, n_jobs=5
# )
# lisa_predictions

##
def read_preds_from_dir(pred_dir, glob_pattern="/*.txt"):
    pred_txts = list(sorted(glob(pred_dir + glob_pattern)))
    each_file = []
    for file in pred_txts:
        with open(file, "r") as f:
            each_file.append(f.read())
    # split by newline
    each_file = [s.split("\n")[:-1] for s in each_file]
    the_len = len(each_file[0])
    for f in each_file:
        # 1200 competition, 600 test
        assert (len(f) == 1200) or (len(f) == 600)

    preds = [[] for _ in range(the_len)]
    for f in each_file:
        for i, s in enumerate(f):
            preds[i].append(s)
    return preds

##
# Validation ("TEST")
text_transform = TextTransform()
test_preds = read_preds_from_dir("/oak/stanford/projects/babelfish/magneto/willett/",
                            glob_pattern="testPartition_seed*.txt")
truth = []
with open("/oak/stanford/projects/babelfish/magneto/willett/testPartitionTruth.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        truth.append(l[:-1])

wers = []
for i in range(len(test_preds[0])):
    wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
    print(f"Seed {i} WER: {wer*100:.2f}%")
    wers.append(wer)
mean_wer = np.array(wers).mean()
print(f"Mean WER %2=0: {mean_wer*100:.2f}%")
##
test_predictions = batch_completions(
    client, test_preds, DIRECT_SYS_MSG, model=model, n_jobs=5
)
##
def cleanup_lisa_pred(lisa_pred):
    "match the cleanup in the competition script"
    # drop , " ? . -
    new = lisa_pred.replace(",", "")
    new = new.replace('"', "")
    new = new.replace("?", "")
    new = new.replace(".", "")
    new = new.replace("-", " ")
    # to lower case
    new = new.lower()
    # convert 1975 to nineteen seventy five
    new = new.replace("1975", "nineteen seventy five")
    # remove leading and trailing whitespace
    new = new.strip()
    return new

clean_test_preds = [cleanup_lisa_pred(l) for l in test_predictions]
# clean_wer = calc_wer(clean_test_preds, truth, text_transform)
clean_wer = calc_wer(clean_test_preds[1::2], truth[1::2], text_transform)
print(f"Test LISA WER %2=1: {clean_wer*100:.2f}%")

clean_wer = calc_wer(clean_test_preds[500:], truth[500:], text_transform)
print(f"Test LISA WER last100: {clean_wer*100:.2f}%")
##
# sorting by best to worst model
sorted_wers = np.argsort(wers)
sorted_test_preds = []
for i in range(len(test_preds)):
    sorted_test_preds.append([])
    for j in sorted_wers:
        sorted_test_preds[i].append(test_preds[i][j])
##
# FINAL MODEL
finetuned_test_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    model="ft:gpt-3.5-turbo-1106:personal::8rkR8IhW",
    n_jobs=5,
)
finetuned_test_preds = [cleanup_lisa_pred(l) for l in finetuned_test_predictions]
finetuned_wer = calc_wer(finetuned_test_preds, truth, text_transform)
print(f"Finetuned Test LISA WER (trained on all Test): {finetuned_wer*100:.2f}%")


##
sorted_test_predictions = batch_completions(
    client, sorted_test_preds, DIRECT_SYS_MSG, model=model, n_jobs=5
)
clean_sorted_test_preds = [cleanup_lisa_pred(l) for l in sorted_test_predictions]
clean_sorted_wer = calc_wer(clean_sorted_test_preds[1::2], truth[1::2], text_transform)
print(f"Sorted Test LISA WER %2=1: {clean_sorted_wer*100:.2f}%")
# 14.9%, with no fine-tuning
# ft:gpt-3.5-turbo-1106:personal::8rg4EtAG
# 13.4% (I think ;)
sorted_test_1epoch_predictions = batch_completions(
    client, sorted_test_preds, DIRECT_SYS_MSG, model="ft:gpt-3.5-turbo-1106:personal::8rigxo2U", n_jobs=5
)
clean_sorted_test_1epoch_preds = [cleanup_lisa_pred(l) for l in sorted_test_1epoch_predictions]
clean_sorted_wer = calc_wer(clean_sorted_test_1epoch_preds[1::2], truth[1::2], text_transform)
print(f"1 epoch Test LISA WER %2=1: {clean_sorted_wer*100:.2f}%")
# ft:gpt-3.5-turbo-1106:personal::8rigxo2U
# 14.55% (worse than 3 epochs!)

best_model_preds = [s[0] for s in sorted_test_preds]
best_model_wer = calc_wer(best_model_preds[1::2], truth[1::2], text_transform)
print(f"Best Model Test WER %2=1: {best_model_wer*100:.2f}%")

best_model_wer = calc_wer(best_model_preds[:500], truth[:500], text_transform)
print(f"Best Model Test WER last100: {best_model_wer*100:.2f}%")
##
lisa_100_wer = calc_wer(clean_sorted_test_preds[500:], truth[500:], text_transform)
print(f"LISA WER last100: {lisa_100_wer*100:.2f}%") # 12.5%
##
last100_no_easy_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    model="ft:gpt-3.5-turbo-1106:personal::8riY3Qp1",
    n_jobs=5,
)
clean_last100_no_easy_preds = [cleanup_lisa_pred(l) for l in last100_no_easy_predictions]
clean_sorted_wer = calc_wer(clean_last100_no_easy_preds[500:], truth[500:], text_transform)
print(f"No easy Test LISA WER Last100: {clean_sorted_wer*100:.2f}%")
# ****BEST MODEL SO FAR****
# "ft:gpt-3.5-turbo-1106:personal::8riY3Qp1" (fine-tuned on 500-no-easy)
# 11.0% (compare to 12.5% for LISA or 16.3% best model)
# 11.5% on re-run next day... 2023-02-13
##
last100_lisa_finetune_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    model="ft:gpt-3.5-turbo-1106:personal::8ricgz8C",
    n_jobs=5,
)
clean_last100_lisa_finetune_preds = [
    cleanup_lisa_pred(l) for l in last100_lisa_finetune_predictions
]
clean_sorted_wer = calc_wer(
    clean_last100_lisa_finetune_preds[500:], truth[500:], text_transform
)
print(f"finetuned Test LISA WER Last100: {clean_sorted_wer*100:.2f}%")
# ft:gpt-3.5-turbo-1106:personal::8ricgz8C (fine-tuned on 500)
# 11.1% (marginally worse than with no-easy examples)
# 11.7% on re-run next day... 2023-02-13
##
first100_lisa_finetune_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    model="ft:gpt-3.5-turbo-1106:personal::8rjhCVdL",
    n_jobs=5,
)
clean_first100_lisa_finetune_preds = [
    cleanup_lisa_pred(l) for l in first100_lisa_finetune_predictions
]
clean_sorted_wer = calc_wer(
    clean_first100_lisa_finetune_preds[:100], truth[:100], text_transform
)
print(f"finetuned Test LISA WER First100: {clean_sorted_wer*100:.2f}%")
# ft:gpt-3.5-turbo-1106:personal::8rjyPAWR (fine-tuned on 500b)
# 10.2%

# ft:gpt-3.5-turbo-1106:personal::8rjhCVdL (fine-tuned on 500b-no-easy)
# 10.35%
##
# top1_jsonl_path = save_finetuning_dset(
#     sorted_test_preds[0::2],
#     truth[0::2],
#     "../../fine_tuning_data/2024-02-12_willet-ensemble.jsonl",
# )

# # save 500 for fine-tuning
# top1_jsonl_path = save_finetuning_dset(
#     sorted_test_preds[:500],
#     truth[:500],
#     "../../fine_tuning_data/2024-02-12_willet-500.jsonl",
# )

# save 500b for fine-tuning
top1_jsonl_path = save_finetuning_dset(
    sorted_test_preds[100:],
    truth[100:],
    "../../fine_tuning_data/2024-02-12_willet-500b.jsonl",
)

# save all for fine-tuning
top1_jsonl_path = save_finetuning_dset(
    sorted_test_preds,
    truth,
    "../../fine_tuning_data/2024-02-12_willet-all.jsonl",
)


from openai import OpenAI

sync_client = OpenAI()

with open(top1_jsonl_path, "rb") as f:
    sync_client.files.create(file=f, purpose="fine-tune")
##
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    # training_file="file-fZAEewYLFQLPGg8FjhVv6mI2",
    # training_file="file-MMPa98OyGRAXO7HMMVihZbQ0",
    # training_file="file-BnJFslgX3VF1f8rg9BBuupjp", # 500b
    training_file="file-3hj6cpw49jQ7s4VKPoTUwFpn",  # all
    model="gpt-3.5-turbo-1106",
    # hyperparameters={"n_epochs": 3},
)
##
# evaluate on %2=1
finetuned_test_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    model="ft:gpt-3.5-turbo-1106:personal::8rg4EtAG",
    n_jobs=5,
)
finetuned_test_preds = [cleanup_lisa_pred(l) for l in finetuned_test_predictions]
finetuned_wer = calc_wer(finetuned_test_preds[1::2], truth[1::2], text_transform)
print(f"Sorted Test LISA WER %2=1: {finetuned_wer*100:.2f}%")
# ft:gpt-3.5-turbo-1106:personal::8rg4EtAG (fine-tuned on 0::2)
# 13.4%, better than no fine-tuning!
# ft:gpt-3.5-turbo-1106:personal::8rjLa89Y (5 epochs 0::2)
# 15.2%, yuck!

# ft:gpt-3.5-turbo-1106:personal::8n2fDmAy (Gaddy LibriSpeech fine-tuned on 270)
# 13.7%
# ft:gpt-3.5-turbo-1106:personal::8nHXtaiS (Best Gaddy fine-tuned test perf)
# worse at 14.5%
# ft:gpt-3.5-turbo-1106:personal::8rhRrDQK (0::2 where all words in truth are in preds)
# 14.22%
##
# spot check predictions
for i, (preds, lisa_pred, tru) in enumerate(zip(sorted_test_preds, finetuned_test_preds, truth)):
    print(f"=================== {i} =======================\n{tru}")
    print(lisa_pred)
    for pred in preds:
        print(pred)
##
# spot check fine-tuned dataset
with open("../../fine_tuning_data/2024-02-12_willet-handlabel.txt", "w") as f:
    for i, (preds, tru) in enumerate(zip(sorted_test_preds[::2], truth[::2])):
        f.write(f"=================== {i} =======================\n{tru}\n")
        for pred in preds:
            f.write(pred)
            f.write("\n")
# drop "impossible"
# 1, 9, 15, 20, 23, 29, 34, 38, 42, 46, 49, 60, 62, 65, 68, 75, 77, 81, 84,
# 88, 91, 97, 101, 102, 105, 106, 107,

# 87: change to "i wanted to go to that golf course down south"
##
skip_idxs = set()
# check if truth has word that wasn't in predictions
# spoiler: this does a lot worse when dropping the hard ones!
for i, (preds, tru) in enumerate(zip(sorted_test_preds, truth)):
    pred = " ".join(preds)
    for word in tru.split():
        if word not in pred:
            skip_idxs.add(i)
skip_idxs, len(skip_idxs)
N = 0
for i in skip_idxs:
    if i % 2 == 0:
        N+=1
        print(i)
# only train on even indices
for i in range(1,len(sorted_test_preds),2):
    skip_idxs.add(i)
# drop skip_idxs
new_test_preds = [p for i, p in enumerate(sorted_test_preds) if i not in skip_idxs]
new_truth = [t for i, t in enumerate(truth) if i not in skip_idxs]
print(f"fine-tune on {len(new_test_preds)}. sanity check: {300-N=}")
top1_jsonl_path = save_finetuning_dset(
    new_test_preds,
    new_truth,
    "../../fine_tuning_data/2024-02-12_willet-no-missing-words.jsonl",
)
with open(top1_jsonl_path, "rb") as f:
    sync_client.files.create(file=f, purpose="fine-tune")

##
# drop easy cases where all predictions are the same
skip_idxs = set()
for i, (preds, tru) in enumerate(zip(sorted_test_preds, truth)):
    pred = preds[0]
    if all(p == pred for p in preds):
        skip_idxs.add(i)
skip_idxs, len(skip_idxs)
N = 0
for i in skip_idxs:
    if i > 100:
        N += 1
# only train on last 500
for i in range(100):
    skip_idxs.add(i)
# drop skip_idxs
new_test_preds = [p for i, p in enumerate(sorted_test_preds) if i not in skip_idxs]
new_truth = [t for i, t in enumerate(truth) if i not in skip_idxs]
print(f"fine-tune on {len(new_test_preds)}. sanity check: {500-N=}")
top1_jsonl_path = save_finetuning_dset(
    new_test_preds,
    new_truth,
    "../../fine_tuning_data/2024-02-12_willet_500b-no-easy.jsonl",
)
##
with open(top1_jsonl_path, "rb") as f:
    sync_client.files.create(file=f, purpose="fine-tune")
##
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    # training_file="file-E3bgAm7lbGdW2ocoPratCEvW",
    # training_file="file-iRu6dTUaCFmN0LmyiJSViHZh", # 500-no-easy
    training_file="file-Haen3M0DJelDa0Cq7zgOmHeV",  # 500b-no-easy
    model="gpt-3.5-turbo-1106",
)

##
# COMPETITION
preds = read_preds_from_dir(
    "/oak/stanford/projects/babelfish/magneto/willett/",
    glob_pattern="/5gramLLMCompetitionSubmission*.txt",
)
preds[1]

##
lisa_predictions = batch_completions(
    client, preds, DIRECT_SYS_MSG, model=model, n_jobs=5
)

##
# save to file
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/direct_lisa_top10ensemble_competition.txt", "w"
) as f:
    for l in lisa_predictions:
        # drop , " ? . -
        new = l.replace(",", "")
        new = new.replace('"', "")
        new = new.replace("?", "")
        new = new.replace(".", "")
        new = new.replace("-", " ")
        # to lower case
        new = new.lower()
        # convert 1975 to nineteen seventy five
        new = new.replace("1975", "nineteen seventy five")
        # remove leading and trailing whitespace
        new = new.strip()
        f.write(new + "\n")

##
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/direct_lisa_top10ensemble_competition.txt", "r"
) as f:
    lines = f.readlines()
    lines = [l[:-1] for l in lines]
    unique_chars = list(set("".join(lines)))
    print(np.array(unique_chars))
    print(len(np.array(unique_chars)))

##
import numpy as np
unique_chars = list(set("".join(lisa_predictions)))
np.array(unique_chars)
##

#!/bin/bash

module load AI/pytorch_23.02-1.13.1-py3

conda activate new_env

python3 20241204word_cls_model.py  --output_directory "./models/cls_model_1205/"
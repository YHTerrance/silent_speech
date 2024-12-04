import random

import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoderLayer

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("model_size", 768, "number of hidden dimensions")
flags.DEFINE_integer("num_layers", 6, "number of layers")
flags.DEFINE_float("dropout", 0.2, "dropout")


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class EEGWordClsModel(nn.Module):
    def __init__(self, num_features, num_outs):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(num_features, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=FLAGS.model_size,
            nhead=8,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=3072,
            dropout=FLAGS.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, FLAGS.num_layers)

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.flatten = nn.Flatten()

        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, FLAGS.model_size))

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        # TODO: figure out wtf this is
        # put channel before time for conv B x C x T
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)  # transpose back to B x T x C
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        # add cls token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # put time first because transformers expect input int the shape (sequence length, batch size, feature dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        cls_output = x[:, 0, :]

        # x = x.transpose(1, 2)  # B x C x T
        # x = self.avgpool(x)  # B x C x 1
        # x = x.transpose(1, 2)  # B x 1 x C
        # x = self.flatten(x)  # (B, 1, C) -> (B, C)

        return self.w_out(cls_output)


class EEGSeqtoSeqModel(nn.Module):
    def __init__(self, num_features, num_outs):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(num_features, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            # ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=FLAGS.model_size,
            nhead=8,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=3072,
            dropout=FLAGS.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, FLAGS.num_layers)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)
        # put channel before time for conv B x C x T
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)  # transpose back to B x T x C
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        # put time first because transformers expect input int the shape (sequence length, batch size, feature dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        return self.w_out(x)

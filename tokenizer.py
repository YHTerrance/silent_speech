import torch
from transformers import AutoTokenizer
from typing import Literal, List, Optional


class CharTokenizer():
    ''' A wrapper around character tokenization to have a consistent interface with other tokeization strategies'''

    def __init__(self):
        # Define special tokens for end-of-sequence, padding, and unknown characters
        self.eos_token = "<|endoftext|>"  # Same as EOS_TOKEN
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

        # Initialize vocabulary with uppercase alphabet characters and space
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,  # Same ID as EOS_TOKEN
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        # Create an inverse mapping from IDs to characters for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Define token IDs for special tokens for easy access
        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, data: str) -> List[str]:
        # Split input data into a list of characters for tokenization
        return [char for char in data]

    def encode(self, data: str, return_tensors: Optional[Literal['pt']] = None) -> List[int]:
        # Encode each character in data to its integer ID, using unk_token if character is not in vocab
        e = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        # If specified, convert to PyTorch tensor format
        if return_tensors == 'pt':
            return torch.tensor(e).unsqueeze(0)
        return e

    def decode(self, data: List[int]) -> str:
        # Decode list of token IDs back to string by mapping each ID to its character
        try:
            return ''.join([self.inv_vocab.get(j) for j in data])
        except:
            # Handle decoding error by converting data to list, if it's a tensor
            data = data.cpu().tolist()
            return ''.join([self.inv_vocab.get(j) for j in data])


class GTokenizer:

    def __init__(self, token_type: Literal['1k', '10k', '50k', 'char'] = 'char', logger=None):

        self.token_type = token_type
        self.vocab, self.inv_vocab = None, None
        if token_type == '1k':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_1k", use_fast=False)
        elif token_type == '10k':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_10k", use_fast=False)
        elif token_type == '20k':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_20k", use_fast=False)
        elif token_type == '50k':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_50k", use_fast=False)
        elif token_type == '100k':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "alexgichamba/hw4_tokenizer_100k", use_fast=False)
        elif token_type == 'char':
            self.tokenizer = CharTokenizer()

        self.EOS_TOKEN = self.tokenizer.eos_token_id
        self.SOS_TOKEN = self.tokenizer.bos_token_id
        self.PAD_TOKEN = self.tokenizer.convert_tokens_to_ids(
            '<|padding|>') if self.token_type != "char" else self.tokenizer.pad_token_id
        self.UNK_TOKEN = self.tokenizer.unk_token_id
        self.VOCAB_SIZE = self.tokenizer.vocab_size

        # Test tokenization methods to ensure everything is working correctly
        test_text = "HI DEEP LEARNERS"
        test_tok = self.tokenize(test_text)
        test_enc = self.encode(test_text)
        test_dec = self.decode(test_enc)

        print(f"[Tokenizer Loaded]: {token_type}")
        print(f"\tEOS_TOKEN:  {self.EOS_TOKEN}")
        print(f"\tSOS_TOKEN:  {self.SOS_TOKEN}")
        print(f"\tPAD_TOKEN:  {self.PAD_TOKEN}")
        print(f"\tUNK_TOKEN:  {self.UNK_TOKEN}")
        print(f"\tVOCAB_SIZE: {self.VOCAB_SIZE}")
        print("Examples:")
        print(
            f"\t[DECODE EOS, SOS, PAD, UNK] : {self.decode([self.EOS_TOKEN, self.SOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN])}")
        print(f"\t[Tokenize HI DEEP LEARNERS] : {test_tok}")
        print(f"\t[Encode   HI DEEP LEARNERS] : {test_enc}")
        print(f"\t[Decode   HI DEEP LEARNERS] : {test_dec}")

    def tokenize(self, data: str) -> List[str]:
        return self.tokenizer.tokenize(data)

    def encode(self, data: str, return_tensors=False) -> List[int]:
        if return_tensors:
            return self.tokenizer.encode(data, return_tensors='pt')
        return self.tokenizer.encode(data)

    def decode(self, data: List[int]) -> str:
        return self.tokenizer.decode(data)

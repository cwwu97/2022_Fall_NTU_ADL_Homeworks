from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
'''
[
    {
    "text": "how long should i cook steak for",
    "intent": "cook_time",
    "id": "eval-0"
    }
]

[
  {
    "tokens": ["i","prefer","a","table","outdoors"],
    "tags": ["O","O","O","O","O"],
    "id": "eval-0"
  }
]
'''

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        test_mode:bool=False
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        texts = list(map(lambda x: x['text'].split(), samples))
        texts = self.vocab.encode_batch(batch_tokens=texts, to_len=self.max_len)
        texts = torch.tensor(texts, dtype=torch.long)
        if not self.test_mode:
            intents = list(map(lambda x: self.label_mapping[x['intent']], samples))
            intents = torch.tensor(intents, dtype=torch.long)
            return texts, intents
        return texts

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    # ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn 
        tokens = list(map(lambda x: x['tokens'], samples))
        tokens = self.vocab.encode_batch(batch_tokens=tokens, to_len=self.max_len)
        tokens = torch.LongTensor(tokens)
        if not self.test_mode:
            tags = list(map(lambda x: x['tags'], samples))
            for idx, tag in enumerate(tags):
                tag = list(map(lambda x: self.label_mapping[x], tag))
                tags[idx] = tag
            # Pad to max_len
            tags = pad_to_len(seqs=tags, to_len=self.max_len, padding=self.label_mapping["[PAD]"])
            tags = torch.tensor(tags, dtype=torch.long)
            return tokens, tags
        return tokens
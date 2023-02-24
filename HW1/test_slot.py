import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = Path(args.cache_dir / "tag2idx.json")
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # Load test data
    test_data_path = Path(args.test_file)
    test_data = json.loads(test_data_path.read_text())
    test_dataset = SeqTaggingClsDataset(test_data, vocab, label_mapping=tag2idx, max_len=args.max_len, test_mode=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    embeddings = torch.load(Path(args.cache_dir / "embeddings.pt"))

    # init and load model
    model = SeqTagger(
        embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=test_dataset.num_classes
        )

    # load weights into model
    ckpt = torch.load(Path(args.ckpt_path))
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()

    # TODO: predict dataset
    total_preds = []
    for idx, tokens in enumerate(tqdm(test_dataloader)):
        # get x
        tokens = tokens.to(DEVICE, dtype=torch.long)
        # model predict
        preds = model(tokens)
        # aggregate evaluation prediction
        preds = torch.argmax(preds, dim=2)
        total_preds.extend(preds.detach().cpu().numpy())
    total_preds_label = []
    for idx in range(len(total_preds)):
        pad_idx = len(test_dataset[idx]['tokens'])
        total_preds_label.append(' '.join([test_dataset.idx2label(pred) for pred in total_preds[idx][:pad_idx]]))
    result = pd.DataFrame({'id':list(map(lambda x: x['id'], test_data)), 'tags':total_preds_label})
    result.to_csv(args.pred_file, index=False)
    print(f'Testing result saved at {args.pred_file}.')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./slot_best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="slot.pred.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
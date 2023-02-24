import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
from tqdm import tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # data = json.loads(args.test_file.read_text())
    # dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, test_mode=True)

    # TODO: crecate DataLoader for test dataset
    test_data_path = Path(args.test_file)
    test_data = json.loads(test_data_path.read_text())
    test_dataset = SeqClsDataset(test_data, vocab, label_mapping=intent2idx, max_len=args.max_len, test_mode=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        test_dataset.num_classes
    )

    # load weights into model
    ckpt = torch.load(Path(args.ckpt_path))
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()

    # TODO: predict dataset
    total_preds = []
    for idx, texts in enumerate(tqdm(test_dataloader)):
        # get x and y
        texts = texts.to(DEVICE, dtype=torch.long)

        # model predict
        preds = model(texts)

        # aggregate evaluation prediction
        preds = torch.argmax(preds, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())

    total_preds_label = [test_dataset.idx2label(pred) for pred in total_preds]

    # TODO: write prediction to file (args.pred_file)
    result = pd.DataFrame({'id':list(map(lambda x: x['id'], test_data)), 'intent':total_preds_label})
    result.to_csv(args.pred_file, index=False)
    print(f'Testing result saved at {args.pred_file}.')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
        default='./data/intent/test.json'
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default='./intent_best.pt'
    )
    parser.add_argument("--pred_file", type=Path, default="intent.pred.csv")

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
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

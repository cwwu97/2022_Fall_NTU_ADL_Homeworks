import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss
# from seqeval.metrics import classification_report
# from seqeval.scheme import IOB2

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def main(args):
    # TODO: implement main function
    print('CUDA:', torch.cuda.is_available())
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # Load intent to index mapping dict
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # Load train data
    train_data_path = Path(args.data_dir / f'{TRAIN}.json')
    train_data = json.loads(train_data_path.read_text())
    train_dataset = SeqTaggingClsDataset(train_data, vocab, label_mapping=tag2idx, max_len=args.max_len, test_mode=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    # Load eval data
    eval_data_path = Path(args.data_dir / f'{DEV}.json')
    eval_data = json.loads(eval_data_path.read_text())
    eval_dataset = SeqTaggingClsDataset(eval_data, vocab, label_mapping=tag2idx, max_len=args.max_len, test_mode=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
    
    # Load word embedding
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(tag2idx))
    model.to(DEVICE)

    # TODO: init optimizer
    for param in model.parameters():
        param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr = args.lr)
    cross_entropy = CrossEntropyLoss()

    num_epoch = args.num_epoch
    best_acc = 0
    for epoch in range(num_epoch):

        # TODO: Training loop - iterate over train dataloader and update model weights
        print(f'Epoch {epoch+1}/{num_epoch} | Training and Evaluating...')
        model.train()
        sum_train_loss = 0
        for idx, (tokens, tags) in enumerate(tqdm(train_dataloader)):
            # get x and y
            tokens = tokens.to(DEVICE, dtype=torch.long) #[batch_size, token]
            tags = tags.to(DEVICE, dtype=torch.long)
            # reset gradient
            optimizer.zero_grad()
            # model predict
            preds = model(tokens)
            # calculate cross entropy loss
            loss = cross_entropy(preds.view(-1, preds.shape[-1]), tags.view(-1))
            # calculate gradient
            loss.backward()
            # update weights by back propagate
            optimizer.step()
            # evaluate
            sum_train_loss += (loss.item()/len(train_dataloader))

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        total_preds = []
        total_tags = []
        sum_eval_loss, avg_eval_acc = 0, 0
        with torch.no_grad():
            for idx, (tokens, tags) in enumerate(tqdm(eval_dataloader)):
                # get x and y
                tokens = tokens.to(DEVICE, dtype=torch.long)
                tags = tags.to(DEVICE, dtype=torch.long)
                # model predict
                preds = model(tokens)
                # calculate cross entropy loss
                loss = cross_entropy(preds.view(-1, preds.shape[-1]), tags.view(-1))
                # aggregate evaluation prediction
                preds = torch.argmax(preds, dim=2)
                total_preds.extend(preds.detach().cpu().numpy())
                total_tags.extend(tags.detach().cpu().numpy())
                # evaluate
                sum_eval_loss += (loss.item()/len(eval_dataloader))

        eval_correct_cnt, eval_tag_cnt = 0, 0  # For token accuracy
        total_preds_list, total_correct_list = [], []  # For classification report
        total_preds_str, total_correct_str = [], []  # For joint accuracy

        for idx in range(len(total_preds)):
            pad_index = np.where(total_tags[idx] == 0)[0][0] # Index where padding starts at
            eval_correct_cnt += (total_tags[idx][:pad_index] == total_preds[idx][:pad_index]).sum().item()
            eval_tag_cnt += len(total_tags[idx][:pad_index])

            total_preds_list.append(list(map(lambda x: eval_dataset.idx2label(x), total_preds[idx][:pad_index])))
            total_correct_list.append(list(map(lambda x: eval_dataset.idx2label(x), total_tags[idx][:pad_index])))

            total_preds_str.append(' '.join(list(map(lambda x: eval_dataset.idx2label(x), total_preds[idx][:pad_index]))))
            total_correct_str.append(' '.join(list(map(lambda x: eval_dataset.idx2label(x), total_tags[idx][:pad_index]))))

        # print(classification_report(total_preds_list, total_correct_list, mode='strict', scheme=IOB2))

        eval_joint_acc = (np.array(total_preds_str) == np.array(total_correct_str)).sum() / len(eval_data)
        eval_token_acc = eval_correct_cnt/eval_tag_cnt

        if eval_joint_acc > best_acc:
            best_acc = eval_joint_acc
            torch.save(model.state_dict(), Path(args.ckpt_dir / 'slot_best.pt'))

        print(f'Training loss: {sum_train_loss:.5f}')
        print(f'Evaluating loss: {sum_eval_loss:.5f} | token acc: {eval_token_acc:.5f} | joint acc: {eval_joint_acc:.5f}')
        print('\n')
        
    # print(classification_report(total_preds_list, total_correct_list, mode='strict', scheme=IOB2))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=40)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
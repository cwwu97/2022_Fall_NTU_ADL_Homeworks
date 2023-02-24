from typing import Dict
import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:

        super(SeqClassifier, self).__init__()
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.lstm = torch.nn.LSTM(input_size=embeddings.size(1), hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.gru = torch.nn.GRU(input_size=embeddings.size(1), hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.encoder_output_size, self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return self.hidden_size * 2
        else:
            return self.hidden_size

    def forward(self, batch):
        # TODO: implement model forward
        x = self.embeddings(batch)  # [batch_size, SeqClsDataset.max_len, embedding_dimension]
        # x, hn = self.gru(x, None)
        x, (hn, cn) = self.lstm(x, None) # [batch_size, SeqClsDataset.max_len, encoder_output_size]
        x = torch.mean(x, dim = 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SeqTagger(SeqClassifier):
    def forward(self, batch):
        # TODO: implement model forward
        x = self.embeddings(batch)
        x = self.dropout(x)
        x, hn = self.gru(x, None)
        x = self.dropout(x)
        x = self.fc(x)
        return x
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


class ClipEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(ClipEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        self.embedding_dim = self.model.config.dim

    def forward(self, batch):
        with torch.no_grad():
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)

            z_text = outputs.last_hidden_state[:, 0, :]

        return z_text
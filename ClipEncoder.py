import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


class ClipEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(ClipEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DistilBERT is significantly lighter than CLIP or BERT-base
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)

        # 1. Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Set to eval mode to disable dropout/norm updates
        self.model.eval()

        # DistilBERT hidden size is 768 (different from CLIP's 512)
        self.embedding_dim = self.model.config.dim

    def forward(self, batch):
        # Ensure no gradients are tracked during inference
        with torch.no_grad():
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)

            # Use the [CLS] token representation (index 0) as the global sentence embedding
            # This is the lightweight version of 'pooler_output'
            z_text = outputs.last_hidden_state[:, 0, :]

        return z_text
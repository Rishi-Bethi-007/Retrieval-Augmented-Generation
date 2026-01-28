import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

class Embedder:
    """
    Responsible for:
    - loading embedding model
    - converting text -> normalized vectors
    """

    def __init__(
            self,
            model_name: str= 'sentence-transformers/all-MiniLM-L6-v2',
    device: str | None = None,
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Convert a list of texts into normalized embeddings.

        Returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        #Tokenize input texts
        inputs = self.tokenizer(
            texts,
            padding= True,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        #Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        ## Token level embeddings
        token_embeddings= outputs.last_hidden_state  # [batch_size, No.of tokens in that chunk, hidden_dim]

        #Attention mask to ignore padding tokens
        attention_mask= inputs['attention_mask'].unsqueeze(-1).float()  # [batch_size, No.of tokens in that chunk]
        masked_token_embeddings= token_embeddings * attention_mask  # [batch_size, No.of tokens in that chunk, hidden_dim]

        #Mean Pooling - Take average of token embeddings
        sum_embeddings= torch.sum(masked_token_embeddings, dim=1)  # [batch_size, hidden_dim]
        count= torch.clamp(attention_mask.sum(dim=1), min=1e-9)  # [batch_size, 1]
        embeddings= sum_embeddings / count  # [batch_size, hidden_dim]

        #Normalize embeddings to unit length
        normalized_embeddings= F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings.cpu()
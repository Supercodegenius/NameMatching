import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NamePairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc_a = self.tokenizer(
            row["name_a"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            row["name_b"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(),
            "attention_mask_a": enc_a["attention_mask"].squeeze(),
            "input_ids_b": enc_b["input_ids"].squeeze(),
            "attention_mask_b": enc_b["attention_mask"].squeeze(),
            "label": torch.tensor(row["label"], dtype=torch.float),
        }


class BiEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)

    def forward(self, batch):
        emb_a = self.encode(batch["input_ids_a"], batch["attention_mask_a"])
        emb_b = self.encode(batch["input_ids_b"], batch["attention_mask_b"])
        return emb_a, emb_b


def contrastive_loss(emb_a, emb_b, labels, margin=0.5):
    cosine_sim = F.cosine_similarity(emb_a, emb_b)
    pos_loss = labels * (1 - cosine_sim)
    neg_loss = (1 - labels) * torch.clamp(cosine_sim - margin, min=0)
    return (pos_loss + neg_loss).mean()


def main():
    df = pd.read_csv("data/train.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = NamePairDataset(train_df, tokenizer)
    val_ds = NamePairDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = BiEncoder(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            emb_a, emb_b = model(batch)
            loss = contrastive_loss(emb_a, emb_b, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")

    model.encoder.save_pretrained("./outputs/biencoder")
    tokenizer.save_pretrained("./outputs/biencoder")
    print("Model saved to ./outputs/biencoder")


if __name__ == "__main__":
    main()

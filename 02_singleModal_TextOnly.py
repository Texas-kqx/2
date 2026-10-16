import os
import json

import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ======================
# 参数
# ======================

root_dir = r"datasets"

train_json=os.path.join(root_dir, "annotations_train.json")
val_json=os.path.join(root_dir, "annotations_val.json")

test_json=os.path.join(root_dir, "annotations_test.json")

model_name="bert-base-uncased"
num_classes=100
batch_size=16
epochs=10
lr=2e-5

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ======================
# Dataset
# ======================

class TextBirdDataset(Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        max_length=64
    ):

        with open(json_file, "r", encoding="utf-8") as f:
            data=json.load(f)

        self.samples=data["annotations"]
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        item=self.samples[index]

        # 一个图片对应10条caption
        captions=item["captions"]

        # 随机选择一条描述训练
        text=captions[
            torch.randint(0, len(captions), (1,)).item()]

        encoding=self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),
            "attention_mask":
                encoding["attention_mask"].squeeze(0),
            "label":
                torch.tensor(
                    item["label"],
                    dtype=torch.long
                )
        }

# ======================
# BERT分类模型
# ======================
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=AutoModel.from_pretrained(
            model_name
        )
        hidden_size=self.bert.config.hidden_size
        self.classifier=nn.Linear(
            hidden_size,
            num_classes
        )
    def forward(self, input_ids, attention_mask):
        output=self.bert( input_ids=input_ids, attention_mask=attention_mask)

        hidden=output.last_hidden_state
        # mask mean pooling
        mask=attention_mask.unsqueeze(-1)
        hidden=hidden*mask
        sentence_embedding=(hidden.sum(dim=1) / mask.sum(dim=1))

        logits=self.classifier(sentence_embedding)
        return logits

# ======================
# DataLoader
# ======================

tokenizer=AutoTokenizer.from_pretrained(model_name)

train_set=TextBirdDataset(train_json, tokenizer)
val_set=TextBirdDataset(val_json, tokenizer)
test_set=TextBirdDataset(test_json, tokenizer)

train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(len(train_set), len(test_set))

# ======================
# train
# ======================

model=BertClassifier()

model=model.to(device)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
print("start training...")
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")
    model.train()
    total_loss=0
    correct = 0
    total = 0
    for batch in train_loader:

        ids=batch["input_ids"].to(device)
        mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)
        optimizer.zero_grad()

        logits=model(ids, mask)

        loss=criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

        preds=torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    train_acc = correct / total
    print("epoch:", epoch+1, "loss:", total_loss/len(train_loader), f"Train Accuracy: {train_acc:.4f}")

# ======================
# test
# ======================
model.eval()
y_true=[]
y_pred=[]

with torch.no_grad():
    for batch in test_loader:
        ids=batch["input_ids"].to(device)
        mask=batch["attention_mask"].to(device)
        labels=batch["label"]

        logits=model(ids, mask)

        preds=torch.argmax(logits, dim=1).cpu()

        y_pred.extend(preds.numpy())

        y_true.extend(labels.numpy())


accuracy=accuracy_score(y_true, y_pred)

precision=precision_score(y_true, y_pred, average="macro", zero_division=0)

recall=recall_score(y_true, y_pred, average="macro", zero_division=0)

f1=f1_score( y_true, y_pred, average="macro", zero_division=0)

print("====================")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"宏平均精确率 (Precision): {precision:.4f}")
print(f"宏平均召回率 (Recall): {recall:.4f}")
print(f"宏平均 F1 分数 (F1-Score): {f1:.4f}")
print("====================")
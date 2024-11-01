import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data as utils
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_confusion_matrix,
)
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sn


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculate Accuracy of predictions.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    correct = 0
    if y_pred.ndim == 2:
        y_pred = torch.argmax(y_pred, dim=1)

    correct += (y_pred == y_true).float().sum()
    return correct


def f1_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


if False:
    from sqlalchemy import create_engine

    engine = create_engine("mysql+pymysql://root:1234@localhost/ares_local")
    conn = engine.connect()
    df = pd.read_sql(
        """SELECT e.descricao, c.id as saida
    FROM ares_local.evento e
    INNER JOIN ares_local.classe c ON c.id = e.classe_id""",
        conn,
    )


df = pd.read_csv("DB.csv")
df = df.dropna()
df = df[df["descricao"].str.contains("Vistos.*Int")]
# Salvar um arquivo no disco
# df.to_csv("DB.csv", index=False)
size_samples = 10000
df = df.sample(size_samples, random_state=42)

print(df.shape)

y = torch.tensor(df["saida"].to_numpy())
bertikal = BertTokenizer.from_pretrained("BERTikal/")
encoded = bertikal(
    df["descricao"].astype(str).to_list(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
X = nn.utils.rnn.pad_sequence(encoded["input_ids"], batch_first=True, padding_value=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30000, 14, padding_idx=0, scale_grad_by_freq=True)
        self.gru = nn.LSTM(
            14, 64, batch_first=True, bidirectional=True, dropout=0.1, num_layers=2
        )
        self.softmax = nn.Sequential(nn.Linear(512 * 128, 28), nn.LogSoftmax(dim=1))

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.gru(emb)
        softmax = self.softmax(torch.flatten(output, start_dim=1))
        return softmax


train_dataset = utils.TensorDataset(X_train, y_train.cuda())

train_dataloader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = utils.TensorDataset(X_test, y_test.cuda())

test_dataloader = utils.DataLoader(test_dataset, batch_size=16)

model = Model().cuda()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
f1_score = []
epoch = 100
for epoch in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        preds = model(inputs.cuda())
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # print statistic16
        running_loss += loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
    model.eval()
    f1_score.append(
        multiclass_f1_score(
            model(X_test.cuda()), y_test.cuda(), average="weighted", num_classes=28
        )
    )
    model.train()
print("Finished Training")

with torch.no_grad():
    model.eval()
    fig, ax = plt.subplots()
    plt.ylabel("F1 Score (Ponderado)")
    plt.xlabel("Época")
    ax.plot(
        [(i + 1) for i in range(len(f1_score))],
        [score.cpu() for score in f1_score],
    )
    plt.savefig(f"f1_score_{size_samples}.png")
    plt.figure(figsize=(20, 10))
    sn.heatmap(
        multiclass_confusion_matrix(
            model(X_test.cuda()).cpu(), y_test.cpu(), num_classes=28
        ),
        vmin=1,
        annot=True,
        fmt=".3g",
        cbar=False,
        linewidths=0.5,
    ).set(xlabel="Predição", ylabel="Valor real")
    plt.savefig(f"confusion_mat_{size_samples}.png")

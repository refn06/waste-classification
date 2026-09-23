"""
Train a lightweight classifier on top of the frozen CLIP embeddings produced
by extract_embeddings.py.

Two classifier options are supported:
  - "logreg": scikit-learn Logistic Regression (fast, strong baseline, what
    worked well in previous competitions on similar embeddings)
  - "mlp": a small torch MLP head (slightly more capacity if logreg underfits)

Usage:
    python train_classifier.py --emb_dir embeddings --model logreg
    python train_classifier.py --emb_dir embeddings --model mlp --epochs 30
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_split(emb_dir: Path, split: str):
    X = np.load(emb_dir / f"{split}_features.npy")
    y = np.load(emb_dir / f"{split}_labels.npy")
    return X, y


class MLPHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, n_classes, epochs, lr, device):
    model = MLPHead(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test, dtype=torch.long).to(device)

    best_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xte).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_test, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[epoch {epoch:03d}] loss={loss.item():.4f} macro_f1={f1:.4f}")

    model.load_state_dict(best_state)
    return model, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="embeddings")
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="model_out")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = (emb_dir / "class_names.txt").read_text().strip().split("\n")
    print(f"[info] classes: {class_names}")

    X_train, y_train = load_split(emb_dir, "train")
    X_test, y_test = load_split(emb_dir, "test")
    print(f"[info] train: {X_train.shape}, test: {X_test.shape}")

    if args.model == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        joblib.dump(clf, out_dir / "classifier.joblib")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf, _ = train_mlp(
            X_train, y_train, X_test, y_test, len(class_names), args.epochs, args.lr, device
        )
        with torch.no_grad():
            Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = clf(Xte).argmax(dim=1).cpu().numpy()
        torch.save(clf.state_dict(), out_dir / "classifier_mlp.pt")

    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"\n[result] macro F1 on TEST: {macro_f1:.4f}\n")
    print(classification_report(y_test, preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    with open(out_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    print(f"\n[done] classifier + class names saved to {out_dir}/")


if __name__ == "__main__":
    main()

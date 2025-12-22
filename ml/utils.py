import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=2):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    report = classification_report(targets, preds, output_dict=True)
    cm = confusion_matrix(targets, preds)

    # AUROC for binary
    auroc = None
    if num_classes == 2:
        try:
            auroc = float(roc_auc_score(targets, probs[:, 1]))
        except Exception:
            auroc = None

    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "auroc": auroc,
    }

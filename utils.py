import matplotlib.pyplot as plt
import numpy as np
import cv2, textwrap, torch

def show_samples(df, img_dir):
    """
    Display images in one row with Question & Answer shown BELOW each image.
    df must contain: img_name, question, answer
    """
    n = len(df)
    plt.figure(figsize=(4*n, 7))  # height increased for Q&A text

    for i, row in enumerate(df.itertuples(), 1):
        img_path = f"{img_dir}/{row.img_name}"

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create subplot
        ax = plt.subplot(1, n, i)
        ax.imshow(img)
        ax.axis("off")

        plt.title(
            f"Sample {i}", 
            fontsize=14,
            loc='center'
        )
        
        # Wrap text nicely (adjust width depending on image size)
        q_text = textwrap.fill(f"Q: {row.question}", width=40)
        a_text = textwrap.fill(f"A: {row.answer}", width=40)

        # Add text BELOW the image using .text()
        ax.text(
            0.5, -0.15, f"{q_text}\n{a_text}",
            ha="center", va="top",
            fontsize=12,
            transform=ax.transAxes
        )

    plt.tight_layout()
    plt.show()


def show_test_samples(df, img_dir, pred_answers, num_samples=5):
    """
    Display test images with Question, Ground Truth, and Predicted Answer.
    df must contain: img_name, question, answer
    pred_answers: list or array of predicted answer strings (same order as df)
    """
    
    df = df.reset_index(drop=True)

    # sample indexes first so predictions align
    idxs = np.random.choice(len(df), size=min(num_samples, len(df)), replace=False)
    
    plt.figure(figsize=(4 * len(idxs), 7))

    for i, idx in enumerate(idxs):
        row = df.iloc[idx]
        pred = pred_answers[idx]

        img_path = f"{img_dir}/{row.img_name}"

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Subplot
        ax = plt.subplot(1, len(idxs), i + 1)
        ax.imshow(img)
        ax.axis("off")
        plt.title(f"Sample {i+1}", fontsize=14)

        # Wrapped text
        q_text = textwrap.fill(f"Q: {row.question}", width=40)
        gt_text = textwrap.fill(f"GT: {row.answer}", width=40)
        pred_text = textwrap.fill(f"Pred: {pred}", width=40)

        # Annotate
        ax.text(
            0.5, -0.18, f"{q_text}\n{gt_text}\n{pred_text}",
            ha="center", va="top", fontsize=11, transform=ax.transAxes
        )

    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses, train_accies, val_accies):
    """
    Plot training & validation loss and accuracy side by side.
    
    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        train_accies (list): Training accuracy per epoch
        val_accies (list): Validation accuracy per epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # --- Loss Plot ---
    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Validation Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # --- Accuracy Plot ---
    axes[1].plot(train_accies, label='Train Accuracy', marker='o')
    axes[1].plot(val_accies, label='Validation Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch and return average loss and accuracy.
    """
    model.train()
    obj = {
        "total_loss" : 0,
        "total_correct" : 0,
        "total_samples" : 0,
        "avg_loss": 0,
        "accuracy": 0,
        "preds" : []
    }

    for imgs, questions, answers in loader:
        imgs, questions, answers = imgs.to(device), questions.to(device), answers.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, questions)
        loss = criterion(outputs, answers)
        batch_preds = outputs.argmax(dim=1)
        loss.backward()
        optimizer.step()

        obj["total_loss"] += loss.item() * answers.size(0)
        obj["preds"].extend(batch_preds.cpu().numpy())
        obj["total_correct"] += (batch_preds == answers).sum().item()
        obj["total_samples"] += answers.size(0)

    obj["avg_loss"] = obj["total_loss"] / obj["total_samples"]
    obj["accuracy"] = obj["total_correct"] / obj["total_samples"]
    return obj


def evaluate_epoch(model, loader, criterion, device):
    """
    Evaluate model on a dataset and return loss and accuracy.
    """
    model.eval()
    obj = {
        "total_loss" : 0,
        "total_correct" : 0,
        "total_samples" : 0,
        "avg_loss": 0,
        "accuracy": 0,
        "preds" : []
    }

    with torch.no_grad():
        for imgs, questions, answers in loader:
            imgs, questions, answers = imgs.to(device), questions.to(device), answers.to(device)
            outputs = model(imgs, questions)
            loss = criterion(outputs, answers)
            batch_preds = outputs.argmax(dim=1)

            obj["total_loss"] += loss.item() * answers.size(0)
            obj["preds"].extend(batch_preds.cpu().numpy())
            obj["total_correct"] += (batch_preds == answers).sum().item()
            obj["total_samples"] += answers.size(0)

    obj["avg_loss"] = obj["total_loss"] / obj["total_samples"]
    obj["accuracy"] = obj["total_correct"] / obj["total_samples"]
    return obj



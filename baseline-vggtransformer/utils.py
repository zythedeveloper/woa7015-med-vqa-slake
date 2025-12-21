from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from dataset import SLAKEDataset
from model import VGG19TransformerVQA
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch, os, json, random, cv2, textwrap


def show_samples(df, img_dir):
    """
    Display images in one row with Question & Answer shown BELOW each image.
    df must contain: img_name, question, answer
    """
    n = len(df)
    plt.figure(figsize=(4*n, 7))  # height increased for Q&A text

    for i, row in enumerate(df.itertuples(), 1):
        img_path = os.path.join(img_dir, row.img_name)

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


def plot_graph(train_losses, val_losses, train_accs, val_accs, num_epochs=50):
    epochs_range = range(1, num_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_samples(
    results,
    dataset_dir,
    num_samples=5,
    only_correct=None,   # None = all, True = correct only, False = incorrect only
    shuffle=True,
    figsize=(15, 5)
):
    # Optional filtering
    if only_correct is not None:
        results = [r for r in results if r["correct"] == only_correct]

    if len(results) == 0:
        print("No samples to display after filtering.")
        return

    # Sampling
    if shuffle:
        samples = random.sample(results, min(num_samples, len(results)))
    else:
        samples = results[:num_samples]

    # Plot
    fig, axes = plt.subplots(1, len(samples), figsize=figsize)

    if len(samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        image_path = os.path.join(dataset_dir, 'Slake1.0', 'imgs', sample["image_path"])
        img = Image.open(image_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        title = (
            f"Q: {sample['question']}\n"
            f"GT: {sample['true_answer']}\n"
            f"Pred: {sample['predicted_answer']}\n"
            f"Conf: {sample['confidence']:.2f}\n"
            f"{'✅ Correct' if sample['correct'] else '❌ Wrong'}"
        )

        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()


def build_answer_vocab(df):
    answers = sorted(df["answer"].dropna().unique())

    ans_to_idx = {ans: idx for idx, ans in enumerate(answers)}
    idx_to_ans = {idx: ans for ans, idx in ans_to_idx.items()}

    return ans_to_idx, idx_to_ans


def train_model(dataset_dir, train_df, validate_df):
    # Hyperparameters
    batch_size = 32
    num_epochs = 25
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build answer vocabulary
    ans_to_idx, idx_to_ans = build_answer_vocab(train_df)
    
    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = SLAKEDataset(
        df=train_df,
        img_dir=dataset_dir,
        tokenizer=tokenizer,
        ans_to_idx=ans_to_idx
    )
    
    val_dataset = SLAKEDataset(
        df=validate_df,
        img_dir=dataset_dir,
        tokenizer=tokenizer,
        ans_to_idx=ans_to_idx
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = VGG19TransformerVQA(
        vocab_size=tokenizer.vocab_size,
        ans_vocab_size=len(ans_to_idx),
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    for epoch in range(num_epochs):
        batch_iterator = tqdm(train_loader, leave=True, desc=f'Processing epoch {epoch+1:02d}')
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in batch_iterator:
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            answers = batch['answer'].to(device)
            
            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += answers.size(0)
            train_correct += predicted.eq(answers).sum().item()

            batch_acc = predicted.eq(answers).sum().item() / answers.size(0)
            batch_iterator.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

        # scheduler.step()

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                questions = batch['question'].to(device)
                answers = batch['answer'].to(device)
                
                outputs = model(images, questions)
                loss = criterion(outputs, answers)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += answers.size(0)
                val_correct += predicted.eq(answers).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch results
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
        # Save model
        save_dir = os.path.join(os.getcwd(), 'weights', f"vqa_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_dir)
        print("Model saved!")

    return model, train_losses, val_losses, train_accs, val_accs


def test_model(model, train_df, test_df, dataset_dir):
    device = next(model.parameters()).device

    # Build answer vocabulary
    ans_to_idx, idx_to_ans = build_answer_vocab(train_df)
    
    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create test dataset
    test_dataset = SLAKEDataset(
        df=test_df,
        img_dir=dataset_dir,
        tokenizer=tokenizer,
        ans_to_idx=ans_to_idx
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    results = []
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            answers = batch['answer'].to(device)
            
            # Get predictions
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            
            # Accumulate loss
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Get the actual answer indices
            true_answer_idx = answers.item()
            pred_answer_idx = predicted.item()
            
            # Convert indices to answer text
            true_answer = idx_to_ans[true_answer_idx]
            pred_answer = idx_to_ans[pred_answer_idx]
            
            # Decode question (remove padding and special tokens)
            question_ids = questions[0].cpu().numpy()
            question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
            
            # Get image path from dataset
            image_path = test_dataset.data.iloc[idx]['img_name']
            
            # Store result
            is_correct = (true_answer_idx == pred_answer_idx)
            results.append({
                'image_path': image_path,
                'question': question_text,
                'true_answer': true_answer,
                'predicted_answer': pred_answer,
                'correct': is_correct,
                'confidence': torch.softmax(outputs, dim=1)[0][pred_answer_idx].item()
            })
            
            test_correct += is_correct
            test_total += 1
    
    # Calculate overall accuracy
    test_accuracy = 100. * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    print(f'\nTest Accuracy: {test_accuracy:.2f}% ({test_correct}/{test_total})')
    
    return results, test_accuracy, avg_test_loss
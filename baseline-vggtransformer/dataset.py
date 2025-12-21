from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch, json, os

class SLAKEDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, ans_to_idx, max_seq_len=50, transform=None):
        self.data = df
        self.img_dir = os.path.join(img_dir, 'imgs')
        self.tokenizer = tokenizer
        self.ans_to_idx = ans_to_idx
        self.max_seq_len = max_seq_len
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Load image
        img_path = f"{self.img_dir}/{item['img_name']}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize question
        question = item['question']
        tokens = self.tokenizer.encode(
            question,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)
        
        # Get answer index
        answer = item['answer']
        answer_idx = self.ans_to_idx.get(answer, 0)  # 0 for unknown
        
        return {
            'image': image,
            'question': tokens,
            'answer': torch.tensor(answer_idx, dtype=torch.long)
        }

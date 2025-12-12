import os, torch, nltk
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

class Vocab:
    def __init__(self, questions, min_freq=2):
        counter = Counter()
        for q in questions:
            tokens = word_tokenize(q.lower())
            counter.update(tokens)

        self.token2idx = {"<PAD>": 0, "<UNK>": 1}

        for word, freq in counter.items():
            if freq >= min_freq:
                self.token2idx[word] = len(self.token2idx)

        self.idx2token = {v: k for k, v in self.token2idx.items()}
    
    def encode(self, sentence):
        tokens = word_tokenize(sentence.lower())
        return [self.token2idx.get(tok, 1) for tok in tokens]
    
    def __len__(self):
        return len(self.token2idx)


class SLAKEDataset(Dataset):
    def __init__(self, df, img_dir, vocab, answer2idx, max_len=30, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.vocab = vocab
        self.answer2idx = answer2idx
        self.max_len = max_len
        self.transform = transform if transform else transforms.ToTensor()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, entry['img_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        question = self.vocab.encode(entry['question'])
        if len(question) < self.max_len:
            question += [0] * (self.max_len - len(question))
        else:
            question = question[:self.max_len]
        question = torch.tensor(question, dtype=torch.long)
        
        answer = torch.tensor(self.answer2idx[entry['answer']], dtype=torch.long)
        return image, question, answer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class Baseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_answers, pretrained=True):
        super(Baseline, self).__init__()
        # Image encoder
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.cnn = models.resnet50(weights=weights)
        self.cnn.fc = nn.Identity()  # remove classification layer
        self.img_feat_dim = 2048
        
        # Question encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.q_feat_dim = hidden_dim
        
        # Fusion + classifier
        self.fc1 = nn.Linear(self.img_feat_dim + self.q_feat_dim, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        
    def forward(self, images, questions):
        img_feats = self.cnn(images)
        q_emb = self.embedding(questions)
        _, h_n = self.gru(q_emb)
        q_feats = h_n.squeeze(0)
        fused = torch.cat([img_feats, q_feats], dim=1)
        x = F.relu(self.fc1(fused))
        out = self.fc2(x)
        return out
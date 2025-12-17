import torch
import torch.nn as nn
from torchvision import models

class VGG19TransformerVQA(nn.Module):
    """
    VGG19 + torch.nn.Transformer architecture for Visual Question Answering
    """
    def __init__(self, vocab_size, ans_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, max_seq_len=50):
        super(VGG19TransformerVQA, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # VGG19 for image feature extraction (remove classifier)
        vgg19 = models.vgg19(pretrained=True)
        self.image_encoder = nn.Sequential(*list(vgg19.features.children()))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.image_projection = nn.Linear(512, d_model)
        self.question_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Answer classifier
        self.answer_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, ans_vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, images, questions, question_mask=None):
        batch_size = images.size(0)
        
        # Extract image features with VGG19
        img_features = self.image_encoder(images)  # (batch, 512, H', W')
        img_features = self.adaptive_pool(img_features)  # (batch, 512, 7, 7)
        
        # Reshape to sequence: (batch, 49, 512)
        img_features = img_features.view(batch_size, 512, -1).permute(0, 2, 1)
        
        # Project to d_model
        img_features = self.image_projection(img_features)  # (batch, 49, d_model)
        
        # Question embedding
        question_emb = self.question_embedding(questions) * (self.d_model ** 0.5)
        question_emb = self.pos_encoder(question_emb)
        
        # Create attention masks
        if question_mask is None:
            question_mask = (questions == 0)  # Assume 0 is padding
        
        # Transformer expects True for positions to mask
        src_key_padding_mask = None  # No masking for image features
        tgt_key_padding_mask = question_mask
        
        # torch.nn.Transformer forward
        # Source: image features, Target: question embeddings
        transformer_out = self.transformer(
            src=img_features,
            tgt=question_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (batch, seq_len, d_model)
        
        # Use first token for classification (or mean pooling)
        pooled = transformer_out[:, 0, :]  # (batch, d_model)
        
        # Answer prediction
        answer_logits = self.answer_head(pooled)  # (batch, ans_vocab_size)
        
        return answer_logits

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
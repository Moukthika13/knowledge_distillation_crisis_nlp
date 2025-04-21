import torch
from torch import nn

class CrisisClassifierTransformer(nn.Module):
    def __init__(self, input_size, output_size=8, hidden_dim=128, num_heads=4,  dim_feedforward=2048,
                 num_layers=4, dropout=0.1, max_len=512, device='cpu'):
        super().__init__()
        # self.device = device

        self.embeddingL = nn.Embedding(input_size, hidden_dim)
        self.posembeddingL = nn.Embedding(max_len, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.informative_linear = nn.Linear(hidden_dim, out_features=2) # informative/not informative
        self.gating_linear = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.humanitarian_cat_linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        pos_size = x.shape[1]
        pos_one_row = torch.arange(pos_size, device=x.device)
        pos = pos_one_row.expand(x.shape)

        embedding = self.embeddingL(x) + self.posembeddingL(pos) # (batch_size, seq_len, hidden_dim)

        encoder_output = self.encoder(embedding) # (batch_size, seq_len, hidden_dim)
        pooled_output = encoder_output.mean(dim=1) # (batch_size, hiddem_dim)


        logits_informative = self.informative_linear(pooled_output) # (batch_size, 2)
        logits_gating = self.gating_linear(pooled_output) # (batch_size, hidden_dim)
        pooled_gating = pooled_output * logits_gating # (batch_size, hidden_dim)
        logits_humanitarian = self.humanitarian_cat_linear(pooled_gating) # (batch_size, 8)

        return logits_informative, logits_humanitarian



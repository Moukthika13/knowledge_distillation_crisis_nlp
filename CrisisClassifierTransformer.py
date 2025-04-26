import torch
from torch import nn
import torch.nn.functional as F

class CrisisClassifierTransformer(nn.Module):
    def __init__(self, input_size, output_size=11, hidden_dim=128, num_heads=4,  dim_feedforward=2048,
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

        # self.informative_linear = nn.Linear(hidden_dim, out_features=2) # informative/not informative
        # self.gating_linear = nn.Sequential(nn.Linear(hidden_dim + 2, hidden_dim), nn.Sigmoid())
        # self.humanitarian_cat_linear = nn.Linear(hidden_dim, output_size)

        self.informative_linear = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_features=2))

        self.gating_linear = nn.Sequential(nn.Linear(hidden_dim + 2, hidden_dim), nn.Sigmoid())

        self.humanitarian_cat_linear = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size))

    def forward(self, x):
        pos_size = x.shape[1]
        pos_one_row = torch.arange(pos_size, device=x.device)
        pos = pos_one_row.expand(x.shape)

        embedding = self.embeddingL(x) + self.posembeddingL(pos) # (batch_size, seq_len, hidden_dim)

        encoder_output = self.encoder(embedding) # (batch_size, seq_len, hidden_dim)
        pooled_output = encoder_output.mean(dim=1) # (batch_size, hiddem_dim)


        logits_informative = self.informative_linear(pooled_output) # (batch_size, 2)
        probability_informative = F.softmax(logits_informative, dim=1) # (batch_size, 2)
        pooled_plus_probs = torch.cat([pooled_output, probability_informative], dim=1) # (batch_size, hidden_dim + 2)

        logits_gating = self.gating_linear(pooled_plus_probs) # (batch_size, hidden_dim)
        pooled_gating = pooled_output * logits_gating # (batch_size, hidden_dim)
        logits_humanitarian = self.humanitarian_cat_linear(pooled_gating) # (batch_size, 8)

        return logits_informative, logits_humanitarian


# from A2
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        # FL = (1-pt)^g * (-log(pt))

        pt = F.softmax(input, dim=1)[range(input.shape[0]), target]
        log_pt = F.log_softmax(input, dim=1)[range(input.shape[0]), target]
        loss = self.weight[target] * (1 - pt).pow(self.gamma) * -log_pt

        loss = loss.mean()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
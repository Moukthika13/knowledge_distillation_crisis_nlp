import torch
from torch import nn


class CrisisClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CrisisClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        self.info_classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2))

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid()
        )
        
        # Second task: crisis classifier
        self.crisis_classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_classes))


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        informativeness_logits = self.info_classifier(pooled_output)
        informativeness_probs = torch.softmax(informativeness_logits, dim=1)
        informativeness_score = informativeness_probs[:, 1].unsqueeze(1)
        
        gated_output = informativeness_score * pooled_output
        #print("I am here")

        crisis_logits = self.crisis_classifier(gated_output)
        #print(crisis_logits)
    
        return informativeness_logits, crisis_logits

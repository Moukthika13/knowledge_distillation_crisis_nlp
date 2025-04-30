import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from CrisisClassifierTransformer import CrisisClassifierTransformer
from trainer import CrisisDataset
from sklearn.preprocessing import LabelEncoder
import pickle

# --- CONFIG ---
MODEL_FILENAME = "CrisisClassifierTransformer_20250423_0226_valInf0.3720_valHum0.5811.pt"
MODEL_PATH = "/Users/aweise/Desktop/group_assignment/DL_7643/group_assignment/models/" + MODEL_FILENAME
DATA_PATH = "/Users/aweise/Desktop/group_assignment/DL_7643/group_assignment/data/data/all_data_en/overlap_output/combined_test.tsv"
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model_pt = torch.load(MODEL_PATH, map_location=DEVICE)
HYPERPARAMETERS = model_pt['hyperparameters']

informative_label_encoder = LabelEncoder()
informative_label_encoder.classes_ = np.array(model_pt['label_encoders']['informative'])

humanitarian_label_encoder = LabelEncoder()
humanitarian_label_encoder.classes_ = np.array(model_pt['label_encoders']['humanitarian'])

informative_idx = informative_label_encoder.transform(['informative'])[0]
not_humanitarian_idx = humanitarian_label_encoder.transform(['not_humanitarian'])[0]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
df = pd.read_csv(DATA_PATH, sep='\t')

test_dataset = CrisisDataset(df, tokenizer, informative_label_encoder, humanitarian_label_encoder, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = CrisisClassifierTransformer(
        input_size=tokenizer.vocab_size,
        hidden_dim=HYPERPARAMETERS['hidden_dim'],
        num_heads=HYPERPARAMETERS['num_heads'],
        dim_feedforward=HYPERPARAMETERS['dim_feedforward'],
        num_layers=HYPERPARAMETERS['num_layers'],
        dropout=HYPERPARAMETERS['dropout'],
        max_len=HYPERPARAMETERS['tokenizer_max_len'],
        device=DEVICE,
    ).to(DEVICE)
model.load_state_dict(model_pt['model_state_dict'])
model.eval()

all_preds_info, all_labels_info, all_preds_human, all_labels_human  = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)

        logits_info, logits_human = model(input_ids)

        preds_info = torch.argmax(logits_info, dim=1).cpu().numpy()
        preds_human = torch.argmax(logits_human, dim=1).cpu().numpy()

        all_preds_info.extend(preds_info)
        all_preds_human.extend(preds_human)
        all_labels_info.extend(batch['label_informative'].numpy())
        all_labels_human.extend(batch['label_humanitarian'].numpy())

all_preds_info = np.array(all_preds_info)
all_labels_info = np.array(all_labels_info)
all_preds_human = np.array(all_preds_human)
all_labels_human = np.array(all_labels_human)

all_preds_human[all_preds_info != informative_idx] = not_humanitarian_idx

print("\n=== Informativeness Task ===")
print("Confusion Matrix:")
print(confusion_matrix(all_labels_info, all_preds_info))
print(classification_report(all_labels_info, all_preds_info, target_names=informative_label_encoder.classes_))
print("Accuracy:", accuracy_score(all_labels_info, all_preds_info))

print("\n=== Humanitarian Category Task ===")
print("Confusion Matrix:")
print(confusion_matrix(all_labels_human, all_preds_human))
print(classification_report(all_labels_human, all_preds_human, target_names=humanitarian_label_encoder.classes_))
print("Accuracy:", accuracy_score(all_labels_human, all_preds_human))
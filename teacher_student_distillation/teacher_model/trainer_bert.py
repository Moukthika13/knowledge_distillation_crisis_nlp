import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class CrisisDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        # self.text = dataframe.text
        # self.labels = dataframe.humanitarian_label
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        encoding = self.tokenizer(
            row['text'],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            #add_special_tokens=True,
            return_tensors="pt",
            #return_token_type_ids=False,
            #return_attention_mask=True,
            
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'crisis_type': int(row["humanitarian_label"]),
            'informativeness': int(row["informativeness_label"]),
        }



def train(model, dataloader, optimizer, criterion_informativeness, criterion_crisis, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        crisis_labels = batch['crisis_type'].to(device)
        informativeness_labels = batch['informativeness'].to(device)

        optimizer.zero_grad()

        informativeness_logits, crisis_logits = model(input_ids, attention_mask)
        #print("I am here2")

        loss1 = criterion_informativeness(informativeness_logits, informativeness_labels)

        # Create a mask
        informative_mask = informativeness_labels == 1  # shape: [B]

        # Apply mask to crisis loss
        #print(crisis_logits[informative_mask])
        #print(crisis_labels[informative_mask])
        if informative_mask.sum() > 0:
            loss2 = criterion_crisis(
                crisis_logits[informative_mask],
                crisis_labels[informative_mask]
            )
        else:
            loss2 = torch.tensor(0.0, device=device)
        
        #loss2 = criterion_crisis(crisis_logits, crisis_labels)
        #print(f"loss1: {loss1}, loss2: {loss2}")
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # _, preds = torch.max(crisis_logits, dim=1)
        # correct_predictions += torch.sum(preds == crisis_labels)

    return total_loss / len(dataloader) #, correct_predictions.double() / len(dataloader.dataset)


from sklearn.metrics import accuracy_score
import numpy as np

def eval(model, dataloader, device):
    model.eval()
    
    all_info_preds = []
    all_info_labels = []
    all_crisis_preds = []
    all_crisis_labels = []

    total_info_loss = 0
    total_crisis_loss = 0
    n_info = 0
    n_crisis = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            info_true = batch["informativeness"].to(device)
            crisis_true = batch["crisis_type"].to(device)

            informativeness_logits, crisis_logits = model(input_ids, attention_mask)


            info_loss = criterion_informativeness(informativeness_logits, info_true)
            total_info_loss += info_loss.item() * input_ids.size(0)
            n_info += input_ids.size(0)
            
            info_preds = torch.argmax(informativeness_logits, dim=1).cpu().numpy()
            crisis_preds = torch.argmax(crisis_logits, dim=1).cpu().numpy()

            all_info_preds.extend(info_preds)
            all_info_labels.extend(batch["informativeness"].numpy())
            all_crisis_preds.extend(crisis_preds)
            all_crisis_labels.extend(batch["crisis_type"].numpy())

            mask = info_true == 1
            if mask.sum() > 0:
                loss_crisis = criterion_crisis(crisis_logits[mask], crisis_true[mask])
                total_crisis_loss += loss_crisis.item() * mask.sum().item()
                n_crisis += mask.sum().item()


    # Accuracy
    #print(np.array(all_crisis_labels)[np.array(all_info_labels) == 1])
    info_acc = accuracy_score(all_info_labels, all_info_preds)
    crisis_acc = accuracy_score(
        np.array(all_crisis_labels)[np.array(all_info_labels) == 1],
        np.array(all_crisis_preds)[np.array(all_info_labels) == 1]
    ) if n_crisis > 0 else float('nan')

    # Loss
    avg_info_loss = total_info_loss / n_info
    avg_crisis_loss = total_crisis_loss / n_crisis if n_crisis > 0 else 0.0

    return avg_info_loss, avg_crisis_loss, info_acc, crisis_acc



if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")

#device = torch.device("cuda")


#### main

##################################### Load and Process data #####################################

# Load the pre-split CSVs
train_df = pd.read_csv("combined_train.tsv", sep="\t")
val_df = pd.read_csv("combined_dev.tsv", sep="\t")

# Drop any rows with missing values in the important columns
train_df = train_df.dropna(subset=["text", "informativeness_label", "humanitarian_label"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["text", "informativeness_label", "humanitarian_label"]).reset_index(drop=True)


# Informativeness labels
#info_labels = sorted(train_df["informativeness_label"].unique())
info_labels = ['not_informative', 'informative']
info2id = {label: idx for idx, label in enumerate(info_labels)}
id2info = {idx: label for label, idx in info2id.items()}

# Humanitarian (crisis type) labels
hum_labels = sorted(train_df["humanitarian_label"].unique())
hum2id = {label: idx for idx, label in enumerate(hum_labels)}
id2hum = {idx: label for label, idx in hum2id.items()}


train_df["informativeness_label"] = train_df["informativeness_label"].map(info2id)
val_df["informativeness_label"] = val_df["informativeness_label"].map(info2id)

train_df["humanitarian_label"] = train_df["humanitarian_label"].map(hum2id)
val_df["humanitarian_label"] = val_df["humanitarian_label"].map(hum2id)

data_train = CrisisDataset(train_df, tokenizer, max_len=128)
data_val = CrisisDataset(val_df, tokenizer, max_len=128)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

##################################### train #################################################

model = CrisisClassifier(n_classes=11).to(device)

train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
val_loader = DataLoader(data_val, batch_size=32, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion_informativeness = nn.CrossEntropyLoss()
criterion_crisis = nn.CrossEntropyLoss()

for epoch in range(3):
    print(f"Epoch {epoch + 1}/{3}")
    train_loss = train(model, train_loader, optimizer, criterion_informativeness, criterion_crisis, device)
    print(f"Train Loss: {train_loss:.4f}")

    avg_info_loss, avg_crisis_loss, informativeness_accuracy, crisis_accuracy = eval(model, val_loader, device)
    print(f"Informativeness Accuracy: {informativeness_accuracy:.4f}")
    print(f"Crisis Accuracy: {crisis_accuracy:.4f}")


##################################### save model and mappings #################################################

torch.save(model.state_dict(), "gated_bert_model.pt")

# Save label mappings
import json
with open("label_mappings.json", "w") as f:
    json.dump({
        "info2id": info2id,
        "id2info": id2info,
        "hum2id": hum2id,
        "id2hum": id2hum
    }, f, indent=4)



##################################### inference on test data  #################################################


test_df = pd.read_csv("combined_test.tsv", sep="\t")
test_df = test_df.dropna(subset=["text", "informativeness_label", "humanitarian_label"]).reset_index(drop=True)

# Apply the same label mappings
test_df["informativeness_label"] = test_df["informativeness_label"].map(info2id)
test_df["humanitarian_label"] = test_df["humanitarian_label"].map(hum2id)

data_test = CrisisDataset(test_df, tokenizer, max_len=128)
test_loader = DataLoader(data_test, batch_size=32)

# Evaluation
info_preds, info_true, crisis_preds, crisis_true = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        info_logits, crisis_logits = model(input_ids, attention_mask)

        info_preds.extend(torch.argmax(info_logits, dim=1).cpu().tolist())
        crisis_preds.extend(torch.argmax(crisis_logits, dim=1).cpu().tolist())

        info_true.extend(batch["informativeness"].tolist())
        crisis_true.extend(batch["crisis_type"].tolist())



info_accuracy = accuracy_score(info_true, info_preds)
print(f"Informativeness Accuracy: {info_accuracy:.4f}")

info_true = np.array(info_true)
crisis_true = np.array(crisis_true)
crisis_preds = np.array(crisis_preds)

# Filter where informativeness == 1
mask = info_true == 1

filtered_crisis_true = crisis_true[mask]
filtered_crisis_preds = crisis_preds[mask]

if len(filtered_crisis_true) > 0:
    crisis_accuracy = accuracy_score(filtered_crisis_true, filtered_crisis_preds)
    print(f"Crisis Type Accuracy (Informative Only): {crisis_accuracy:.4f}")
else:
    print("No informative tweets found in test set for crisis accuracy.")




##################################### metrics and plot  #################################################


print("Informativeness Report:\n")
print(classification_report(info_true, info_preds, target_names=[id2info[i] for i in sorted(id2info)]))

# Crisis report
print("Crisis Type Report:\n")
print(classification_report(crisis_true, crisis_preds, target_names=[id2hum[i] for i in sorted(id2hum)]))

# Confusion matrix
def plot_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Humanitatian data {title}: {title:.2f}")
    plt.savefig(f"humanitarian_data{title}.png", dpi=300)

plot_cm(info_true, info_preds, [id2info[i] for i in sorted(id2info)], "Informativeness Confusion Matrix")
plot_cm(crisis_true, crisis_preds, [id2hum[i] for i in sorted(id2hum)], "Crisis Type Confusion Matrix")


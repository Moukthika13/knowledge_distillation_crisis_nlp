import pickle

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from CrisisClassifierTransformer import CrisisClassifierTransformer
import optuna

DATA_FOLDER = "/Users/aweise/Documents/OMSCS/2025_Spring/DL_7643/group_assignment/data/data/all_data_en/overlap_output"
DEV_DATA_PATH = DATA_FOLDER + "/combined_dev.tsv"
TRAIN_DATA_PATH = DATA_FOLDER + "/combined_train.tsv"
TEST_DATA_PATH = DATA_FOLDER + "/combined_test.tsv"
HYPERPARAMETER_CHOICES = {
    # 'epochs' : [5, 10, 20, 50],
    'epochs' : [50],
    # 'hidden_dim' : [32, 64, 128, 256, 512],
    'hidden_dim' : [512],
    # 'dim_feedforward' : [64, 128, 256, 512, 1024, 2048],
    'dim_feedforward' : [2048],
    # 'num_heads' : [2, 4, 8, 16],
    'num_heads' : [8],
    # 'num_layers' : [2, 4, 8, 16],
    'num_layers' : [2],
    # 'dropout' : [0.0, 0.3, 0.6, 0.9],
    'dropout' : [0.3],
    # 'lr' : [1e-5, 1e-4, 1e-3, 1e-2],
    'lr' : [1e-5],
    # 'batch_size' : [32, 64, 128],
    'batch_size' : [32],
    # 'alpha' : [0.1, 0.3, 0.6, 0.9, 1.5, 2.0],
    'alpha' : [0.1],
    # 'tokenizer_max_len' : [64, 128, 256],
    'tokenizer_max_len' : [128],
}

HYPERPARAMETERS = {
    'epochs' : 10,
    'hidden_dim' : 512,
    'dim_feedforward' : 2048,
    'num_heads' : 8,
    'num_layers' : 2,
    'dropout' : 0.3,
    'lr' : 1e-5,
    'batch_size' : 32,
    'alpha' : 0.1,
    'tokenizer_max_len' : 128,
}

class CrisisDataset(Dataset):
    def __init__(self, df, tokenizer, informative_label_encoder, humanitarian_label_encoder, max_len=128):
        self.text = df['text'].tolist()
        self.informative_labels = informative_label_encoder.transform(df['informativeness_label'])
        self.humanitarian_labels = humanitarian_label_encoder.transform(df['humanitarian_label'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_informative': torch.tensor(self.informative_labels[idx], dtype=torch.long),
            'label_humanitarian': torch.tensor(self.humanitarian_labels[idx], dtype=torch.long),
        }


def train_one_epoch(model, dataloader, optimizer, device, alpha=1.0):
    model.train()
    losses, informative_losses, humanitarian_losses = [], [], []
    progress_bar = tqdm(dataloader, ascii=True)
    for batch_idx, data in enumerate(progress_bar):
        input_ids = data['input_ids'].to(device)
        # attention_mask = data['attention_mask'].to(device)
        label_informative = data['label_informative'].to(device)
        label_humanitarian = data['label_humanitarian'].to(device)

        logits_informative, logits_humanitarian = model(input_ids)
        loss_informative = nn.CrossEntropyLoss()(logits_informative, label_informative)

        mask = label_informative == 0  # 0 == informative
        if mask.any():
            loss_humanitarian = nn.CrossEntropyLoss()(logits_humanitarian[mask], label_humanitarian[mask])
        else:
            loss_humanitarian = torch.tensor(0., dtype=torch.float, device=device)

        # scale only humanitarian loss because it's the more complex task
        loss = loss_informative + alpha * loss_humanitarian

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # total_loss += loss.item()
        # total_informative_loss += loss_informative.item()
        # total_humanitarian_loss += loss_humanitarian.item()

        losses.append(loss.item())
        informative_losses.append(loss_informative.item())
        humanitarian_losses.append(loss_humanitarian.item())

        progress_bar.set_description_str(
            f"Batch {batch_idx + 1} | Total: {loss.item():.4f} | Info: {loss_informative.item():.4f} | Human: {loss_humanitarian.item():.4f}"
        )

    return {
        'mean_losses': np.mean(losses),
        'mean_informative_losses': np.mean(informative_losses),
        'mean_humanitarian_losses': np.mean(humanitarian_losses),
    }

def evaluate(model, dataloader, device):
    model.eval()
    informative_losses, humanitarian_losses = [], []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Val')
        for batch_idx, data in enumerate(progress_bar):
            input_ids = data['input_ids'].to(device)
            label_informative = data['label_informative'].to(device)
            label_humanitarian = data['label_humanitarian'].to(device)
            logits_informative, logits_humanitarian = model(input_ids)

            loss_informative = nn.CrossEntropyLoss()(logits_informative, label_informative)

            mask = label_informative == 0  # 0 == informative
            if mask.any():
                loss_humanitarian = nn.CrossEntropyLoss()(logits_humanitarian[mask], label_humanitarian[mask])
            else:
                loss_humanitarian = torch.tensor(0., dtype=torch.float, device=device)

            informative_losses.append(loss_informative.item())
            humanitarian_losses.append(loss_humanitarian.item())

            progress_bar.set_description_str(
                f"Batch {batch_idx + 1} | Info: {loss_informative.item():.4f} | Human: {loss_humanitarian.item():.4f}"
            )

        return {
            'mean_informative_losses': np.mean(informative_losses),
            'mean_humanitarian_losses': np.mean(humanitarian_losses),
        }

def get_data(tokenizer_max_len=128, batch_size=64):
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')

    informative_label_encoder = LabelEncoder().fit(train_df['informativeness_label'])
    humanitarian_label_encoder = LabelEncoder().fit(train_df['humanitarian_label'])

    print(informative_label_encoder.classes_)
    print(humanitarian_label_encoder.classes_)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    training_dataset = CrisisDataset(
        train_df,
        tokenizer,
        informative_label_encoder,
        humanitarian_label_encoder,
        max_len=tokenizer_max_len
    )
    validation_dataset = CrisisDataset(
        dev_df,
        tokenizer,
        informative_label_encoder,
        humanitarian_label_encoder,
        max_len=tokenizer_max_len
    )
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)  # HYPERPARAM

    return {
        'tokenizer': tokenizer,
        'training_dataloader': training_dataloader,
        'validation_dataloader': validation_dataloader,
        'informative_label_encoder': informative_label_encoder,
        'humanitarian_label_encoder': humanitarian_label_encoder,
    }

def objective(trial):
    hyperparam_suggestions = {
        param: trial.suggest_categorical(param, choices)
        for param, choices in HYPERPARAMETER_CHOICES.items()
    }

    data = get_data(
        hyperparam_suggestions['tokenizer_max_len'],
        hyperparam_suggestions['batch_size'],
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = CrisisClassifierTransformer(
        input_size=data['tokenizer'].vocab_size,
        hidden_dim=hyperparam_suggestions['hidden_dim'],
        num_heads=hyperparam_suggestions['num_heads'],
        dim_feedforward=hyperparam_suggestions['dim_feedforward'],
        num_layers=hyperparam_suggestions['num_layers'],
        dropout=hyperparam_suggestions['dropout'],
        max_len=hyperparam_suggestions['tokenizer_max_len'],
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam_suggestions['lr'])

    # for i in range(hyperparam_suggestions['epochs']):
    for i in range(1):
        train_one_epoch(model, data['training_dataloader'], optimizer, device, hyperparam_suggestions['alpha'])

    evaluate_output = evaluate(model, data['validation_dataloader'], device)
    return (evaluate_output['mean_informative_losses'] +
            hyperparam_suggestions['alpha'] * evaluate_output['mean_humanitarian_losses'])

def main(model_class, num_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    study_name = f"{model_class.__name__}_study.pkl"
    with open(study_name, 'wb') as f:
        pickle.dump(study, f)

    print(f"Best params: {study.best_params}")
    params = study.best_params

    data = get_data(params['tokenizer_max_len'], params['batch_size'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    best_model = CrisisClassifierTransformer(
        input_size=data['tokenizer'].vocab_size,
        hidden_dim=params['hidden_dim'],
        num_heads=params['num_heads'],
        dim_feedforward=params['dim_feedforward'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        max_len=params['tokenizer_max_len'],
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=params['lr'])

    train_losses, informative_losses, humanitarian_losses = [], [], []
    validation_informative_losses, validation_humanitarian_losses = [], []
    for epoch in range(params['epochs']):
        training_output = train_one_epoch(best_model, data['training_dataloader'], optimizer, device, params['alpha'])
        evaluate_output = evaluate(best_model, data['validation_dataloader'], device)
        print(
            f"Epoch {epoch}: Train Tot={training_output['mean_losses']:.4f}, "
            f"Inf={training_output['mean_informative_losses']:.4f}, "
            f"Hum={training_output['mean_humanitarian_losses']:.4f} | "
            f"Val Inf={evaluate_output['mean_informative_losses']:.4f}, "
            f"Hum={evaluate_output['mean_humanitarian_losses']:.4f}"
        )
        train_losses.append(training_output['mean_losses'])
        informative_losses.append(training_output['mean_informative_losses'])
        humanitarian_losses.append(training_output['mean_humanitarian_losses'])
        validation_informative_losses.append(evaluate_output['mean_informative_losses'])
        validation_humanitarian_losses.append(evaluate_output['mean_humanitarian_losses'])


    torch.save(best_model.state_dict(), 'models/crisis_transformer_10_epochs.pt')

    plt.figure()
    epochs = range(1, params['epochs'] + 1)
    plt.plot(epochs, train_losses, label='Train Total Loss')
    plt.plot(epochs, informative_losses, label='Train Inf Loss')
    plt.plot(epochs, humanitarian_losses, label='Train Hum Loss')
    plt.plot(epochs, validation_informative_losses, '--', label='Val Inf Loss')
    plt.plot(epochs, validation_humanitarian_losses, '--', label='Val Hum Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()


if __name__ == "__main__":
    main(CrisisClassifierTransformer, 1)
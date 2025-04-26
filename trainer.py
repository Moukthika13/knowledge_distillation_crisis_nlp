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
from CrisisClassifierTransformer import CrisisClassifierTransformer, FocalLoss
import optuna
from datetime import datetime
from collections import Counter


DATA_FOLDER = "/Users/aweise/Desktop/group_assignment/DL_7643/group_assignment/data/data/all_data_en/overlap_output"
DEV_DATA_PATH = DATA_FOLDER + "/combined_dev.tsv"
TRAIN_DATA_PATH = DATA_FOLDER + "/combined_train.tsv"
TEST_DATA_PATH = DATA_FOLDER + "/combined_test.tsv"

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

HYPERPARAMETER_CHOICES = {
    'epochs' : [5, 10, 20],
    'hidden_dim' : [512, 768, 1024],
    'dim_feedforward' : [1024, 2048, 3072],
    'num_heads' : [4, 8, 16],
    'num_layers' : [4, 6, 8, 10],
    'dropout' : [0.3, 0.6, 0.9],
    'lr' : [5e-6, 1e-5, 2e-5],
    'batch_size' : [32, 64, 128],
    'alpha' : [0.005, 0.01, 0.05, 0.1],
    'tokenizer_max_len' : [64, 128, 256],
    'gamma': [1.0, 2.0, 3.0],
}

HYPERPARAMETERS = {
    'epochs' : 10,
    'hidden_dim' : 1024, #512,
    'dim_feedforward' : 2048,
    'num_heads' : 8,
    'num_layers' : 8, # 2,
    'dropout' : 0.3,
    'lr' : 1e-5,
    'batch_size' : 32,
    'alpha' : 0.01,
    'tokenizer_max_len' : 128,
    'gamma' : 2.0,
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

def get_data(tokenizer_max_len=128, batch_size=64):
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')

    informative_label_encoder = LabelEncoder().fit(train_df['informativeness_label'])
    assert informative_label_encoder.transform(['informative'])[0] == 0, \
        "Expected 'informative' to be encoded as 0"
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

    humanitarian_class_counts = Counter(train_df['humanitarian_label'])
    total = sum(humanitarian_class_counts.values())
    humanitarian_class_weights = torch.tensor([
        1.0 - (humanitarian_class_counts[i] / total)
        for i in range(len(humanitarian_label_encoder.classes_))
    ], dtype=torch.float)

    return {
        'tokenizer': tokenizer,
        'training_dataloader': training_dataloader,
        'validation_dataloader': validation_dataloader,
        'informative_label_encoder': informative_label_encoder,
        'humanitarian_label_encoder': humanitarian_label_encoder,
        'humanitarian_class_weights': humanitarian_class_weights,
    }

def get_model(vocab_size, params, device=DEVICE):
    return CrisisClassifierTransformer(
        input_size=vocab_size,
        hidden_dim=params['hidden_dim'],
        num_heads=params['num_heads'],
        dim_feedforward=params['dim_feedforward'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        max_len=params['tokenizer_max_len'],
        device=device,
    ).to(device)

def compute_loss(
        logits_informative,
        logits_humanitarian,
        label_informative,
        label_humanitarian,
        alpha,
        device=DEVICE
    ):
    loss_informative = nn.CrossEntropyLoss()(logits_informative, label_informative)
    mask = label_informative == 0  # 0 == informative
    if mask.any():
        loss_humanitarian = nn.CrossEntropyLoss()(logits_humanitarian[mask], label_humanitarian[mask])
    else:
        loss_humanitarian = torch.tensor(0., dtype=torch.float, device=device)

    return {
        'loss': compute_total_loss(loss_informative, loss_humanitarian, alpha),
        'loss_informative': loss_informative,
        'loss_humanitarian': loss_humanitarian,
    }

def compute_loss_focal(
        logits_informative,
        logits_humanitarian,
        label_informative,
        label_humanitarian,
        alpha,
        humanitarian_weights,
        gamma,
        device=DEVICE
    ):
    loss_informative = nn.CrossEntropyLoss()(logits_informative, label_informative)
    mask = label_informative == 0 # 0 == informative

    focal_loss = FocalLoss(weight=humanitarian_weights.to(device), gamma=gamma)

    if mask.any():
        loss_humanitarian = focal_loss(logits_humanitarian[mask], label_humanitarian[mask])
    else:
        loss_humanitarian = torch.tensor(0., dtype=torch.float, device=device)

    return {
        'loss': compute_total_loss(loss_informative, loss_humanitarian, alpha),
        'loss_informative': loss_informative,
        'loss_humanitarian': loss_humanitarian,
    }

def compute_total_loss(loss_informative, loss_humanitarian, alpha):
    return alpha * loss_informative + (1 - alpha) * loss_humanitarian

def run_one_epoch(
        model,
        dataloader,
        optimizer=None,
        device=DEVICE,
        alpha=1.0,
        gamma=1.0,
        is_train=True,
        humanitarian_weights=None
):
    model.train() if is_train else model.eval()

    losses, informative_losses, humanitarian_losses = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress_bar = tqdm(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            input_ids = data['input_ids'].to(device)
            label_informative = data['label_informative'].to(device)
            label_humanitarian = data['label_humanitarian'].to(device)

            logits_informative, logits_humanitarian = model(input_ids)

            # loss_output = compute_loss(
            #     logits_informative,
            #     logits_humanitarian,
            #     label_informative,
            #     label_humanitarian,
            #     alpha
            # )

            loss_output = compute_loss_focal(
                logits_informative,
                logits_humanitarian,
                label_informative,
                label_humanitarian,
                alpha,
                humanitarian_weights=humanitarian_weights,
                gamma=gamma,
            )

            loss = loss_output['loss']
            loss_informative = loss_output['loss_informative']
            loss_humanitarian = loss_output['loss_humanitarian']

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

def train_and_evaluate(model,
                       training_dataloader,
                       validation_dataloader,
                       optimizer,
                       epochs,
                       alpha=1.0,
                       gamma=1.0,
                       humanitarian_weights=None,
                       device=DEVICE):
    train_losses, informative_losses, humanitarian_losses = [], [], []
    validation_informative_losses, validation_humanitarian_losses = [], []
    for epoch in range(epochs):
        training_output = run_one_epoch(
            model,
            training_dataloader,
            optimizer,
            device,
            alpha,
            gamma,
            is_train=True,
            humanitarian_weights=humanitarian_weights,
        )
        evaluate_output = run_one_epoch(
            model,
            validation_dataloader,
            device=device,
            alpha=alpha,
            gamma=gamma,
            is_train=False,
            humanitarian_weights=humanitarian_weights,
        )

        print(
            f"Epoch {epoch+1}: Train Tot={training_output['mean_losses']:.4f}, "
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

    return {
        'train_losses': train_losses,
        'informative_losses': informative_losses,
        'humanitarian_losses': humanitarian_losses,
        'validation_informative_losses': validation_informative_losses,
        'validation_humanitarian_losses': validation_humanitarian_losses,
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


    model = get_model(
        data['tokenizer'].vocab_size,
        hyperparam_suggestions,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam_suggestions['lr'])

    losses = train_and_evaluate(
        model,
        data['training_dataloader'],
        data['validation_dataloader'],
        optimizer,
        epochs=hyperparam_suggestions['epochs'],
        alpha=hyperparam_suggestions['alpha'],
        gamma=hyperparam_suggestions['gamma'],
        humanitarian_weights=data['humanitarian_class_weights'],
    )

    return compute_total_loss(
        losses['validation_informative_losses'][-1],
        losses['validation_humanitarian_losses'][-1],
        hyperparam_suggestions['alpha']
    )

def plot_losses(losses):
    plt.figure()
    epochs = range(1, len(losses['train_losses']) + 1)
    plt.plot(epochs, losses['train_losses'], label='Train Total Loss')
    plt.plot(epochs, losses['informative_losses'], label='Train Inf Loss')
    plt.plot(epochs, losses['humanitarian_losses'], label='Train Hum Loss')
    plt.plot(epochs, losses['validation_informative_losses'], '--', label='Val Inf Loss')
    plt.plot(epochs, losses['validation_humanitarian_losses'], '--', label='Val Hum Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()

def main(model_class, num_trials=0):
    if num_trials > 0:
        # study = optuna.create_study(direction='minimize')
        # study.optimize(objective, n_trials=num_trials)

        # study_name = f"{model_class.__name__}_study.pkl"
        # with open(study_name, 'wb') as f:
        #     pickle.dump(study, f)
        study_name = f"{model_class.__name__}_study"
        storage_path = f"sqlite:///{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=num_trials)

        print(f"Best params: {study.best_params}")
        with open(f"{study_name}.pkl", "wb") as f:
            pickle.dump(study, f)
        params = study.best_params
    else:
        params = HYPERPARAMETERS

    data = get_data(params['tokenizer_max_len'], params['batch_size'])

    model = get_model(data['tokenizer'].vocab_size, params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    losses = train_and_evaluate(
        model,
        data['training_dataloader'],
        data['validation_dataloader'],
        optimizer,
        params['epochs'],
        params['alpha'],
        params['gamma'],
        humanitarian_weights=data['humanitarian_class_weights'],
    )

    plot_losses(losses)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    val_inf_loss = losses['validation_informative_losses'][-1]
    val_hum_loss = losses['validation_humanitarian_losses'][-1]
    pt_file = (
        f"models/{model_class.__name__}_{timestamp}_valInf{val_inf_loss:.4f}_valHum{val_hum_loss:.4f}.pt"
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': params,
        'label_encoders': {
            'informative': data['informative_label_encoder'].classes_.tolist(),
            'humanitarian': data['humanitarian_label_encoder'].classes_.tolist(),
        },
    },
    pt_file)


if __name__ == "__main__":
    main(CrisisClassifierTransformer, num_trials=100)
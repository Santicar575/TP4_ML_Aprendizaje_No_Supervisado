import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy

class ImageDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            image = torch.tensor(self.X[idx], dtype=torch.float32)
            label = self.y[idx]
                
            return image, label
    
def get_dataloaders(X_train_std, y_train, X_val_std, y_val, batch_size=128):
    """
    Crea los DataLoaders a partir de los arrays ya separados estratificadamente y estandarizados.
    """
    # Instanciamos los datasets
    train_ds = ImageDataset(X_train_std, y_train)
    val_ds = ImageDataset(X_val_std, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_p=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec
    
    def get_latent_features(model, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        model.eval() 
        with torch.no_grad():
            z = model.encoder(X_tensor) 
        return z.cpu().numpy()
    
    def reconstruct(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            X_rec = self.forward(X)
        return X_rec.cpu().numpy()

class Trainer():
    def __init__(self, model, train_loader, val_loader, lr=0.001, wd=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.MSELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def train(self, early_stopping=False, patience=10, min_delta=0.001, epochs=300):

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        loss_hist = []
        val_loss_hist = []

        for epoch in range(epochs):
            train_loss, val_loss = self.run_epoch()
            loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early Stopping en epoch {epoch+1}!")
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break
                        
        return loss_hist, val_loss_hist

    def run_epoch(self):
        self.model.train()
        total_loss_train = 0

        # training loop
        for batch in self.train_loader:
            inputs, _ = batch
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs) 
            loss.backward()
            
            self.optimizer.step()
            total_loss_train += loss.item()
        
        self.model.eval()
        total_loss_val = 0
        
        # validation loop
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                total_loss_val += loss.item()

        train_loss = total_loss_train / len(self.train_loader)
        val_loss = total_loss_val / len(self.val_loader)
        return train_loss, val_loss

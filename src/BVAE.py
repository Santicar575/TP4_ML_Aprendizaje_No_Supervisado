import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_p=0.0):
        super().__init__()
        
        # Encoder
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        )
        
        # Capas de proyección para Mu y LogVar
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar
    
    def get_latent_features(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval() 
        with torch.no_grad():
            mu, logvar = self.encode(X_tensor)
            z = self.reparameterize(mu, logvar)
        return z.cpu().numpy()
    
    def reconstruct(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            x_rec, _, _ = self.forward(X_tensor)
        return x_rec.cpu().numpy()

class BVAETrainer():
    def __init__(self, model, train_loader, val_loader, lr=0.001, wd=0.0, beta=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300, eta_min=1e-5)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) * x.size(1))
        
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def train(self, early_stopping=False, patience=10, min_delta=0.001, epochs=300):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Guardo las 3 curvas por separado
        history = {'train_total': [], 'train_recon': [], 'train_kl': [],
                   'val_total': [], 'val_recon': [], 'val_kl': []}

        for epoch in range(epochs):
            t_total, t_recon, t_kl = self.run_epoch(is_train=True)
            v_total, v_recon, v_kl = self.run_epoch(is_train=False)
            
            history['train_total'].append(t_total); history['train_recon'].append(t_recon); history['train_kl'].append(t_kl)
            history['val_total'].append(v_total); history['val_recon'].append(v_recon); history['val_kl'].append(v_kl)
            
            self.scheduler.step()

            if early_stopping:
                # El early stopping se hace sobre la Loss Total de validación
                if v_total < best_val_loss - min_delta:
                    best_val_loss = v_total
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early Stopping en epoch {epoch+1}!")
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break
                        
        return history

    def run_epoch(self, is_train):
        if is_train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
            
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0

        with torch.set_grad_enabled(is_train):
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                
                outputs, mu, logvar = self.model(inputs)
                loss, recon, kl = self.loss_function(outputs, inputs, mu, logvar)
                
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                total_loss_epoch += loss.item()
                recon_loss_epoch += recon.item()
                kl_loss_epoch += kl.item()

        num_batches = len(loader)
        return total_loss_epoch / num_batches, recon_loss_epoch / num_batches, kl_loss_epoch / num_batches
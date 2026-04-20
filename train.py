import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import SiameseCNN
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 8
    
    # Dataset and DataLoader
    train_dataset = ChangeDetectionDataset(root_dir='dataset', split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ChangeDetectionDataset(root_dir='dataset', split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, Loss, Optimizer
    model = SiameseCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (baseline, recent, mask) in enumerate(train_loader):
            baseline, recent, mask = baseline.to(device), recent.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(baseline, recent)
            
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for baseline, recent, mask in val_loader:
                baseline, recent, mask = baseline.to(device), recent.to(device), mask.to(device)
                outputs = model(baseline, recent)
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'erbil_siamese_model.pth')
            print(f"-> Saved new best model with Val Loss: {val_loss:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train()

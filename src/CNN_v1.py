import torch
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = nn.BCELoss()

    def forward(x):
        return self.model(x)

    def fit(self, X_train, y_train, epochs=100, batch_size=32, verbose=False):
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.train()  # Set to training mode
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_X, batch_y in bar:
                # Forward pass
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            accuracy = correct / total
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        if not verbose:
            print(f"Training complete - Final Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

if __name__ == "__main__":
    model = CNN()
    summary(model)
    
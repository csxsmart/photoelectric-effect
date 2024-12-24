import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)


# 定义Dataset
class PhotoelectricDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        I_noisy = data['I_noisy'].values
        self.I_noisy = torch.tensor(I_noisy, dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1]

    def __len__(self):
        return len(self.I_noisy)

    def __getitem__(self, idx):
        return self.I_noisy[idx], self.I_noisy[idx]  # 输入和目标相同


# 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(csv_file, model_save_path, num_epochs=200, batch_size=8, learning_rate=1e-3):
    # 创建Dataset和DataLoader
    dataset = PhotoelectricDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型、定义损失函数和优化器
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_noisy, batch_clean in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_noisy)
            loss = criterion(outputs, batch_clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_noisy.size(0)
        epoch_loss /= len(dataloader.dataset)
        loss_history.append(epoch_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to '{model_save_path}'.")

    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    csv_file = 'experimental_data.csv'
    model_save_path = 'autoencoder_model.pth'
    train_autoencoder(csv_file, model_save_path)

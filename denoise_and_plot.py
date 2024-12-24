import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 定义Autoencoder模型（必须与训练时的结构相同）
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


def denoise_data(csv_file, model_path):
    # 加载数据
    data = pd.read_csv(csv_file)
    nu = data['nu'].values
    I_noisy = data['I_noisy'].values

    # 实例化模型并加载参数
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 转换数据为张量
    I_noisy_tensor = torch.tensor(I_noisy, dtype=torch.float32).unsqueeze(1)

    # 进行降噪
    with torch.no_grad():
        I_denoised_tensor = model(I_noisy_tensor)
    I_denoised = I_denoised_tensor.squeeze().numpy()

    print("Denoising complete.")

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(nu, I_noisy, 'o', markersize=6, label='Noisy Data', alpha=0.7)
    plt.plot(nu, I_denoised, 'r-', label='Denoised Data', linewidth=2)
    plt.xlabel('Frequency ν (Hz)')
    plt.ylabel('Photoelectric Current I (A)')
    plt.title('Denoised Photoelectric Current vs Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存降噪后的数据
    denoised_df = pd.DataFrame({
        'nu': nu,
        'I_denoised': I_denoised
    })
    denoised_df.to_csv('denoised_data.csv', index=False)
    print("Denoised data saved to 'denoised_data.csv'.")


if __name__ == "__main__":
    csv_file = 'experimental_data.csv'
    model_path = 'autoencoder_model.pth'
    denoise_data(csv_file, model_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 物理常数
h_true = 6.626e-34  # 普朗克常数 (J·s)
phi = 2.0e-19       # 功函数 (J)

# 数据范围
nu_min = 2.50e14    # 最低频率 (Hz)
nu_max = 4.00e14    # 最高频率 (Hz)
num_points = 26     # 数据点数，每隔1.00e12 Hz一个点

# 生成频率数据
nu = np.arange(nu_min, nu_max + 1e12, 1.00e12)

# 理论光电流关系
I0 = 1e-6  # 光电流比例常数 (A·Hz/J)
nu0 = phi / h_true  # 阈值频率 ≈ 3.02e14 Hz
I_clean = I0 * (nu - nu0)
I_clean[nu < nu0] = 0  # 阈值频率以下光电流为零

# 添加噪声
noise_level = 1e-7  # 噪声水平
noise = noise_level * np.random.randn(len(nu))
I_noisy = I_clean + noise
I_noisy[I_noisy < 0] = 0  # 光电流不能为负

# 创建DataFrame并保存为CSV
df = pd.DataFrame({
    'nu': nu,
    'I_noisy': I_noisy
})
df.to_csv('experimental_data.csv', index=False)
print("CSV文件 'experimental_data.csv' 已成功创建。")

# 绘制有噪声数据与真实数据
plt.figure(figsize=(10, 6))
plt.plot(nu, I_noisy, 'o', markersize=6, label='Noisy Data', alpha=0.7)
plt.plot(nu, I_clean, 'r-', label='Clean Data', linewidth=2)
plt.xlabel('Frequency ν (Hz)')
plt.ylabel('Photoelectric Current I (A)')
plt.title('Simulated Photoelectric Effect Data')
plt.legend()
plt.grid(True)
plt.show()

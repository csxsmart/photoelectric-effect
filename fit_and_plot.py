import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 定义光电效应的理论模型
def photoelectric_current(nu, I0, h, phi):
    """
    理论光电流方程基于爱因斯坦的光电效应。

    参数：
    - nu: 频率 (Hz)
    - I0: 比例常数 (A·Hz/J)
    - h: 普朗克常数 (J·s)
    - phi: 功函数 (J)

    返回：
    - I: 光电流 (A)
    """
    nu0 = phi / h
    return I0 * (nu - nu0) * (nu > nu0)


def fit_data(denoised_csv, initial_guess=[1e-6, 6.626e-34, 2.0e-19]):
    # 加载降噪后的数据
    data = pd.read_csv(denoised_csv)
    nu = data['nu'].values
    I_denoised = data['I_denoised'].values

    # 设置拟合掩码，选择频率大于阈值频率的80%
    h_initial = initial_guess[1]
    phi_initial = initial_guess[2]
    nu_threshold = phi_initial / h_initial
    fit_mask = nu > (nu_threshold * 0.8)
    nu_fit = nu[fit_mask]
    I_fit = I_denoised[fit_mask]

    if len(I_fit) == 0:
        print("No data points available for fitting. Please check the frequency range.")
        return

    # 进行曲线拟合
    popt, pcov = curve_fit(photoelectric_current, nu_fit, I_fit, p0=initial_guess)
    I0_opt, h_opt, phi_opt = popt
    I0_err, h_err, phi_err = np.sqrt(np.diag(pcov))

    print("\nOptimized Parameters:")
    print(f"I0 = {I0_opt:.3e} ± {I0_err:.3e} A·Hz/J")
    print(f"h  = {h_opt:.3e} ± {h_err:.3e} J·s")
    print(f"Phi = {phi_opt:.3e} ± {phi_err:.3e} J")

    # 生成拟合曲线
    I_fitted = photoelectric_current(nu_fit, *popt)

    # 绘制拟合结果
    plt.figure(figsize=(10, 6))
    plt.plot(nu_fit, I_fit, 'b.', label='Denoised Data for Fitting')
    plt.plot(nu_fit, I_fitted, 'r-', label='Fitted Curve', linewidth=2)
    plt.xlabel('Frequency ν (Hz)')
    plt.ylabel('Photoelectric Current I (A)')
    plt.title('Photoelectric Effect Data Fitting Results')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存拟合结果
    fitted_df = pd.DataFrame({
        'nu_fit': nu_fit,
        'I_fit': I_fit,
        'I_fitted': I_fitted
    })
    fitted_df.to_csv('fitted_data.csv', index=False)
    print("Fitted data saved to 'fitted_data.csv'.")


if __name__ == "__main__":
    denoised_csv = 'denoised_data.csv'
    fit_data(denoised_csv)

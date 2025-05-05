import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify

def f(x):
    """原始函数 f(x) = 1 + 0.5*tanh(2x)"""
    return 1 + 0.5 * np.tanh(2 * x)

def get_analytical_derivative():
    """获取解析导数函数"""
    x = symbols('x')
    expr = diff(1 + 0.5 * tanh(2 * x), x)
    return lambdify(x, expr)

def calculate_central_difference(x, f):
    """使用中心差分法计算数值导数"""
    dy = []
    for i in range(1, len(x)-1):
        h = x[i+1] - x[i]
        dy.append((f(x[i+1]) - f(x[i-1])) / (2 * h))
    return np.array(dy)

def richardson_derivative_all_orders(x, f, h, max_order=3):
    """使用Richardson外推法计算不同阶数的导数值"""
    R = np.zeros((max_order + 1, max_order + 1))
    
    # 计算第一列（不同步长的中心差分）
    for i in range(max_order + 1):
        hi = h / (2**i)
        R[i, 0] = (f(x + hi) - f(x - hi)) / (2 * hi)
    
    # Richardson外推
    for j in range(1, max_order + 1):
        for i in range(max_order - j + 1):
            R[i, j] = (4**j * R[i+1, j-1] - R[i, j-1]) / (4**j - 1)
    
    return [R[0, j] for j in range(1, max_order + 1)]

def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """Create comparison plots to show derivative results and errors"""
    # Create four subplots, including one for step size sensitivity analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Calculate analytical solution
    analytical = df_analytical(x)
    analytical_central = df_analytical(x_central)
    
    # First subplot: Derivative comparison
    ax1.plot(x, analytical, 'b-', label='Analytical Solution')
    ax1.plot(x_central, dy_central, 'ro', markersize=4, label='Central Difference')
    ax1.plot(x, dy_richardson[:, 1], 'g^', markersize=4, label='Richardson (2nd Order)')
    ax1.set_title('Derivative Comparison')
    ax1.set_xlabel('x')
    ax1.set_ylabel('dy/dx')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Error comparison
    error_central = np.abs(dy_central - analytical_central)
    error_richardson = np.abs(dy_richardson[:, 1] - analytical)
    
    ax2.plot(x_central, error_central, 'ro', markersize=4, label='Central Difference Error')
    ax2.plot(x, error_richardson, 'g^', markersize=4, label='Richardson Error')
    ax2.set_yscale('log')
    ax2.set_title('Error Analysis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.legend()
    ax2.grid(True)
    
    # Third subplot: Different orders of Richardson extrapolation errors
    for i, order in enumerate(['1st', '2nd', '3rd']):
        error = np.abs(dy_richardson[:, i] - analytical)
        ax3.plot(x, error, marker='^', markersize=4, label=f'Richardson {order}')
    ax3.set_yscale('log')
    ax3.set_title('Richardson Extrapolation Error Comparison')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Absolute Error (log scale)')
    ax3.legend()
    ax3.grid(True)
    
    # Fourth subplot: Step size sensitivity analysis
    h_values = np.logspace(-6, -1, 20)
    x_test = 0.0  # Analyze step size sensitivity at x=0
    central_errors = []
    richardson_errors = []
    expected = df_analytical(x_test)
    
    for h in h_values:
        # Central difference error
        central_result = (f(x_test + h) - f(x_test - h)) / (2 * h)
        central_errors.append(abs(central_result - expected))
        
        # Richardson extrapolation error (2nd order)
        rich_result = richardson_derivative_all_orders(x_test, f, h, max_order=3)[1]
        richardson_errors.append(abs(rich_result - expected))
    
    ax4.loglog(h_values, central_errors, 'ro-', label='Central Difference')
    ax4.loglog(h_values, richardson_errors, 'g^-', label='Richardson (2nd Order)')
    ax4.set_title('Step Size Sensitivity Analysis')
    ax4.set_xlabel('Step Size h (log scale)')
    ax4.set_ylabel('Absolute Error (log scale)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数，运行数值微分实验"""
    # 设置参数
    h_initial = 0.1  # 使用更小的初始步长以提高精度
    max_order = 3
    N_points = 200   # 增加采样点数以获得更平滑的曲线
    x = np.linspace(-2, 2, N_points)
    
    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    dy_central = calculate_central_difference(x, f)
    x_central = x[1:-1]
    
    # 计算Richardson外推导数
    dy_richardson = np.array([
        richardson_derivative_all_orders(xi, f, h_initial, max_order=max_order)
        for xi in x
    ])
    
    # 绘制结果对比图
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify

def f(x):
    """计算函数值 f(x) = 1 + 0.5*tanh(2x)
    
    参数：
        x: 标量或numpy数组，输入值
    
    返回：
        标量或numpy数组，函数值
    """
    return 1 + 0.5 * np.tanh(2 * x)

def get_analytical_derivative():
    """使用sympy获取解析导数函数
    
    返回：
        可调用函数，用于计算导数值
    """
    x = symbols('x')
    f_sym = 1 + 0.5 * tanh(2 * x)
    df_sym = diff(f_sym, x)
    df_func = lambdify(x, df_sym, 'numpy')
    return df_func

def calculate_central_difference(x, f, h=0.01):
    """使用中心差分法计算数值导数
    
    参数：
        x: numpy数组，要计算导数的点
        f: 可调用函数，要求导的函数
        h: 浮点数，步长
    
    返回：
        numpy数组，x处的导数值
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def richardson_derivative(f, x, h, max_order=3):
    """使用Richardson外推法计算导数值
    
    参数：
        f: 可调用函数，要求导的函数
        x: 标量，要计算导数的点
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数
    
    返回：
        浮点数，外推后的导数值
    """
    d = np.zeros((max_order + 1, max_order + 1))
    
    for i in range(max_order + 1):
        h_i = h / (2 ** i)
        d[i, 0] = (f(x + h_i) - f(x - h_i)) / (2 * h_i)
        
        for j in range(1, i + 1):
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (4 ** j - 1)
    
    return d[max_order, max_order]

def richardson_derivative_all_orders(x, f, h, max_order=3):
    """使用Richardson外推法计算不同阶数的导数值
    
    参数：
        x: 标量，要计算导数的点
        f: 可调用函数，要求导的函数
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数
    
    返回：
        列表，不同阶数计算的导数值
    """
    d = np.zeros((max_order + 1, max_order + 1))
    results = []
    
    for i in range(max_order + 1):
        h_i = h / (2 ** i)
        d[i, 0] = (f(x + h_i) - f(x - h_i)) / (2 * h_i)
        
        for j in range(1, i + 1):
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (4 ** j - 1)
        
        results.append(d[i, i] if i > 0 else d[i, 0])
    
    return results

def create_comparison_plot(x, dy_central, dy_richardson, df_analytical, h_values):
    """创建对比图，展示导数计算结果和误差分析
    
    参数：
        x: numpy数组，所有x坐标点
        dy_central: numpy数组，中心差分法计算的导数值
        dy_richardson: numpy数组，Richardson方法计算的导数值
        df_analytical: 可调用函数，解析导数函数
        h_values: 步长列表
    """
    # 计算解析导数
    dy_analytical = df_analytical(x)
    
    # 计算误差
    error_central = np.abs(dy_central - dy_analytical)
    error_richardson = np.abs(dy_richardson - dy_analytical)
    
    # 计算不同步长下的误差
    test_point = 0.5  # 选择一个测试点
    central_errors = []
    richardson_errors = []
    
    for h in h_values:
        central_diff = calculate_central_difference(np.array([test_point]), f, h)[0]
        central_errors.append(np.abs(central_diff - df_analytical(test_point)))
        
        rich_diff = richardson_derivative(f, test_point, h)
        richardson_errors.append(np.abs(rich_diff - df_analytical(test_point)))
    
    # 计算不同阶数的Richardson外推误差
    h_initial = 0.1
    richardson_orders = []
    max_order = 5
    for order in range(1, max_order + 1):
        rich_diff = richardson_derivative(f, test_point, h_initial, order)
        richardson_orders.append(np.abs(rich_diff - df_analytical(test_point)))
    
    # 创建四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 导数对比图
    ax1.plot(x, dy_analytical, label='解析解', linewidth=2)
    ax1.plot(x, dy_central, '--', label='中心差分法', alpha=0.8)
    ax1.plot(x, dy_richardson, ':', label='Richardson外推法', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")
    ax1.set_title('导数计算方法比较')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 误差分析图
    ax2.plot(x, error_central, label='中心差分法误差')
    ax2.plot(x, error_richardson, label='Richardson外推法误差')
    ax2.set_xlabel('x')
    ax2.set_ylabel('绝对误差')
    ax2.set_title('误差分布')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 步长敏感性分析图（双对数坐标）
    ax3.loglog(h_values, central_errors, 'o-', label='中心差分法')
    ax3.loglog(h_values, richardson_errors, 's-', label='Richardson外推法')
    ax3.set_xlabel('步长 h (log)')
    ax3.set_ylabel('绝对误差 (log)')
    ax3.set_title('步长敏感性分析 (双对数坐标)')
    ax3.legend()
    ax3.grid(True, which="both", ls="--")
    
    # 4. Richardson外推不同阶数误差对比图
    orders = range(1, max_order + 1)
    ax4.semilogy(orders, richardson_orders, 'o-')
    ax4.set_xlabel('外推阶数')
    ax4.set_ylabel('绝对误差 (log)')
    ax4.set_title('Richardson外推不同阶数误差')
    ax4.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig('derivative_comparison.png')
    plt.show()

def main():
    """运行数值微分实验的主函数"""
    # 设置实验参数
    x = np.linspace(-2, 2, 200)
    h_central = 0.01
    h_richardson = 0.1
    
    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    dy_central = calculate_central_difference(x, f, h_central)
    
    # 计算Richardson外推导数
    dy_richardson = np.array([richardson_derivative(f, xi, h_richardson) for xi in x])
    
    # 准备步长列表用于分析
    h_values = [10**-i for i in range(1, 7)]
    
    # 绘制结果对比图
    create_comparison_plot(x, dy_central, dy_richardson, df_analytical, h_values)

if __name__ == '__main__':
    main()

# 1. 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. 生成数据：红色点（0）和蓝色点（1）
np.random.seed(114514)  # 固定随机数，保证每次结果一样

# 红色点：中心在 (2,2)，方差1
red_points = np.random.randn(50, 2) + [2, 2]
# 蓝色点：中心在 (5,5)，方差1
blue_points = np.random.randn(50, 2) + [5, 5]

# 合并数据
X = np.vstack([red_points, blue_points])   # 坐标
y = np.hstack([np.zeros(50), np.ones(50)]) # 标签 0=红, 1=蓝

# 3. 划分训练集和测试集（留一部分数据来检验模型好坏）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)
# 对训练集和测试集做预测（预测结果是连续值，比如0.3、0.8）
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)
# 把连续值变成类别：大于0.5就是1（蓝），否则0（红）
y_train_class_lr = (y_train_pred_lr > 0.5).astype(int)
y_test_class_lr = (y_test_pred_lr > 0.5).astype(int)

# 5. 训练线性SVM模型
svm = SVC(kernel='linear', C=100)  # C很大表示不允许犯错误（硬间隔）
svm.fit(X_train, y_train)
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)

# 6. 计算准确率
acc_lr_train = accuracy_score(y_train, y_train_class_lr)
acc_lr_test  = accuracy_score(y_test,  y_test_class_lr)
acc_svm_train = accuracy_score(y_train, y_train_pred_svm)
acc_svm_test  = accuracy_score(y_test,  y_test_pred_svm)

print("线性回归 - 训练集准确率: {:.2f}%".format(acc_lr_train*100))
print("线性回归 - 测试集准确率: {:.2f}%".format(acc_lr_test*100))
print("SVM - 训练集准确率: {:.2f}%".format(acc_svm_train*100))
print("SVM - 测试集准确率: {:.2f}%".format(acc_svm_test*100))

# 7. 画出两个模型的决策边界（辅助理解）
def plot_decision_boundary(model, X, y, title, ax):
    # 找到坐标范围
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # 对于线性回归，我们需要自定义预测函数（因为它是连续值）
    if isinstance(model, LinearRegression):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = (Z > 0.5).astype(int).reshape(xx.shape)
    else:  # SVM
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[y==0][:,0], X[y==0][:,1], c='red', edgecolor='k', label='Red')
    ax.scatter(X[y==1][:,0], X[y==1][:,1], c='blue', edgecolor='k', label='Blue')
    ax.set_title(title)
    ax.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
plot_decision_boundary(lr, X, y, "Linear Regression (threshold 0.5)", ax1)
plot_decision_boundary(svm, X, y, "Linear SVM", ax2)
plt.show()
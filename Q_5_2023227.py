import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

n_points = 100
x = np.random.uniform(0, 2*np.pi, n_points)
x = np.sort(x)
noise = np.random.normal(0, 0.1, n_points)
y = np.sin(x) + noise

degrees = [1, 2, 3, 4]

folds = 5

indices = np.arange(n_points)
np.random.shuffle(indices)
fold_size = n_points // folds

errors = {}

for d in degrees:
    mse_folds = []
    for fold in range(folds):
        start = fold * fold_size
        if fold == folds - 1:
            test_idx = indices[start:]
        else:
            test_idx = indices[start:start+fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        
        coeffs = np.polyfit(x_train, y_train, d)
        
        y_pred = np.polyval(coeffs, x_test)
        
        mse = np.mean((y_test - y_pred) ** 2)
        mse_folds.append(mse)
    
    errors[d] = np.mean(mse_folds)

best_degree = min(errors, key=errors.get)
print("MSE for each degree:", errors)
print("Selected degree:", best_degree)

best_coeffs = np.polyfit(x, y, best_degree)

x_grid = np.linspace(0, 2*np.pi, 200)
y_true = np.sin(x_grid)
y_model = np.polyval(best_coeffs, x_grid)

plt.figure(figsize=(10, 6))
plt.plot(x_grid, y_true, label='True function sin(x)', color='green')
plt.scatter(x, y, label='Noisy training points', color='blue', s=30)
plt.plot(x_grid, y_model, label=f'Polynomial Regression (degree {best_degree})', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Model Performance with 5-Fold Cross-Validation')
plt.legend()

plt.savefig('regression_plot.png')
print("Plot saved as regression_plot.png")

plt.figure(figsize=(15, 10))
for i, d in enumerate(degrees):
    coeffs = np.polyfit(x, y, d)
    y_model = np.polyval(coeffs, x_grid)
    
    plt.subplot(2, 2, i+1)
    plt.plot(x_grid, y_true, label='True function sin(x)', color='green')
    plt.scatter(x, y, label='Noisy training points', color='blue', s=20, alpha=0.5)
    plt.plot(x_grid, y_model, label=f'Polynomial (degree {d})', color='red')
    plt.title(f'Degree {d} Polynomial (MSE: {errors[d]:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('all_models_comparison.png')
print("All models comparison saved as all_models_comparison.png")

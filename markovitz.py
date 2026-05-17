import data_loader
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

mu = data_loader.mu
num_stocks = len(mu)

sigma = data_loader.sigma

# Portfolio volitility
def volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix @ weights)

# Return of portfolio
def return_(weight, mean_returns):
    return np.sum(weight * mean_returns)

# Constraining weights
bounds_ = tuple((0, 1) for _ in range(num_stocks))
ones_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

# Setting up variables
args_ = (sigma,)
init_weights = np.ones(num_stocks) / num_stocks

# Finding minimum risk portfolio
def minimum_variance():
    result = sco.minimize(volatility, init_weights, args=args_,
                          method='SLSQP', bounds=bounds_, constraints=ones_constraint)
    return result

def efficient_return(target_return):
    constraints = ({'type': 'eq', 'fun': lambda x: return_(x, mu) - target_return},
                   ones_constraint)
    
    result = sco.minimize(volatility, init_weights, args=args_,
                          method='SLSQP', bounds=bounds_, constraints=constraints)
    return result

# Minimum variance portfolio
min_var_result = minimum_variance()
min_var_vol = min_var_result.fun
min_var_ret = return_(min_var_result.x, mu)

# Efficient frontier
target_returns = np.linspace(min_var_ret, mu.max(), 50)
frontier_volatility = []

for target in target_returns:
    opt_result = efficient_return(target)
    frontier_volatility.append(opt_result.fun)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(frontier_volatility, target_returns, 'b', label='Efficient Frontier')
plt.scatter(min_var_vol, min_var_ret, color='r', label='Minimum Variance Portfolio')
plt.title('Markowitz Efficient Frontier')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()
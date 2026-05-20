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

def analytical_efficient_frontier(mu_vec, sigma_mat, target_returns):
    """
    Wyznacza analityczną granicę efektywną (dopuszczająca krótką sprzedaż)
    za pomocą metody mnożników Lagrange'a z użyciem numpy.linalg.solve.
    """
    mu_np = np.array(mu_vec)
    sigma_np = np.array(sigma_mat)
    ones = np.ones_like(mu_np)
    
    # Rozwiązanie układów równań w celu stabilnego wyznaczenia Sigma^-1 * mu i Sigma^-1 * 1
    g1 = np.linalg.solve(sigma_np, mu_np)
    g2 = np.linalg.solve(sigma_np, ones)
    
    # Parametry analityczne granicy
    A = np.dot(mu_np, g1)
    B = np.dot(mu_np, g2)
    C = np.dot(ones, g2)
    D = A * C - B**2
    
    risks = []
    for r in target_returns:
        var = (C * r**2 - 2 * B * r + A) / D
        risks.append(np.sqrt(var))
        
    return np.array(risks)


def analytical_portfolio_weights(mu_vec, sigma_mat, target_return):
    """
    Wyznacza wagi portfela dla stopy docelowej przy użyciu mnożników Lagrange'a
    (krótka sprzedaż dozwolona).
    """
    mu_np = np.array(mu_vec)
    sigma_np = np.array(sigma_mat)
    ones = np.ones_like(mu_np)
    
    g1 = np.linalg.solve(sigma_np, mu_np)
    g2 = np.linalg.solve(sigma_np, ones)
    
    A = np.dot(mu_np, g1)
    B = np.dot(mu_np, g2)
    C = np.dot(ones, g2)
    D = A * C - B**2
    
    lambda1 = (C * target_return - B) / D
    lambda2 = (A - B * target_return) / D
    
    return lambda1 * g1 + lambda2 * g2


if __name__ == "__main__":
    # Portfel minimalnej wariancji
    min_var_result = minimum_variance()
    min_var_vol = min_var_result.fun
    min_var_ret = return_(min_var_result.x, mu)

    # Numeryczna granica efektywna (z ograniczeniem braku krótkiej sprzedaży: w_i >= 0)
    target_returns = np.linspace(min_var_ret, mu.max(), 50)
    frontier_volatility = []

    for target in target_returns:
        opt_result = efficient_return(target)
        frontier_volatility.append(opt_result.fun)

    # Analityczna granica efektywna (z dozwoloną krótką sprzedażą)
    analytical_volatility = analytical_efficient_frontier(mu, sigma, target_returns)

    # Rysowanie wykresu porównawczego
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_volatility, target_returns, 'b-', linewidth=2, label='Granica numeryczna (w >= 0)')
    plt.plot(analytical_volatility, target_returns, 'g--', linewidth=2, label='Granica analityczna Lagrange\'a (krótka sprzedaż dozwolona)')
    plt.scatter(min_var_vol, min_var_ret, color='r', s=80, zorder=5, label='Portfel min. wariancji (numeryczny)')
    plt.title('Porównanie klasycznych granic efektywnych Markowitza')
    plt.xlabel('Roczna zmienność (odchylenie standardowe)')
    plt.ylabel('Roczna oczekiwana stopa zwrotu')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
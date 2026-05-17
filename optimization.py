import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

rng = np.random.default_rng(12345)

portfolio_dir = os.path.join(os.path.dirname(__file__), 'Portfolio-optimization')
csv_path = os.path.join(portfolio_dir, 'dane.csv')

df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

prices = df["Close"]
log_ret = (np.log(prices / prices.shift(1))).dropna()

mu = log_ret.mean() * 252
sigma = log_ret.cov() * 252

num_stocks = len(mu)

print(f"Liczba akcji: {num_stocks}")
print(f"Średnie zwroty:\n{mu}")
print(f"\nMacierz kowariancji:\n{sigma}")

# hermonogram chlodzenia 
def geometric_cooling(T0, alpha, k):
    return T0 * (alpha ** k)

# regula metropolisa (minimalizacja)
def metropolis_accept(delta, T, rng):
    if delta <= 0:
        return True
    if T <= 0:
        return False
    return rng.random() < np.exp(-delta / T)

# projekcja wektora w na sympleks (algorytm duchi)
def project_simplex(w):
    n = len(w)
    w_sorted = np.sort(w)[::-1]
    cumsum = np.cumsum(w_sorted)
    
    rho = 0
    for j in range(n):
        if w_sorted[j] + (1.0 - cumsum[j]) / (j + 1) > 0:
            rho = j
    
    theta = (1.0 - cumsum[rho]) / (rho + 1)
    w_proj = np.maximum(w + theta, 0.0)
    
    return w_proj

# funkcja celu dla optymalizacji portfela (min wariancje oraz kara za niedostarczenie docelowego zwrotu)
def portfolio_objective_with_shortfall(weights, mu, sigma, target_return=0.0, penalty=1000.0):
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = weights @ sigma @ weights
    
    shortfall = max(0.0, target_return - portfolio_return)
    
    objective = portfolio_variance + penalty * shortfall
    
    return objective


# symulowane wyzarzanie dla optymalizacji portfela z projekcja na sympleks
def symulowane_wyzarzanie_portfel(
    mu,
    sigma,
    target_return=0.0,
    penalty=1000.0,
    n_iter=5000,
    T0=1.0,
    alpha=0.995,
    step_size=0.1,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng()
    
    num_assets = len(mu)
    
    x = np.ones(num_assets) / num_assets
    
    def f(w):
        return portfolio_objective_with_shortfall(w, mu, sigma, target_return, penalty)
    
    fx = f(x)
    
    best_x = x.copy()
    best_fx = fx
    
    history_x = [x.copy()]
    history_f = [fx]
    history_best_f = [best_fx]
    history_T = [T0]
    accepted_flags = []
    
    for k in range(n_iter):
        T = geometric_cooling(T0, alpha, k)
        
        local_scale = step_size * (0.2 + 0.8 * T / T0)
        
        perturbation = local_scale * rng.normal(size=num_assets)
        candidate = x + perturbation
        
        candidate = project_simplex(candidate)
        
        f_candidate = f(candidate)
        delta = f_candidate - fx
        
        accepted = metropolis_accept(delta, T, rng)
        accepted_flags.append(accepted)
        
        if accepted:
            x = candidate
            fx = f_candidate
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        history_x.append(x.copy())
        history_f.append(fx)
        history_best_f.append(best_fx)
        history_T.append(T)
    
    return {
        "best_weights": best_x,
        "best_objective": best_fx,
        "best_return": np.dot(best_x, mu),
        "best_risk": np.sqrt(best_x @ sigma @ best_x),
        "history_x": np.array(history_x),
        "history_f": np.array(history_f),
        "history_best_f": np.array(history_best_f),
        "history_T": np.array(history_T),
        "accepted_flags": np.array(accepted_flags, dtype=bool)
    }

# wykres zbieznosci funkcji celu
def plot_convergence(history_f, history_best_f, title, filename='convergence.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(history_f, label="wartość bieżąca", alpha=0.7)
    plt.plot(history_best_f, label="najlepsza dotąd", linewidth=2)
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# wykres harmonogramu temp
def plot_temperature(history_T, title, filename='temperature.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(history_T)
    plt.xlabel("Iteracja")
    plt.ylabel("Temperatura")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# wykres wag portfela
def plot_weights(weights, title, filename='weights.png'):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(weights))
    plt.bar(indices, weights)
    plt.xlabel("Indeks akcji")
    plt.ylabel("Waga")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# wykres granicy efektywności z zaznaczonym wynikiem symiulowanego wyzarzania
def plot_efficient_frontier_with_result(mu, sigma, result, n_points=50, filename='efficient_frontier.png'):
    min_return = mu.min()
    max_return = mu.max()
    target_returns = np.linspace(min_return, max_return, n_points)
    
    risks = []
    for target in target_returns:
        from scipy.optimize import minimize
        
        def obj(w):
            return w @ sigma @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target}
        ]
        bounds = [(0, 1) for _ in range(len(mu))]
        
        res = minimize(
            obj, 
            np.ones(len(mu)) / len(mu), 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if res.success:
            risks.append(np.sqrt(res.fun))
        else:
            risks.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(risks, target_returns, 'b-', linewidth=2, label='Granica efektywności')
    plt.scatter(
        result['best_risk'], 
        result['best_return'], 
        color='red', 
        s=100, 
        zorder=5,
        label=f'Wynik SA (zwrot={result["best_return"]:.4f}, ryzyko={result["best_risk"]:.4f})'
    )
    plt.xlabel('Ryzyko (odchylenie standardowe)')
    plt.ylabel('Oczekiwany zwrot')
    plt.title('Granica efektywności portfela')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# funkcja do uruchamiania optymalizacji portfela
def uruchamianie_optymalizacji_portfela():
    target_return = 0.15
    
    print(f"OPTYMALIZACJA PORTFELA - SYMULOWANE WYŻARZANIE")
    print(f"Docelowy zwrot: {target_return:.4f}")
    print(f"Liczba akcji: {num_stocks}")
    
    result = symulowane_wyzarzanie_portfel(
        mu=mu,
        sigma=sigma,
        target_return=target_return,
        penalty=500.0,
        n_iter=8000,
        T0=2.0,
        alpha=0.9985,
        step_size=0.15,
        rng=rng
    )
    
    print(f"WYNIKI OPTYMALIZACJI")
    print(f"Najlepsza wartość funkcji celu: {result['best_objective']:.6f}")
    print(f"Oczekiwany zwrot portfela: {result['best_return']:.6f}")
    print(f"Ryzyko portfela (odch. std.): {result['best_risk']:.6f}")
    print(f"Współczynnik Sharpe'a (przy rf=0): {result['best_return'] / result['best_risk']:.6f}")
    
    print(f"\nWagi portfela:")
    for i, w in enumerate(result['best_weights']):
        if w > 0.001: 
            print(f"  Akcja {i}: {w:.6f} ({w*100:.2f}%)")
    
    print(f"\nSuma wag: {np.sum(result['best_weights']):.10f}")
    print(f"Minimalna waga: {np.min(result['best_weights']):.10f}")
    print(f"Liczba akcji w portfelu (waga > 0.1%): {np.sum(result['best_weights'] > 0.001)}")
    
    acceptance_rate = np.mean(result['accepted_flags'])
    print(f"\nWspółczynnik akceptacji: {acceptance_rate:.4f}")
    
    
    plot_convergence(
        result['history_f'],
        result['history_best_f'],
        "Optymalizacja portfela: zbieżność funkcji celu",
        filename='convergence.png'
    )
    
    plot_temperature(
        result['history_T'],
        "Optymalizacja portfela: harmonogram temperatury",
        filename='temperature.png'
    )
    
    plot_weights(
        result['best_weights'],
        f"Optymalne wagi portfela (zwrot={result['best_return']:.4f}, ryzyko={result['best_risk']:.4f})",
        filename='weights.png'
    )
    
    plot_efficient_frontier_with_result(mu, sigma, result, n_points=30, filename='efficient_frontier.png')
    
    
    return result


if __name__ == "__main__":
    result = uruchamianie_optymalizacji_portfela()

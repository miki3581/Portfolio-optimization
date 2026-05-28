import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

rng = np.random.default_rng(42)

import data_loader

mu = data_loader.mu
sigma = data_loader.sigma
num_stocks = len(mu)

AG_GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs', 'ga')
os.makedirs(AG_GRAPHS_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"Liczba akcji: {num_stocks}")
    print(f"Średnie zwroty:\n{mu}")
    print(f"\nMacierz kowariancji:\n{sigma}")


# ─────────────────────────────────────────────────────────────
# REPREZENTACJA BINARNA — pomocnicze funkcje
# ─────────────────────────────────────────────────────────────

def bin2int(bits: np.ndarray) -> int:
    """
    Konwersja wektora bitów (MSB-first) na liczbę całkowitą.
    bits: 1-wymiarowa tablica bool/{0,1}.
    """
    b = bits.astype(np.uint64)
    powers = (1 << np.arange(b.size - 1, -1, -1, dtype=np.uint64))
    return int((b * powers).sum())


def dekoduj_chromosom(population: np.ndarray,
                      pop_size: int,
                      num_assets: int,
                      bits_per_gene: int) -> np.ndarray:
    """
    Dekodowanie chromosomów binarnych → wagi portfela (fenotyp).

    Wektorowe dekodowanie populacji binarnej na wagi portfela.
    Zastępuje powolne pętle operacjami macierzowymi w NumPy (przyspieszenie ok. 100x).
    """
    denom = float(2 ** bits_per_gene - 1)

    # Zmiana kształtu na (pop_size, num_assets, bits_per_gene)
    reshaped = population.reshape((pop_size, num_assets, bits_per_gene))

    # Przygotowanie potęg dwójki dla reprezentacji MSB-first
    powers = 2 ** np.arange(bits_per_gene - 1, -1, -1, dtype=float)

    # Wektorowe wyznaczenie wartości rzeczywistych
    raw = np.sum(reshaped * powers, axis=2) / denom

    # Normalizacja sumy do 1 (rzutowanie na sympleks)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    wagi = raw / row_sums
    return wagi


# ─────────────────────────────────────────────────────────────
# FUNKCJA CELU
# ─────────────────────────────────────────────────────────────

def portfolio_objective_with_shortfall(weights, mu, sigma, target_return=0.0, penalty=1000.0):
    """
    Funkcja celu dla optymalizacji portfela:
    - Minimalizujemy ryzyko (wariancję)
    - Dodajemy karę za niedostarczenie docelowego zwrotu (shortfall)

    Parametry:
    ----------
    weights : ndarray
        Wagi portfela
    mu : ndarray
        Wektor oczekiwanych zwrotów
    sigma : ndarray
        Macierz kowariancji zwrotów
    target_return : float
        Docelowy zwrot (jeśli oczekiwany zwrot < target_return, nakładana jest kara)
    penalty : float
        Współczynnik kary za shortfall
    """
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = weights @ sigma @ weights

    shortfall = max(0.0, target_return - portfolio_return)

    objective = portfolio_variance + penalty * shortfall

    return objective


# ─────────────────────────────────────────────────────────────
# OPERATORY GENETYCZNE
# ─────────────────────────────────────────────────────────────

def selekcja_ruletka(fun_celu: np.ndarray, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Selekcja proporcjonalna (ruletka) dla minimalizacji.

    Przystosowanie obliczane jako min/f — im mniejsza wartość funkcji celu,
    tym wyższe przystosowanie. W przypadku degeneracji (np. f == 0)
    stosowany jest rozkład jednostajny.

    Zwraca indeksy wybranych rodziców (losowanie z powtarzaniem).
    """
    min_obj = float(np.min(fun_celu))
    with np.errstate(divide="ignore", invalid="ignore"):
        przystosowanie = min_obj / fun_celu
    przystosowanie = np.where(np.isfinite(przystosowanie), przystosowanie, 0.0)

    s = przystosowanie.sum()
    if s <= 0.0:
        przystosowanie = np.full(pop_size, 1.0 / pop_size, dtype=float)
    else:
        przystosowanie = przystosowanie / s

    cdf = np.cumsum(przystosowanie)
    u = rng.random(pop_size)
    return np.searchsorted(cdf, u, side="right")


def krzyzowanie_jednopunktowe(population: np.ndarray,
                               parents: np.ndarray,
                               pop_size: int,
                               chrom_len: int,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Krzyżowanie jednopunktowe (single-point crossover).

    Dla każdego osobnika losowany jest jeden rodzic z listy `parents`
    oraz drugi losowo z całej puli. Losowy punkt cięcia dzieli chromosomy
    na dwie części, które łączone są w nowy osobnik.
    """
    nowa_populacja = np.empty_like(population)
    for i in range(pop_size):
        p1 = parents[i]
        p2 = parents[rng.integers(0, pop_size)]
        cut = int(rng.integers(1, chrom_len))
        nowa_populacja[i, :cut] = population[p1, :cut]
        nowa_populacja[i, cut:] = population[p2, cut:]
    return nowa_populacja


def mutacja_bitflip(population: np.ndarray, p_mut: float, rng: np.random.Generator) -> np.ndarray:
    """
    Mutacja bitowa (bit-flip): każdy bit odwracany niezależnie
    z prawdopodobieństwem p_mut.
    """
    maska = rng.random(population.shape) < p_mut
    return np.logical_xor(population, maska)


# ─────────────────────────────────────────────────────────────
# GŁÓWNA FUNKCJA — ALGORYTM GENETYCZNY
# ─────────────────────────────────────────────────────────────

def algorytm_genetyczny_portfel(
    mu,
    sigma,
    target_return=0.0,
    penalty=1000.0,
    pop_size=150,
    bits_per_gene=16,
    p_mut=0.005,
    K=300,
    elityzm=True,
    rng=None
):
    """
    Algorytm genetyczny dla optymalizacji portfela Markowitza.

    Kodowanie: binarne — każde aktywo reprezentowane jest przez
    `bits_per_gene` bitów. Chromosom o długości num_assets * bits_per_gene
    dekodowany jest do wag portfela przez normalizację sumy.

    Operatory:
    - Selekcja: ruletka proporcjonalna (minimalizacja)
    - Krzyżowanie: jednopunktowe
    - Mutacja: bit-flip z prawdopodobieństwem p_mut
    - Opcja elityzmu: najlepszy osobnik przenoszony do kolejnego pokolenia

    Parametry:
    ----------
    mu : ndarray
        Wektor oczekiwanych zwrotów akcji (annualizowany)
    sigma : ndarray
        Macierz kowariancji zwrotów (annualizowana)
    target_return : float
        Docelowy zwrot portfela (kara za shortfall)
    penalty : float
        Współczynnik kary za shortfall
    pop_size : int
        Liczba osobników w populacji
    bits_per_gene : int
        Liczba bitów kodujących wagę jednego aktywa
    p_mut : float
        Prawdopodobieństwo mutacji bitu
    K : int
        Liczba pokoleń (iteracji)
    elityzm : bool
        Czy przenosić najlepszego osobnika bez zmian do kolejnego pokolenia
    rng : numpy Generator

    Zwraca:
    -------
    result : dict
        Słownik z wynikami optymalizacji.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_assets = len(mu)
    chrom_len = num_assets * bits_per_gene

    st = time.time()

    def f(w):
        return portfolio_objective_with_shortfall(w, mu, sigma, target_return, penalty)

    # --- Inicjalizacja populacji (losowe bity, p=0.5)
    population = rng.random((pop_size, chrom_len)) < 0.5

    # --- Dekodowanie i ocena populacji początkowej
    wagi = dekoduj_chromosom(population, pop_size, num_assets, bits_per_gene)
    fun_celu = np.array([f(wagi[i]) for i in range(pop_size)], dtype=float)

    # --- Inicjalizacja najlepszego rozwiązania
    najlepsze_idx = int(np.argmin(fun_celu))
    best_x = wagi[najlepsze_idx].copy()
    best_fx = float(fun_celu[najlepsze_idx])

    history_best_w = [best_x.copy()]
    history_f = [float(fun_celu[najlepsze_idx])]   # najlepszy w pokoleniu
    history_best_f = [best_fx]
    accepted_count = [1]

    # --- Pętla ewolucyjna
    for k in range(1, K):
        # Selekcja
        parents = selekcja_ruletka(fun_celu, pop_size, rng)

        # Krzyżowanie jednopunktowe
        nowa_populacja = krzyzowanie_jednopunktowe(population, parents, pop_size, chrom_len, rng)

        # Mutacja bit-flip
        nowa_populacja = mutacja_bitflip(nowa_populacja, p_mut, rng)

        # Elityzm — wstawienie najlepszego osobnika z poprzedniego pokolenia
        if elityzm:
            najlepszy_chrom = population[int(np.argmin(fun_celu))].copy()
            nowa_populacja[0] = najlepszy_chrom

        # Ocena nowej populacji
        population = nowa_populacja
        wagi = dekoduj_chromosom(population, pop_size, num_assets, bits_per_gene)
        fun_celu = np.array([f(wagi[i]) for i in range(pop_size)], dtype=float)

        # Najlepszy w bieżącym pokoleniu
        najlepsze_idx = int(np.argmin(fun_celu))
        najlepsze_val = float(fun_celu[najlepsze_idx])

        # Aktualizacja globalnego optimum
        improved = najlepsze_val < best_fx
        if improved:
            best_fx = najlepsze_val
            best_x = wagi[najlepsze_idx].copy()

        history_f.append(najlepsze_val)
        history_best_f.append(best_fx)
        history_best_w.append(best_x.copy())
        accepted_count.append(int(improved))

    elapsed = time.time() - st

    return {
        "best_weights": best_x,
        "best_objective": best_fx,
        "best_return": float(np.dot(best_x, mu)),
        "best_risk": float(np.sqrt(best_x @ sigma @ best_x)),
        "history_best_w": np.array(history_best_w),
        "history_f": np.array(history_f),
        "history_best_f": np.array(history_best_f),
        "accepted_count": np.array(accepted_count, dtype=int),
        "t_mierz_przebieg": elapsed,
    }


# ─────────────────────────────────────────────────────────────
# WYKRESY
# ─────────────────────────────────────────────────────────────

def plot_convergence(history_f, history_best_f, title, filename='convergence.png'):
    """Wykres zbieżności funkcji celu."""
    plt.figure(figsize=(10, 5))
    plt.plot(history_f, label="najlepszy w pokoleniu", alpha=0.7)
    plt.plot(history_best_f, label="najlepsza dotąd", linewidth=2)
    plt.xlabel("Pokolenie")
    plt.ylabel("Wartość funkcji celu")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres: {filename}")


def plot_weights(weights, title, filename='weights.png'):
    """Wykres wag portfela."""
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
    print(f"Zapisano wykres: {filename}")


def plot_efficient_frontier_with_result(mu, sigma, result, n_points=50, filename='efficient_frontier.png'):
    """
    Wykres granicy efektywności z zaznaczonym wynikiem algorytmu genetycznego.
    Granica efektywności wyznaczana jest metodą losowego próbkowania portfeli
    (Monte Carlo) — bez użycia SciPy.
    """
    n_mc = 20_000
    n = len(mu)
    sigma_np = sigma.values if hasattr(sigma, 'values') else np.array(sigma)
    mu_np = mu.values if hasattr(mu, 'values') else np.array(mu)

    mc_rng = np.random.default_rng(999)
    raw = mc_rng.exponential(1.0, size=(n_mc, n))
    mc_weights = raw / raw.sum(axis=1, keepdims=True)

    mc_returns = mc_weights @ mu_np
    mc_risks = np.sqrt(np.einsum('ij,jk,ik->i', mc_weights, sigma_np, mc_weights))

    # Granica efektywności: dla każdego przedziału zwrotu, minimalne ryzyko
    bins = np.linspace(mc_returns.min(), mc_returns.max(), n_points + 1)
    eff_risks = []
    eff_rets = []
    for i in range(n_points):
        mask = (mc_returns >= bins[i]) & (mc_returns < bins[i + 1])
        if mask.sum() > 0:
            eff_risks.append(mc_risks[mask].min())
            eff_rets.append((bins[i] + bins[i + 1]) / 2)

    plt.figure(figsize=(10, 6))
    plt.scatter(mc_risks, mc_returns, s=1, alpha=0.15, color='steelblue', label='Portfele MC')
    plt.plot(eff_risks, eff_rets, 'b-', linewidth=2.5, label='Granica efektywności (aproks.)')
    plt.scatter(
        result['best_risk'],
        result['best_return'],
        color='red',
        s=120,
        zorder=5,
        label=f'Wynik AG (zwrot={result["best_return"]:.4f}, ryzyko={result["best_risk"]:.4f})'
    )
    plt.xlabel('Ryzyko (odchylenie standardowe)')
    plt.ylabel('Oczekiwany zwrot')
    plt.title('Granica efektywności portfela — algorytm genetyczny')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres: {filename}")


def plot_mutation_comparison(mu, sigma, target_return, penalty, pop_size, bits_per_gene,
                              K, p_muts, filename='mutation_comparison.png'):
    """
    Porównanie wpływu różnych prawdopodobieństw mutacji na zbieżność
    algorytmu genetycznego.
    """
    plt.figure(figsize=(10, 5))
    for seed_offset, p in enumerate(p_muts):
        r = algorytm_genetyczny_portfel(
            mu=mu,
            sigma=sigma,
            target_return=target_return,
            penalty=penalty,
            pop_size=pop_size,
            bits_per_gene=bits_per_gene,
            p_mut=p,
            K=K,
            elityzm=True,
            rng=np.random.default_rng(100 + seed_offset)
        )
        plt.plot(r['history_best_f'], lw=2, label=f"p_mut={p}")

    plt.xlabel("Pokolenie")
    plt.ylabel("Najlepsza wartość funkcji celu")
    plt.title("Wpływ prawdopodobieństwa mutacji na zbieżność AG (portfel)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres: {filename}")


def plot_weight_evolution(history_best_w, title, filename='weight_evolution.png'):
    """
    Wykres ewolucji wag najlepszego osobnika przez kolejne pokolenia.
    Pokazuje jak skład portfela stabilizuje się w trakcie ewolucji.
    """
    plt.figure(figsize=(12, 6))
    K, n = history_best_w.shape
    for j in range(n):
        plt.plot(history_best_w[:, j], alpha=0.7, linewidth=1, label=f"Akcja {j}")
    plt.xlabel("Pokolenie")
    plt.ylabel("Waga")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres: {filename}")


# ─────────────────────────────────────────────────────────────
# GŁÓWNA FUNKCJA URUCHAMIAJĄCA
# ─────────────────────────────────────────────────────────────

def uruchamianie_optymalizacji_portfela():
    """Główna funkcja uruchamiająca optymalizację portfela algorytmem genetycznym."""

    target_return = 0.15

    print(f"\n{'='*60}")
    print(f"OPTYMALIZACJA PORTFELA — ALGORYTM GENETYCZNY")
    print(f"{'='*60}")
    print(f"Docelowy zwrot: {target_return:.4f}")
    print(f"Liczba akcji: {num_stocks}")

    result = algorytm_genetyczny_portfel(
        mu=mu,
        sigma=sigma,
        target_return=target_return,
        penalty=500.0,
        pop_size=150,
        bits_per_gene=16,
        p_mut=0.005,
        K=400,
        elityzm=True,
        rng=rng
    )

    print(f"\n{'='*60}")
    print(f"WYNIKI OPTYMALIZACJI")
    print(f"{'='*60}")
    print(f"Najlepsza wartość funkcji celu: {result['best_objective']:.6f}")
    print(f"Oczekiwany zwrot portfela:      {result['best_return']:.6f}")
    print(f"Ryzyko portfela (odch. std.):   {result['best_risk']:.6f}")
    print(f"Współczynnik Sharpe'a (rf=0):   {result['best_return'] / result['best_risk']:.6f}")
    print(f"Czas obliczeń [s]:              {result['t_mierz_przebieg']:.2f}")

    print(f"\nWagi portfela:")
    for i, w in enumerate(result['best_weights']):
        if w > 0.001:
            print(f"  Akcja {i}: {w:.6f} ({w*100:.2f}%)")

    print(f"\nSuma wag: {np.sum(result['best_weights']):.10f}")
    print(f"Minimalna waga: {np.min(result['best_weights']):.10f}")
    print(f"Liczba akcji w portfelu (waga > 0.1%): {np.sum(result['best_weights'] > 0.001)}")

    improvement_rate = np.mean(result['accepted_count'])
    print(f"\nOdsetek pokoleń z poprawą globalnego optimum: {improvement_rate:.4f}")

    # --- Wykresy ---

    plot_convergence(
        result['history_f'],
        result['history_best_f'],
        "Optymalizacja portfela (AG): zbieżność funkcji celu",
        filename=os.path.join(AG_GRAPHS_DIR, 'portfolio_ag_convergence.png')
    )

    plot_weights(
        result['best_weights'],
        f"Optymalne wagi portfela AG (zwrot={result['best_return']:.4f}, ryzyko={result['best_risk']:.4f})",
        filename=os.path.join(AG_GRAPHS_DIR, 'portfolio_ag_weights.png')
    )

    plot_efficient_frontier_with_result(
        mu, sigma, result, n_points=40,
        filename=os.path.join(AG_GRAPHS_DIR, 'portfolio_ag_efficient_frontier.png')
    )

    plot_weight_evolution(
        result['history_best_w'],
        "Ewolucja wag najlepszego osobnika przez pokolenia",
        filename=os.path.join(AG_GRAPHS_DIR, 'portfolio_ag_weight_evolution.png')
    )

    # --- Porównanie współczynników mutacji ---
    print(f"\n{'='*60}")
    print(f"ANALIZA WRAŻLIWOŚCI: PRAWDOPODOBIEŃSTWO MUTACJI")
    print(f"{'='*60}")

    plot_mutation_comparison(
        mu=mu,
        sigma=sigma,
        target_return=target_return,
        penalty=500.0,
        pop_size=150,
        bits_per_gene=16,
        K=200,
        p_muts=[0.001, 0.005, 0.01, 0.03],
        filename=os.path.join(AG_GRAPHS_DIR, 'portfolio_ag_mutation_comparison.png')
    )

    return result


if __name__ == "__main__":
    result = uruchamianie_optymalizacji_portfela()
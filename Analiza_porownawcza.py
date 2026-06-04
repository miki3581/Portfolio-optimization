"""
Skrypt integruje implementacje Symulowanego Wyżarzania (SA) i Algorytmu
Genetycznego (GA) w jedno środowisko testowe i przeprowadza:
  1. Grid Search po wybranym hiperparametrze każdego algorytmu.
  2. Analizę wrażliwości (5 ziaren losowości) dla najlepszej konfiguracji.
  3. Wykresy zbieżności (skala log) i granicy efektywnej Markowitza.
  4. Zapis wyników liczbowych do pliku Markdown (tabele_wynikow.md).
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importy z modułów projektowych
import data_loader
import markovitz
from Algorytm_SA import symulowane_wyzarzanie_portfel
from Algorytm_genetyczny import algorytm_genetyczny_portfel

# Parametry eksperymentu
RNG_SEED = 16
TARGET_RETURN = 0.15 # docelowa roczna stopa zwrotu
PENALTY = 500.0 # kara za niedostarczenie docelowego zwrotu


# 1. GRID SEARCH — Symulowane Wyżarzanie

def run_sa_grid_search(mu, sigma):
    """
    Przeprowadza grid search dla współczynnika chłodzenia alpha w SA.

    Testowane wartości: [0.80, 0.90, 0.99, 0.995, 0.999]
    Pozostałe hiperparametry są stałe (n_iter=5000, T0=2.0, step_size=0.1).
    """
    alphas = [0.80, 0.90, 0.99, 0.995, 0.999]
    results = {}

    print("\n[Grid Search SA] Testowanie współczynnika chłodzenia alpha ...")
    print(f"  {'alpha':<8} {'f_cel':>12} {'zwrot':>8} {'ryzyko':>8} {'czas [s]':>10}")
    print("  " + "-" * 52)

    for alpha in alphas:
        rng = np.random.default_rng(RNG_SEED)
        t0  = time.time()
        # Tłumimy ew. ostrzeżenia o overflow przy małych temperaturach
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = symulowane_wyzarzanie_portfel(
                mu=mu,
                sigma=sigma,
                target_return=TARGET_RETURN,
                penalty=PENALTY,
                n_iter=5000,
                T0=2.0,
                alpha=alpha,
                step_size=0.1,
                rng=rng,
            )
        duration = time.time() - t0

        results[alpha] = {
            "best_f"       : res["best_objective"],
            "duration"     : duration,
            "history_best_f": res["history_best_f"],
            "best_return"  : res["best_return"],
            "best_risk"    : res["best_risk"],
            "best_weights" : res["best_weights"],
        }
        print(
            f"  {alpha:<8.4f} {res['best_objective']:>12.6f} "
            f"{res['best_return']*100:>7.2f}% {res['best_risk']*100:>7.2f}% "
            f"{duration:>10.2f}"
        )

    return results


# 2. GRID SEARCH — Algorytm Genetyczny

def run_ga_grid_search(mu, sigma):
    """
    Przeprowadza grid search dla wielkości populacji pop_size w GA.

    Testowane wartości: [50, 150, 300]
    Pozostałe hiperparametry: bits_per_gene=16, p_mut=0.005, K=250, elityzm=True.
    """
    pop_sizes = [50, 150, 300]
    results   = {}

    print("\n[Grid Search GA] Testowanie wielkości populacji pop_size ...")
    print(f"  {'pop_size':<10} {'f_cel':>12} {'zwrot':>8} {'ryzyko':>8} {'czas [s]':>10}")
    print("  " + "-" * 54)

    for pop in pop_sizes:
        rng = np.random.default_rng(RNG_SEED)
        res = algorytm_genetyczny_portfel(
            mu=mu,
            sigma=sigma,
            target_return=TARGET_RETURN,
            penalty=PENALTY,
            pop_size=pop,
            bits_per_gene=16,
            p_mut=0.005,
            K=250,
            elityzm=True,
            rng=rng,
        )
        results[pop] = {
            "best_f"        : res["best_objective"],
            "duration"      : res["t_mierz_przebieg"],
            "history_best_f": res["history_best_f"],
            "best_return"   : res["best_return"],
            "best_risk"     : res["best_risk"],
            "best_weights"  : res["best_weights"],
        }
        print(
            f"  {pop:<10} {res['best_objective']:>12.6f} "
            f"{res['best_return']*100:>7.2f}% {res['best_risk']*100:>7.2f}% "
            f"{res['t_mierz_przebieg']:>10.2f}"
        )

    return results


# 3. ANALIZA WRAŻLIWOŚCI

def run_sensitivity_analysis(mu, sigma, sa_alpha, ga_pop, n_seeds=5):
    """
    Uruchamia oba algorytmy na n_seeds różnych ziarnach losowości,
    aby ocenić stabilność rozwiązań dla najlepszych znalezionych parametrów.

    Parametry
    ---------
    sa_alpha : float  — najlepszy współczynnik chłodzenia SA z grid search
    ga_pop   : int    — najlepsza wielkość populacji GA z grid search
    n_seeds  : int    — liczba niezależnych uruchomień (domyślnie 5)
    """
    seeds  = [16, 123, 234, 2026, 9999][:n_seeds]
    sa_fs  = []
    ga_fs  = []

    print(
        f"\n[Analiza Wrażliwości] SA(alpha={sa_alpha}) vs GA(pop={ga_pop}), "
        f"{n_seeds} ziaren losowości ..."
    )

    for seed in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res_sa = symulowane_wyzarzanie_portfel(
                mu=mu, sigma=sigma,
                target_return=TARGET_RETURN, penalty=PENALTY,
                n_iter=5000, T0=2.0, alpha=sa_alpha, step_size=0.1,
                rng=np.random.default_rng(seed),
            )
        sa_fs.append(res_sa["best_objective"])

        res_ga = algorytm_genetyczny_portfel(
            mu=mu, sigma=sigma,
            target_return=TARGET_RETURN, penalty=PENALTY,
            pop_size=ga_pop, bits_per_gene=16, p_mut=0.005, K=250, elityzm=True,
            rng=np.random.default_rng(seed),
        )
        ga_fs.append(res_ga["best_objective"])

    summary = {
        "SA": {
            "mean" : float(np.mean(sa_fs)),
            "std"  : float(np.std(sa_fs)),
            "best" : float(np.min(sa_fs)),
            "worst": float(np.max(sa_fs)),
        },
        "GA": {
            "mean" : float(np.mean(ga_fs)),
            "std"  : float(np.std(ga_fs)),
            "best" : float(np.min(ga_fs)),
            "worst": float(np.max(ga_fs)),
        },
    }

    print(f"  SA  — śr.: {summary['SA']['mean']:.6f}  std: {summary['SA']['std']:.6f}"
          f"  best: {summary['SA']['best']:.6f}  worst: {summary['SA']['worst']:.6f}")
    print(f"  GA  — śr.: {summary['GA']['mean']:.6f}  std: {summary['GA']['std']:.6f}"
          f"  best: {summary['GA']['best']:.6f}  worst: {summary['GA']['worst']:.6f}")

    return summary


# 4. WYKRESY PORÓWNAWCZE

def generate_comparison_plots(mu, sigma, sa_results, ga_results):
    """
    Tworzy dwa wykresy:
      (a) Zbieżność funkcji celu w skali logarytmicznej (porownanie_zbieznosci.png)
      (b) Ryzyko–Zwrot z granicami efektywnymi i wynikami SA/GA (porownanie_granica.png)
    """
    # wybieramy najlepsze przebiegi z grid search
    best_sa_alpha = min(sa_results, key=lambda k: sa_results[k]["best_f"])
    best_ga_pop   = min(ga_results, key=lambda k: ga_results[k]["best_f"])

    # a) Wykres zbieżności
    fig, ax = plt.subplots(figsize=(10, 6))

    sa_hist = sa_results[best_sa_alpha]["history_best_f"]
    ga_hist = ga_results[best_ga_pop]["history_best_f"]

    # Normalizujemy oś X do [0, 1], żeby przebiegi były porównywalne wizualnie
    ax.plot(
        np.linspace(0, 1, len(sa_hist)), sa_hist,
        label=f"SA (α = {best_sa_alpha})", linewidth=2, color="#d95f02",
    )
    ax.plot(
        np.linspace(0, 1, len(ga_hist)), ga_hist,
        label=f"GA (pop_size = {best_ga_pop})", linewidth=2, color="#7570b3",
    )

    ax.set_yscale("log")
    ax.set_title(
        "Porównanie zbieżności algorytmów optymalizacji",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Znormalizowany postęp (0 = start, 1 = koniec)", fontsize=10)
    ax.set_ylabel("Wartość funkcji celu (skala log)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    import os
    os.makedirs('graphs/compare', exist_ok=True)
    
    fig.tight_layout()
    fig.savefig("graphs/compare/porownanie_zbieznosci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Wykresy] Zapisano: graphs/compare/porownanie_zbieznosci.png")

    # b) Wykres granic efektywnych
    print("[Wykresy] Wyznaczanie granicy numerycznej Markowitza (SciPy SLSQP) ...")

    min_var_res = markovitz.minimum_variance()
    min_var_ret = markovitz.return_(min_var_res.x, mu)

    # Zakres stóp zwrotu do wyznaczenia granic efektywnych
    target_returns = np.linspace(
        max(min_var_ret - 0.05, mu.min() - 0.02),
        mu.max() + 0.05,
        50,
    )

    # Numeryczna granica (scipy SLSQP, w_i >= 0)
    num_risks, num_rets = [], []
    for r in target_returns:
        opt = markovitz.efficient_return(r)
        if opt.success:
            num_risks.append(opt.fun)
            num_rets.append(r)

    # Analityczna granica (Lagrange, krótka sprzedaż dozwolona)
    ana_risks = markovitz.analytical_efficient_frontier(mu, sigma, target_returns)

    # Wyniki SA i GA
    sa_best = sa_results[best_sa_alpha]
    ga_best = ga_results[best_ga_pop]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        num_risks, num_rets,
        "-", linewidth=2.5, color="#1b9e77",
        label="Granica numeryczna (Markowitz, w ≥ 0)",
    )
    ax.plot(
        ana_risks, target_returns,
        "--", linewidth=2.0, color="#666666",
        label="Granica analityczna Lagrange'a (krótka sprzedaż dozwolona)",
    )

    # Indywidualne spółki
    ax.scatter(
        np.sqrt(np.diag(sigma)), mu,
        color="#a6d854", alpha=0.6, s=40, zorder=3,
        label="Spółki WIG15",
    )

    # Wyniki metaheurystyk
    sharpe_sa = sa_best["best_return"] / sa_best["best_risk"]
    sharpe_ga = ga_best["best_return"] / ga_best["best_risk"]

    ax.scatter(
        sa_best["best_risk"], sa_best["best_return"],
        color="#d95f02", marker="o", s=140, zorder=6,
        label=f"SA (α={best_sa_alpha}, Sharpe={sharpe_sa:.2f})",
    )
    ax.scatter(
        ga_best["best_risk"], ga_best["best_return"],
        color="#7570b3", marker="s", s=120, zorder=6,
        label=f"GA (pop={best_ga_pop}, Sharpe={sharpe_ga:.2f})",
    )

    ax.axhline(
        y=TARGET_RETURN, color="#e7298a", linestyle=":", linewidth=1.5,
        label=f"Stopa docelowa ({TARGET_RETURN*100:.0f}%)",
    )

    ax.set_title(
        "Granice efektywne portfela i wyniki metaheurystyk",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Ryzyko portfela (odchylenie standardowe, roczne)", fontsize=10)
    ax.set_ylabel("Oczekiwana roczna stopa zwrotu", fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig("graphs/compare/porownanie_granica.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Wykresy] Zapisano: graphs/compare/porownanie_granica.png")


# 5. ZAPIS WYNIKÓW DO MARKDOWN

def save_markdown_tables(sa_results, ga_results, sens_results):
    """
    Tworzy plik tabele_wynikow.md z tabelami Markdown zawierającymi:
      - wyniki grid search SA i GA,
      - wyniki analizy wrażliwości,
      - optymalne wagi portfeli.
    """
    best_sa_alpha = min(sa_results, key=lambda k: sa_results[k]["best_f"])
    best_ga_pop   = min(ga_results, key=lambda k: ga_results[k]["best_f"])

    sa_weights = sa_results[best_sa_alpha]["best_weights"]
    ga_weights = ga_results[best_ga_pop]["best_weights"]

    # Nazwy spółek z indeksu DataFrame (z data_loader)
    mu_series = data_loader.mu
    tickers   = list(mu_series.index)

    lines = []
    lines.append("# Raport Wyników Optymalizacji Portfela WIG15\n")
    lines.append(
        "_Wygenerowano automatycznie przez `Analiza_porownawcza.py`._\n"
    )
    lines.append(
        f"> **Parametry eksperymentu:** TARGET_RETURN = {TARGET_RETURN*100:.0f}%,"
        f" PENALTY = {PENALTY}, RNG_SEED = {RNG_SEED}\n"
    )

    # Tabela 1: Grid Search SA
    lines.append("## 1. Grid Search: Symulowane Wyżarzanie (SA)\n")
    lines.append(
        "Testowano wpływ współczynnika chłodzenia $\\alpha$ przy stałej liczbie 5000 iteracji,"
        " T₀ = 2,0, step_size = 0,1.\n"
    )
    lines.append("| α | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for alpha, r in sa_results.items():
        marker = " ★" if alpha == best_sa_alpha else ""
        lines.append(
            f"| {alpha:.4f}{marker} | {r['best_f']:.6f} "
            f"| {r['best_return']*100:.2f}% | {r['best_risk']*100:.2f}% "
            f"| {r['duration']:.3f} |"
        )
    lines.append("")

    # Tabela 2: Grid Search GA
    lines.append("## 2. Grid Search: Algorytm Genetyczny (GA)\n")
    lines.append(
        "Testowano wpływ wielkości populacji $pop\\_size$ przy 250 pokoleniach,"
        " bits_per_gene = 16, p_mut = 0,005.\n"
    )
    lines.append("| pop_size | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for pop, r in ga_results.items():
        marker = " ★" if pop == best_ga_pop else ""
        lines.append(
            f"| {pop}{marker} | {r['best_f']:.6f} "
            f"| {r['best_return']*100:.2f}% | {r['best_risk']*100:.2f}% "
            f"| {r['duration']:.3f} |"
        )
    lines.append("")

    # Tabela 3: Analiza wrażliwości
    lines.append("## 3. Analiza wrażliwości (5 niezależnych ziaren losowości)\n")
    lines.append(
        "Walidacja stabilności najlepszych konfiguracji na 5 różnych seedach.\n"
    )
    lines.append("| Algorytm | Śr. F | Std Dev F | Najlepsza F | Najgorsza F |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for method, m in sens_results.items():
        lines.append(
            f"| {method} | {m['mean']:.6f} | {m['std']:.6f} "
            f"| {m['best']:.6f} | {m['worst']:.6f} |"
        )
    lines.append("")

    # Tabela 4: Optymalne wagi portfeli
    lines.append("## 4. Optymalne wagi portfeli\n")
    lines.append(
        f"Zestawienie alokacji kapitału dla najlepszych konfiguracji:"
        f" SA (α = {best_sa_alpha}) i GA (pop_size = {best_ga_pop}).\n"
    )
    lines.append(
        f"| # | Spółka | Waga SA (α={best_sa_alpha}) | Waga GA (pop={best_ga_pop}) |"
    )
    lines.append("| :--- | :--- | ---: | ---: |")
    for i, tick in enumerate(tickers):
        lines.append(
            f"| {i+1} | **{tick}** | {sa_weights[i]*100:.2f}% | {ga_weights[i]*100:.2f}% |"
        )
    lines.append("")

    # Podsumowanie
    sa_best = sa_results[best_sa_alpha]
    ga_best = ga_results[best_ga_pop]
    lines.append("## 5. Podsumowanie\n")
    lines.append("| Miara | SA | GA |")
    lines.append("| :--- | ---: | ---: |")
    lines.append(
        f"| Najlepsza funkcja celu | {sa_best['best_f']:.6f} | {ga_best['best_f']:.6f} |"
    )
    lines.append(
        f"| Stopa zwrotu | {sa_best['best_return']*100:.2f}% | {ga_best['best_return']*100:.2f}% |"
    )
    lines.append(
        f"| Ryzyko | {sa_best['best_risk']*100:.2f}% | {ga_best['best_risk']*100:.2f}% |"
    )
    lines.append(
        f"| Współczynnik Sharpe'a | "
        f"{sa_best['best_return']/sa_best['best_risk']:.4f} | "
        f"{ga_best['best_return']/ga_best['best_risk']:.4f} |"
    )
    lines.append(
        f"| Stabilność (Std Dev, 5 seedów) | "
        f"{sens_results['SA']['std']:.6f} | {sens_results['GA']['std']:.6f} |"
    )

    content = "\n".join(lines) + "\n"

    with open("tabele_wynikow.md", "w", encoding="utf-8") as fh:
        fh.write(content)

    print("[Markdown] Zapisano wyniki do: tabele_wynikow.md")


# MAIN

if __name__ == "__main__":
    print("=" * 70)
    print("  ZINTEGROWANA ANALIZA PORÓWNAWCZA — PORTFEL WIG15")
    print("  Rola: Integrator i Analityk (Osoba 4)")
    print("=" * 70)

    mu    = data_loader.mu
    sigma = data_loader.sigma
    n     = len(mu)
    print(f"\n  Liczba aktywów: {n}")
    print(f"  Docelowa stopa zwrotu: {TARGET_RETURN*100:.0f}%")
    print(f"  Kara za shortfall: {PENALTY}")

    # 1. Grid Search SA
    sa_res = run_sa_grid_search(mu, sigma)

    # 2. Grid Search GA
    ga_res = run_ga_grid_search(mu, sigma)

    # Wybór najlepszych parametrów
    best_sa_alpha = min(sa_res, key=lambda k: sa_res[k]["best_f"])
    best_ga_pop   = min(ga_res, key=lambda k: ga_res[k]["best_f"])
    print(f"\n  Najlepszy alpha SA : {best_sa_alpha}")
    print(f"  Najlepszy pop_size GA : {best_ga_pop}")

    # 3. Analiza Wrażliwości
    sens_res = run_sensitivity_analysis(mu, sigma, best_sa_alpha, best_ga_pop)

    # 4. Wykresy porównawcze
    generate_comparison_plots(mu, sigma, sa_res, ga_res)

    # 5. Zapis tabel Markdown
    save_markdown_tables(sa_res, ga_res, sens_res)

    print("\n" + "=" * 70)
    print("  ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("  Wygenerowane pliki:")
    print("    - graphs/compare/porownanie_zbieznosci.png")
    print("    - graphs/compare/porownanie_granica.png")
    print("    - tabele_wynikow.md")
    print("=" * 70)

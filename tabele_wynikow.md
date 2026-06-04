# Raport Wyników Optymalizacji Portfela WIG15

_Wygenerowano automatycznie przez `Analiza_porownawcza.py`._

> **Parametry eksperymentu:** TARGET_RETURN = 15%, PENALTY = 500.0, RNG_SEED = 16

## 1. Grid Search: Symulowane Wyżarzanie (SA)

Testowano wpływ współczynnika chłodzenia $\alpha$ przy stałej liczbie 5000 iteracji, T₀ = 2,0, step_size = 0,1.

| α | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |
| :--- | ---: | ---: | ---: | ---: |
| 0.8000 | 0.034818 | 19.92% | 18.66% | 0.628 |
| 0.9000 ★ | 0.034787 | 19.11% | 18.65% | 0.639 |
| 0.9900 | 0.034825 | 19.51% | 18.66% | 0.652 |
| 0.9950 | 0.034794 | 19.61% | 18.65% | 0.641 |
| 0.9990 | 0.038436 | 22.69% | 19.61% | 0.646 |

## 2. Grid Search: Algorytm Genetyczny (GA)

Testowano wpływ wielkości populacji $pop\_size$ przy 250 pokoleniach, bits_per_gene = 16, p_mut = 0,005.

| pop_size | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |
| :--- | ---: | ---: | ---: | ---: |
| 50 | 0.035428 | 19.93% | 18.82% | 1.397 |
| 150 ★ | 0.035123 | 19.73% | 18.74% | 4.046 |
| 300 | 0.035561 | 19.13% | 18.86% | 8.136 |

## 3. Analiza wrażliwości (5 niezależnych ziaren losowości)

Walidacja stabilności najlepszych konfiguracji na 5 różnych seedach.

| Algorytm | Śr. F | Std Dev F | Najlepsza F | Najgorsza F |
| :--- | ---: | ---: | ---: | ---: |
| SA | 0.034803 | 0.000011 | 0.034787 | 0.034821 |
| GA | 0.035239 | 0.000213 | 0.035019 | 0.035628 |

## 4. Optymalne wagi portfeli

Zestawienie alokacji kapitału dla najlepszych konfiguracji: SA (α = 0.9) i GA (pop_size = 150).

| # | Spółka | Waga SA (α=0.9) | Waga GA (pop=150) |
| :--- | :--- | ---: | ---: |
| 1 | **ALE.WA** | 16.71% | 16.06% |
| 2 | **ALR.WA** | 0.00% | 0.34% |
| 3 | **BDX.WA** | 9.17% | 9.10% |
| 4 | **CDR.WA** | 7.75% | 10.02% |
| 5 | **DNP.WA** | 13.86% | 12.59% |
| 6 | **KGH.WA** | 0.00% | 1.01% |
| 7 | **LPP.WA** | 2.22% | 1.12% |
| 8 | **MBK.WA** | 0.00% | 0.29% |
| 9 | **PEO.WA** | 0.00% | 0.24% |
| 10 | **PGE.WA** | 0.00% | 1.51% |
| 11 | **PKN.WA** | 23.14% | 21.26% |
| 12 | **PKO.WA** | 0.23% | 1.32% |
| 13 | **PZU.WA** | 21.60% | 20.66% |
| 14 | **SPL.WA** | 0.00% | 0.15% |
| 15 | **TPE.WA** | 5.30% | 4.33% |

## 5. Podsumowanie

| Miara | SA | GA |
| :--- | ---: | ---: |
| Najlepsza funkcja celu | 0.034787 | 0.035123 |
| Stopa zwrotu | 19.11% | 19.73% |
| Ryzyko | 18.65% | 18.74% |
| Współczynnik Sharpe'a | 1.0243 | 1.0528 |
| Stabilność (Std Dev, 5 seedów) | 0.000011 | 0.000213 |

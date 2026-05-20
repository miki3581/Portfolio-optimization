# Raport z części 4

## 1. Cel i zakres prac

Zgodnie z podziałem obowiązków, główne zadania obejmowały:

1. **Zintegrowanie** implementacji Symulowanego Wyżarzania (SA) i Algorytmu Genetycznego (GA) w jedno spójne środowisko testowe.
2. **Przeprowadzenie analizy porównawczej (Grid Search)** dla wybranych hiperparametrów obu metod:
   - SA: współczynnik chłodzenia $\alpha \in \{0{,}80,\ 0{,}90,\ 0{,}99,\ 0{,}995,\ 0{,}999\}$
   - GA: wielkość populacji $pop\_size \in \{50,\ 150,\ 300\}$
3. **Zaimplementowanie benchmarku klasycznego** — zarówno numerycznej (scipy SLSQP, $w_i \ge 0$), jak i analitycznej (mnożniki Lagrange'a, krótka sprzedaż dozwolona) granicy efektywnej.
4. **Przeprowadzenie analizy wrażliwości** badającej stabilność algorytmów względem ziarna losowości (5 niezależnych uruchomień: seedy `{16, 123, 234, 2026, 9999}`).
5. **Przygotowanie wizualizacji** wyników (wykresy zbieżności i granicy efektywnej).
6. **Zapis wyników liczbowych** do pliku `tabele_wynikow.md`.

Wszystkie zadania zostały zrealizowane w skrypcie `Analiza_porownawcza.py`.

---

## 2. Opis implementacji (`Analiza_porownawcza.py`)

### 2.1. Integracja modułów

Skrypt importuje funkcje optymalizacyjne bezpośrednio z plików grupy:

| Moduł | Importowana funkcja | Opis |
| :--- | :--- | :--- |
| `Algorytm_SA.py` | `symulowane_wyzarzanie_portfel` | SA z rzutowaniem na sympleks |
| `Algorytm_genetyczny.py` | `algorytm_genetyczny_portfel` | GA z kodowaniem binarnym |
| `data_loader.py` | `mu`, `sigma` | Annualizowane dane WIG15 |
| `markovitz.py` | `minimum_variance`, `efficient_return`, `analytical_efficient_frontier` | Benchmark Markowitza (SciPy) |

Oba algorytmy przyjmują ustandaryzowany interfejs: `(mu, sigma, target_return, penalty, ..., rng)`. Ziarno losowości (`rng`) jest zawsze przekazywane jawnie — gwarantuje to **powtarzalność wyników**.

### 2.2. Benchmark klasyczny (Granica efektywna)

Na wykresie ryzyko–zwrot naniesione są dwie granice efektywne z modułu `markovitz.py`:

- **Numeryczna** (scipy SLSQP, $w_i \ge 0$) — realistyczny benchmark bez krótkiej sprzedaży.
- **Analityczna** (metoda mnożników Lagrange'a, $\Sigma^{-1}$ wyznaczane przez `numpy.linalg.solve`) — granica teoretyczna, dopuszczająca krótką sprzedaż; stanowi dolne ograniczenie ryzyka.

### 2.3. Grid Search (Strojenie Hiperparametrów)

#### Symulowane wyżarzanie — współczynnik chłodzenia $\alpha$

Testowane wartości: $\alpha \in \{0{,}80,\ 0{,}90,\ 0{,}99,\ 0{,}995,\ 0{,}999\}$.  
Stałe parametry: `n_iter = 5000`, `T0 = 2,0`, `step_size = 0,1`, `RNG_SEED = 16`.

| $\alpha$ | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |
| :--- | ---: | ---: | ---: | ---: |
| 0.8000 | 0.034818 | 19.92% | 18.66% | 0.611 |
| **0.9000 ★** | **0.034787** | **19.11%** | **18.65%** | **0.613** |
| 0.9900 | 0.034825 | 19.51% | 18.66% | 0.607 |
| 0.9950 | 0.034794 | 19.61% | 18.65% | 0.614 |
| 0.9990 | 0.038436 | 22.69% | 19.61% | 0.622 |

Zaobserwowane zależności:
- Zbyt szybkie chłodzenie ($\alpha \le 0{,}80$) sprawia, że algorytm traci stochastyczny charakter jeszcze przed przeszukaniem przestrzeni — skutkuje to gorszą jakością rozwiązania.
- Najlepszy wynik uzyskano dla $\alpha = 0{,}90$, choć różnice między wartościami 0,80–0,995 są niewielkie (rzędu $10^{-5}$), co świadczy o względnej odporności algorytmu w tym zakresie.
- Wyraźnie gorszy wynik dla $\alpha = 0{,}999$ (0.038436 vs ~0.03482) wskazuje, że zbyt wolne chłodzenie uniemożliwia efektywną intensyfikację — algorytm „błądzi" do końca bez wyraźnej konwergencji.

#### Algorytm genetyczny — wielkość populacji $pop\_size$

Testowane wartości: $pop\_size \in \{50,\ 150,\ 300\}$.  
Stałe parametry: `K = 250`, `bits_per_gene = 16`, `p_mut = 0,005`, elityzm włączony.

| $pop\_size$ | Funkcja celu | Stopa zwrotu | Ryzyko | Czas [s] |
| :--- | ---: | ---: | ---: | ---: |
| 50 | 0.035428 | 19.93% | 18.82% | 1.349 |
| **150 ★** | **0.035123** | **19.73%** | **18.74%** | **3.967** |
| 300 | 0.035561 | 19.13% | 18.86% | 8.002 |

Zaobserwowane zależności:
- Mała populacja (`pop_size = 50`) daje gorszy wynik z powodu ograniczonej różnorodności genetycznej i ryzyka przedwczesnej konwergencji.
- Najlepszy wynik uzyskała populacja 150 osobników — stanowi dobry kompromis między czasem obliczeń a jakością.
- Większa populacja (`pop_size = 300`) nie poprawiła wyniku przy 250 pokoleniach — prawdopodobnie potrzebuje więcej generacji do pełnego wykorzystania swojej różnorodności.

### 2.4. Analiza wrażliwości

Funkcja `run_sensitivity_analysis()` uruchamia najlepsze konfiguracje SA ($\alpha = 0{,}90$) i GA ($pop\_size = 150$) na 5 ziarnach losowości (`{16, 123, 234, 2026, 9999}`). Raportowane są: średnia, odchylenie standardowe, najlepszy i najgorszy wynik funkcji celu.

| Algorytm | Śr. F | Std Dev F | Najlepsza F | Najgorsza F |
| :--- | ---: | ---: | ---: | ---: |
| SA | 0.034803 | 0.000011 | 0.034787 | 0.034821 |
| GA | 0.035239 | 0.000213 | 0.035019 | 0.035628 |

### 2.5. Wykresy (`generate_comparison_plots`)

#### `porownanie_zbieznosci.png`

Wykres liniowy w **skali logarytmicznej** prezentujący ewolucję najlepszej dotąd wartości funkcji celu. Oś X jest **znormalizowana do [0, 1]**, co pozwala na wizualne porównanie algorytmów o różnej granulacji (SA: 5001 iteracji, GA: 250 pokoleń).

#### `porownanie_granica.png`

Klasyczny wykres ryzyko–zwrot (płaszczyzna $\sigma$–$\mu$) zawierający:
- numeryczną granicę efektywną (zielona linia ciągła),
- analityczną granicę Lagrange'a (szara linia przerywana),
- indywidualne spółki WIG15 (małe zielone punkty),
- najlepsze portfele SA i GA z wartością współczynnika Sharpe'a,
- poziomą linię docelowej stopy zwrotu (15%).

---

## 3. Pliki wyjściowe

| Plik | Opis |
| :--- | :--- |
| `porownanie_zbieznosci.png` | Zbieżność funkcji celu (skala log, znormalizowana oś X) |
| `porownanie_granica.png` | Granice efektywne i wyniki SA/GA na płaszczyźnie ryzyko–zwrot |
| `tabele_wynikow.md` | Tabele liczbowe z wynikami grid search, analizy wrażliwości i optymalnych wag |

---

## 4. Główne wnioski z analizy

1. **Wpływ harmonogramu chłodzenia (SA):** W badanym zakresie ($\alpha \in [0{,}80,\ 0{,}995]$) różnice w jakości rozwiązania są minimalne (rzędu $10^{-5}$), co świadczy o odporności SA na dokładny dobór tempa chłodzenia — o ile mieści się ono w rozsądnym przedziale. Zdecydowanie gorszy wynik dla $\alpha = 0{,}999$ wskazuje, że zbyt wolne chłodzenie jest poważnym problemem.

2. **Efektywność SA vs. GA:** Symulowane Wyżarzanie z rzutowaniem na sympleks wykazuje znaczną przewagę — zarówno szybkościową (poniżej 1 sekundy vs ~4 sekundy dla GA 150), jak i jakościową (najlepsza F: 0.034787 vs 0.035123).

3. **Stabilność:** SA jest znacznie bardziej stabilny (std = 0.000011) niż GA (std = 0.000213). Duży rozrzut GA wynika ze stochastycznych operatorów genetycznych i binarnego kodowania — wynik zależy silniej od konkretnego przebiegu.

4. **Jakość portfeli:** Oba algorytmy zbliżają się do numerycznej granicy efektywnej Markowitza. SA osiąga współczynnik Sharpe'a 1.0243, GA — 1.0528. Warto odnotować, że mimo gorszej wartości funkcji celu, GA uzyskuje nieznacznie wyższy Sharpe, co wynika z nieco wyższej stopy zwrotu przy zbliżonym ryzyku.

5. **Zalecenie:** W tym wariancie problemu Markowitza (15 aktywów WIG15, $w_i \ge 0$, cel 15%) rekomendowane jest **Symulowane Wyżarzanie ($\alpha = 0{,}90$)** — oferuje lepszy wynik funkcji celu, jest wielokrotnie szybsze i charakteryzuje się znacznie wyższą stabilnością wyników między uruchomieniami.

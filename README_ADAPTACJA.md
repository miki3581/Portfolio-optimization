# Adaptacja Symulowanego Wyżarzania do Optymalizacji Portfela

## Opis

Ten skrypt (`NMO_3_adaptation.py`) stanowi adaptację algorytmu symulowanego wyżarzania z pliku `NMO_3_Symulowane_wyzarzanie (1).txt` do problemu optymalizacji portfela inwestycyjnego.

## Główne Zmiany w Stosunku do Oryginału

### 1. Funkcja Celu z Karą (Shortfall)

Zamiast prostej funkcji celu, zaimplementowano funkcję z karą za nieosiągnięcie docelowego zwrotu:

```python
objective = portfolio_variance + penalty * max(0, target_return - portfolio_return)
```

- **Minimalizujemy wariancję portfela** (ryzyko)
- **Kara za shortfall**: jeśli oczekiwany zwrot jest niższy niż docelowy, dodawana jest kara
- Parametr `penalty` (domyślnie 500.0) kontroluje wagę kary

### 2. Rzutowanie na Sympleks

Kluczowa różnica w stosunku do oryginalnego skryptu - zamiast prostego przycinania (clipping) do granic, wagi są rzutowane na sympleks:

```python
def project_simplex(w):
    """
    Projekcja na sympleks: sum(w) = 1, w >= 0
    Algorytm Duchi et al. 2008
    """
```

**Ograniczenia sympleksu:**
- Suma wag = 1 (całość kapitału zainwestowana)
- Wszystkie wagi >= 0 (bez krótkiej sprzedaży)

### 3. Generator Propozycji

Propozycje nowych rozwiązań generowane są przez:
1. Dodanie losowej perturbacji gaussowskiej
2. Rzutowanie na sympleks

```python
perturbation = local_scale * rng.normal(size=num_assets)
candidate = x + perturbation
candidate = project_simplex(candidate)
```

Skala perturbacji maleje wraz z temperaturą (adaptive step size).

## Parametry Algorytmu

### Parametry Optymalizacji
- `target_return`: Docelowy zwrot portfela (domyślnie 0.15 = 15%)
- `penalty`: Kara za shortfall (domyślnie 500.0)
- `n_iter`: Liczba iteracji (domyślnie 8000)

### Parametry Symulowanego Wyżarzania
- `T0`: Temperatura początkowa (domyślnie 2.0)
- `alpha`: Parametr chłodzenia geometrycznego (domyślnie 0.9985)
- `step_size`: Skala kroku dla propozycji (domyślnie 0.15)

## Wyniki

Skrypt generuje następujące wyjścia:

### Konsola
- Liczba akcji w portfelu
- Najlepsza wartość funkcji celu
- Oczekiwany zwrot portfela
- Ryzyko portfela (odchylenie standardowe)
- Współczynnik Sharpe'a
- Wagi wszystkich akcji (powyżej 0.1%)
- Suma wag (weryfikacja = 1.0)
- Współczynnik akceptacji

### Wykresy (pliki PNG)
1. `portfolio_convergence.png` - Zbieżność funkcji celu
2. `portfolio_temperature.png` - Harmonogram temperatury
3. `portfolio_weights.png` - Wagi portfela (wykres słupkowy)
4. `portfolio_efficient_frontier.png` - Granica efektywności z zaznaczonym wynikiem SA

## Uruchomienie

```bash
python NMO_3_adaptation.py
```

## Zależności

- `numpy` - operacje numeryczne
- `pandas` - wczytywanie danych
- `matplotlib` - generowanie wykresów
- `scipy` - optymalizacja (do wykresu granicy efektywności)

## Przykładowy Wynik

```
============================================================
WYNIKI OPTYMALIZACJI
============================================================
Najlepsza wartość funkcji celu: 0.034857
Oczekiwany zwrot portfela: 0.184266
Ryzyko portfela (odch. std.): 0.186700
Współczynnik Sharpe'a (przy rf=0): 0.986965

Wagi portfela:
  Akcja 0: 0.172618 (17.26%)
  Akcja 1: 0.002663 (0.27%)
  Akcja 2: 0.099280 (9.93%)
  Akcja 3: 0.085652 (8.57%)
  Akcja 4: 0.148602 (14.86%)
  Akcja 6: 0.007921 (0.79%)
  Akcja 10: 0.232512 (23.25%)
  Akcja 12: 0.207536 (20.75%)
  Akcja 14: 0.043217 (4.32%)

Suma wag: 1.0000000000
```

## Struktura Folderów

```
.
├── NMO_3_adaptation.py                    # Główny skrypt
├── NMO_3_Symulowane_wyzarzanie (1).txt   # Oryginalny skrypt
├── Portfolio-optimization/
│   ├── dane.csv                           # Dane giełdowe
│   ├── data_loader.py                     # Loader danych
│   └── markovitz.py                       # Porównanie - metoda Markowitza
├── portfolio_convergence.png              # Wykres zbieżności
├── portfolio_temperature.png              # Wykres temperatury
├── portfolio_weights.png                  # Wykres wag
└── portfolio_efficient_frontier.png       # Granica efektywności
```

## Porównanie z Metodą Markowitza

Skrypt w folderze `Portfolio-optimization/markovitz.py` implementuje klasyczną metodę Markowitza używając `scipy.optimize`. Można porównać wyniki obu metod:

- **Markowitz (SLSQP)**: Optymalizacja deterministyczna, gwarancja optimum lokalnego
- **Symulowane Wyżarzanie**: Optymalizacja stochastyczna, lepsza eksploracja przestrzeni, możliwość ucieczki z minimów lokalnych

## Uwagi Techniczne

### Algorytm Projekcji na Sympleks
Wykorzystano algorytm Duchi et al. (2008), który znajduje optymalną projekcję wektora na sympleks jednostkowy w czasie O(n log n).

### Harmonogram Chłodzenia
Zastosowano geometryczne chłodzenie: T(k) = T0 * alpha^k

### Adaptacyjna Skala Kroku
Skala kroku propozycji jest adaptacyjna: 
`local_scale = step_size * (0.2 + 0.8 * T/T0)`

To zapewnia większą eksplorację na początku i dokładniejsze przeszukiwanie pod koniec.

## Możliwe Rozszerzenia

1. **Wielokryterialna optymalizacja**: Dodać inne cele (np. VaR, CVaR)
2. **Ograniczenia na wagi**: Dodać minimum/maksimum na pojedyncze wagi
3. **Koszty transakcyjne**: Uwzględnić koszty realokacji portfela
4. **Różne harmonogramy chłodzenia**: Testować liniowe, logarytmiczne
5. **Porównanie z innymi metodami**: PSO, algorytmy genetyczne
6. **Backtesting**: Testowanie strategii na danych historycznych

## Autor
Adaptacja na podstawie materiałów z kursu "Nieklasyczne Metody Optymalizacji", SGH

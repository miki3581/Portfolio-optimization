# Optymalizacja Portfela Inwestycyjnego (WIG15)

Projekt realizowany w ramach przedmiotu **Nieklasyczne Metody Optymalizacji (NMO)**. Celem projektu jest optymalizacja wag portfela inwestycyjnego składającego się z 15 spółek indeksu WIG15 przy użyciu algorytmów heurystycznych oraz porównanie ich z klasycznym modelem Markowitza.

## 🚀 Główne cechy

- **Algorytm genetyczny (GA):** Implementacja z kodowaniem binarnym wag i elitaryzmem.
- **Symulowane wyżarzanie (SA):** Implementacja z rzutowaniem na sympleks (zapewnienie sumy wag = 1) i geometrycznym schematem chłodzenia.
- **Model Markowitza:** Benchmark klasyczny (numeryczny i analityczny) wyznaczający granicę efektywną (Efficient Frontier).
- **Analiza porównawcza:** Automatyczny Grid Search parametrów, analiza wrażliwości na ziarno losowości oraz zaawansowane wizualizacje wyników.
- **Dane realne:** Automatyczne pobieranie danych giełdowych z Yahoo Finance.

## 📁 Struktura projektu

- `Analiza_porownawcza.py` – Główny skrypt integrujący wszystkie moduły i generujący raporty.
- `Algorytm_SA.py` – Implementacja algorytmu symulowanego wyżarzania.
- `Algorytm_genetyczny.py` – Implementacja algorytmu genetycznego.
- `markovitz.py` – Narzędzia do klasycznej optymalizacji średnia-wariancja.
- `downloader.py` – Skrypt pobierający dane z Yahoo Finance (WIG15).
- `data_loader.py` – Moduł przetwarzający surowe dane na stopy zwrotu i macierz kowariancji.
- `dane.csv` – Pobrane dane historyczne cen zamknięcia.
- `tabele_wynikow.md` – Szczegółowe wyniki liczbowe eksperymentów.
- `*.png` – Wygenerowane wizualizacje (zbieżność, granica efektywna, ewolucja wag).

## 🛠️ Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/TwojUser/Portfolio-optimization.git
   cd Portfolio-optimization
   ```

2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install numpy pandas matplotlib yfinance scipy
   ```

## 💻 Użycie

### Pobieranie danych
Jeśli chcesz zaktualizować dane wejściowe:
```bash
python downloader.py
```

### Uruchomienie pełnej analizy
Główny skrypt przeprowadza Grid Search, analizę wrażliwości i generuje wykresy:
```bash
python Analiza_porownawcza.py
```

### Uruchomienie poszczególnych algorytmów
Możesz również uruchomić algorytmy niezależnie, aby zobaczyć ich jednostkowe działanie:
```bash
python Algorytm_SA.py
python Algorytm_genetyczny.py
```

## 📊 Wyniki i wnioski

Z przeprowadzonej analizy wynika, że:
1. **Symulowane wyżarzanie (SA)** wykazuje przewagę nad algorytmem genetycznym zarówno pod kątem szybkości obliczeń, jak i stabilności wyników.
2. **SA** osiąga lepsze wartości funkcji celu (niższe ryzyko przy założonej stopie zwrotu).
3. Oba algorytmy skutecznie zbliżają się do **numerycznej granicy efektywnej Markowitza**.


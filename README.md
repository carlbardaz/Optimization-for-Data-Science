# 🚀 Active-Set EVD Solver for Quadratic Min-Cost Flow Problems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Active--Set-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Progetto per il corso di Optimization for Data Science** > *Università di Pisa – Data Science and Business Informatics*

Questo repository contiene l'implementazione di un risolutore personalizzato basato sul **Metodo Active-Set** per problemi di **Quadratic Min-Cost Flow**. Il core dell'algoritmo utilizza la **Decomposizione agli Autovalori (EVD)** per gestire con robustezza matrici Hessiane semidefinite positive (PSD) e sistemi KKT singolari.

---

## 🎯 Obiettivo del Progetto
Il solver affronta il problema di ottimizzazione quadratica vincolata:
$$\min { \frac{1}{2}x^T Q x + q^T x }$$
soggetto a:
- **Vincoli di bilancio:** $Ex = b$
- **Vincoli di capacità (box):** $0 \le x \le u$

Il progetto analizza in particolare l'efficacia della decomposizione **EVD** per determinare direzioni di discesa nello spazio nullo quando la matrice KKT diventa singolare.

---

## ✨ Caratteristiche Principali

* **Active-Set Strategy:** Gestione dinamica dei vincoli di capacità attivi (Lower/Upper bounds).
* **EVD Robustness:** Risoluzione del sistema KKT tramite Eigenvalue Decomposition per una stabilità numerica superiore.
* **Singular System Handling:** Capacità di identificare e percorrere direzioni nello spazio nullo (Fase 2a) o calcolare soluzioni a norma minima (Fase 2b) in caso di singolarità.
* **Fase 1 integrata:** Ricerca del punto iniziale ammissibile tramite programmazione lineare.
* **Benchmarking:** Confronto diretto integrato con il solver industriale **HiGHS**.

---

## 🛠️ Struttura del Codice

Il file principale `ASEVDQP.py` include:
- `solve_qp_evd`: L'algoritmo principale di Active-Set.
- `_build_kkt_system`: Assemblaggio dinamico del sistema KKT ridotto.
- `find_feasible_start`: Metodo per trovare il punto di partenza (LP-based).
- `solve_single_qp_problem_with_comparison`: Utility per testare il solver e confrontarlo con i benchmark.

---

## 📈 Risultati e Performance

L'algoritmo è stato testato su dataset di diverse dimensioni (**Netgen 1000-3000**) e istanze casuali (PD/PSD).

| Classe | Successo | Precisione (AAE vs HiGHS) | Note |
| :--- | :---: | :---: | :--- |
| **Netgen 1000** | 100% | $7.87 \times 10^{-2}$ | Alta accuratezza |
| **Random PSD** | 100% | $10^{-11}$ | Estremamente preciso su piccola scala |

> **Nota:** Sebbene il solver Custom mostri un'elevata robustezza, il costo computazionale per iterazione (dovuto all'EVD) lo rende ideale come prototipo di precisione piuttosto che per applicazioni su larghissima scala, dove solver come HiGHS o Gurobi mantengono una scalabilità superiore.

---

## 🚀 Come Iniziare

### Requisiti
```bash
pip install numpy scipy qpsolvers
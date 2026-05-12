import numpy as np
# Importa eigh (autovalori/autovettori per matrici simmetriche), 
# norm (norma vettoriale) e pinv (pseudoinversa di Moore-Penrose).
from numpy.linalg import eigh, norm, pinv
import time
# Importa linprog (risolutore di programmazione lineare) per la Fase 1.
from scipy.optimize import linprog 
# Importa solve_qp (risolutore standard di QP) e available_solvers (per il confronto).
from qpsolvers import solve_qp, available_solvers # type: ignore
# Importa csc_matrix (matrice sparsa) per l'interfaccia con qpsolvers.
from scipy.sparse import csc_matrix 

# --- Costanti e Tolleranza ---
# Tolleranza numerica usata per i confronti di uguaglianza o la singolarità.
TOL = 1e-8


# --- Funzioni Ausiliarie (Mantenute per la logica del custom solver) ---

def _build_kkt_system(Q, q, E, b, u, x, F, L, U):

    m = Q.shape[0] # Numero totale di variabili
    n = E.shape[0] # Numero di vincoli di uguaglianza

    F_indices = np.array(F, dtype=int) # Indici delle variabili libere
    U_indices = np.array(U, dtype=int) # Indici delle variabili al limite superiore

    n_free = len(F) # Dimensione delle variabili libere
    kkt_dim = n_free + n # Dimensione della matrice KKT

    K = np.zeros((kkt_dim, kkt_dim)) # Matrice KKT
    rhs = np.zeros(kkt_dim) # Vettore del lato destro (right-hand side)

    if n_free > 0:
        # Blocco superiore sinistro: Q_FF (Hessiana ridotta)
        Q_FF = Q[np.ix_(F_indices, F_indices)]
        # Matrice dei vincoli di uguaglianza ridotta E_F
        E_F = E[:, F_indices]

        K[:n_free, :n_free] = Q_FF
        K[:n_free, n_free:] = E_F.T # Blocco superiore destro
        K[n_free:, :n_free] = E_F # Blocco inferiore sinistro

        q_tilde = q[F_indices].copy()
        # Calcola il termine dovuto alle variabili fissate U nel gradiente ridotto
        if len(U) > 0:
            Q_FU = Q[np.ix_(F_indices, U_indices)]
            if x[U_indices].size > 0: 
                q_tilde = q_tilde + Q_FU @ x[U_indices]
        # Lato destro, parte superiore (corrispondente al gradiente ridotto)
        rhs[:n_free] = -q_tilde

    b_bar = b.copy()
    # Calcola il termine dovuto alle variabili fissate nel lato destro dei vincoli E*x=b
    if len(U) > 0:
        E_U = E[:, U_indices]
        if x[U_indices].size > 0: 
             b_bar = b_bar - E_U @ x[U_indices]
    # Lato destro, parte inferiore (corrispondente ai vincoli di uguaglianza)
    rhs[n_free:] = b_bar

    return K, rhs


def _calculate_step_size(x, d_k, u, F):
  
    alpha = np.inf
    blocking_idx = None
    is_upper_bound = False

    # Considera solo gli indici in F (variabili libere)
    valid_indices = [i for i in F if 0 <= i < len(d_k) and 0 <= i < len(x) and 0 <= i < len(u)]

    for i in valid_indices:
        # Caso 1: d_k[i] è positivo (movimento verso l'alto, blocco su u[i])
        if d_k[i] > TOL:  
            alpha_i = (u[i] - x[i]) / d_k[i]
            if alpha_i < alpha:
                alpha = alpha_i
                blocking_idx = i
                is_upper_bound = True
        # Caso 2: d_k[i] è negativo (movimento verso il basso, blocco su 0)
        elif d_k[i] < -TOL:  
            alpha_i = -x[i] / d_k[i] 
            if alpha_i < alpha:
                alpha = alpha_i
                blocking_idx = i
                is_upper_bound = False

    # Il passo non deve superare 1.0 (passo intero verso la soluzione KKT)
    alpha = min(1.0, alpha)

    return alpha, blocking_idx, is_upper_bound

def _check_and_release(g_lagrangian, F, L, U):

    F_new = F[:] 
    L_new = L[:]
    U_new = U[:]
    released_constraint = False

    L_indices = np.array(L_new, dtype=int)
    U_indices = np.array(U_new, dtype=int)

    # Controlla i vincoli al limite inferiore (L): moltiplicatore deve essere >= 0
    if len(L_new) > 0:
        g_L = g_lagrangian[L_indices]
        most_negative_g_idx_in_L = np.argmin(g_L) # Cerca il moltiplicatore più negativo
        i_global = L_indices[most_negative_g_idx_in_L] 

        if g_L[most_negative_g_idx_in_L] < -TOL:
            # Rilascio: g_L < 0, il vincolo è "troppo attivo"
            L_new.remove(i_global)
            if i_global not in F_new: 
                 F_new.append(i_global) # Aggiunge al set libero
            released_constraint = True
            return F_new, L_new, U_new, released_constraint # Ritorna subito (strategia "single-release")

    # Controlla i vincoli al limite superiore (U): moltiplicatore deve essere <= 0
    if len(U_new) > 0 and not released_constraint:
        g_U = g_lagrangian[U_indices]
        most_positive_g_idx_in_U = np.argmax(g_U) # Cerca il moltiplicatore più positivo
        i_global = U_indices[most_positive_g_idx_in_U] 

        if g_U[most_positive_g_idx_in_U] > TOL:
            # Rilascio: g_U > 0, il vincolo è "troppo attivo"
            U_new.remove(i_global)
            if i_global not in F_new: 
                F_new.append(i_global) # Aggiunge al set libero
            released_constraint = True

    return F_new, L_new, U_new, released_constraint

def _update_sets_add(F, L, U, blocking_idx, is_upper_bound):

    F_new = F[:] 
    L_new = L[:]
    U_new = U[:]

    if blocking_idx is None:
        return F_new, L_new, U_new

    # La variabile bloccante deve essere rimossa dal set libero F
    if blocking_idx in F_new:
        F_new.remove(blocking_idx)

    # Aggiungi al set L o U
    if is_upper_bound:
        if blocking_idx not in U_new: 
             U_new.append(blocking_idx)
    else:
        if blocking_idx not in L_new: 
             L_new.append(blocking_idx)

    return F_new, L_new, U_new


def find_feasible_start(E, b, u, TOL=1e-8):

    n_nodes, n_arcs = E.shape # n_nodes è il numero di vincoli di uguaglianza

    # Coefficienti dell'obiettivo: 0 per x, 1 per le variabili artificiali
    c_x = np.zeros(n_arcs)
    c_a = np.ones(n_nodes)
    c_obj = np.concatenate([c_x, c_a, c_a])

    I_nodes = np.eye(n_nodes)
    # Matrice dei vincoli di uguaglianza A_eq: [E | I | -I]
    A_eq = np.hstack([E, I_nodes, -I_nodes])

    b_eq = b # Lato destro dei vincoli

    # Limiti di box per x: [0, u]
    bounds_x = list(zip(np.zeros(n_arcs), u))
    # Limiti per a_pos e a_neg: [0, inf]
    bounds_a_pos = [(0, None)] * n_nodes
    bounds_a_neg = [(0, None)] * n_nodes
    all_bounds = bounds_x + bounds_a_pos + bounds_a_neg

    try:
        # Risolve l'LP con linprog (metodo 'highs' è efficiente)
        res = linprog(c_obj,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=all_bounds,
                      method='highs')

        if not res.success:
            return "solver_failed", None

        # Se il valore obiettivo è > TOL, le variabili artificiali sono positive 
        # (non si è trovato un punto ammissibile).
        if res.fun > TOL:
            return "infeasible", None
        else:
            # Soluzione ammissibile trovata: prende solo la parte relativa a x
            x_feasible = res.x[:n_arcs]
            return "feasible", x_feasible

    except Exception as e:
        return "solver_failed", None


# --- Funzione Solver Principale (solve_qp_evd) ---
def solve_qp_evd(Q, q, E, b, u, x_start, max_iter=5000):

    start_time = time.process_time() 
    iter_times = [] 

    m = Q.shape[0] # Numero di variabili
    n = E.shape[0] # Numero di vincoli di uguaglianza

    if m == 0: 
        # Caso triviale (nessuna variabile)
        x = np.array([])  
        f_val = 0.0       
        total_time = time.process_time() - start_time
        return x, f_val, total_time, 0, iter_times

    x = x_start.copy() # Punto iniziale (ammissibile)
    L = [] # Set degli indici fissati al limite inferiore (x_i = 0)
    U = [] # Set degli indici fissati al limite superiore (x_i = u_i)
    F = [i for i in range(len(x))] # Set degli indici liberi (0 < x_i < u_i)
    mu = np.zeros(n) # Moltiplicatori di Lagrange dei vincoli di uguaglianza E*x=b
    iterations = 0 

    for k in range(max_iter):
        iter_start_time = time.process_time() 
        iterations = k + 1 

        K, rhs = _build_kkt_system(Q, q, E, b, u, x, F, L, U)

        d_k = np.zeros(m) # Direzione di ricerca (non nulla solo per gli indici in F)
        lmbda_min = np.inf 

        # 1. Analisi della Matrice KKT
        if K.shape[0] > 0 and K.shape[0] == K.shape[1]: 
            try:
                # Decomposizione agli autovalori/autovettori
                Lmbda, V = eigh(K)
                if len(Lmbda) > 0:
                    # L'autovalore assoluto minimo è un indicatore di singolarità
                    lmbda_min = np.min(np.abs(Lmbda))
            except np.linalg.LinAlgError:
                lmbda_min = -1 # Segnala un errore o singolarità
        elif len(F) == 0:
             # Caso in cui tutte le variabili sono fissate (K è vuota o ridotta a E*x=b)
             if n > 0 and E.shape[0] == n :
                  mu = np.zeros(n)
             else:
                  mu = np.zeros(0)
             pass 

        F_np = np.array(F, dtype=int) 

        # 2. Risoluzione del Sistema KKT

        if lmbda_min > TOL: # CASO REGOLARE (K non singolare)
            # Soluzione del sistema K*z = rhs, dove z = [d_F, mu]
            if K.shape[0] > 0 and K.shape[0] == K.shape[1] : 
                try:
                    # Risoluzione del sistema lineare tramite EVD (stabile)
                    rhs_prime = V.T @ rhs
                    y = rhs_prime / Lmbda # Calcolo del vettore y nella base di V
                    z = V @ y # Ritorna alla base canonica (z = K^{-1} * rhs)
                    
                    if len(z) >= len(F): 
                        x_F_sol = z[:len(F)] # Soluzione nel sottospazio F
                        mu = z[len(F):] if len(z) > len(F) else mu # Moltiplicatori mu
                        if len(F) > 0: 
                             # La direzione di ricerca è la differenza tra la soluzione KKT 
                             # e il punto corrente (d_k = x_kkt - x)
                             d_k[F_np] = x_F_sol - x[F_np]
                    else:
                        lmbda_min = -1 # Errore dimensionale
                except np.linalg.LinAlgError:
                    lmbda_min = -1 

        if lmbda_min <= TOL: # CASO SINGOLARE (K quasi-singolare o singolare)
            # Si cerca una direzione di discesa negativa di curvatura (Null Space Direction)
            if K.shape[0] > 0 and len(F) > 0: 
                
                # Calcolo del gradiente ridotto (solo per F)
                g_F = Q[np.ix_(F_np, F_np)] @ x[F_np] + q[F_np]
                U_np = np.array(U, dtype=int)
                if len(U) > 0: # Aggiunge il contributo dei vincoli fissati U
                     Q_FU = Q[np.ix_(F_np, U_np)]
                     if x[U_np].size > 0:
                         g_F += Q_FU @ x[U_np]

                # 2a. Proiezione sul Null Space (Null Space Direction)
                null_space_indices = np.where(np.abs(Lmbda) <= TOL)[0] # Indici degli autovalori nulli
                if len(null_space_indices) > 0 and V.shape[1] > max(null_space_indices):
                     V0 = V[:, null_space_indices] # Base dello spazio nullo di K
                     if V0.shape[0] >= len(F): 
                         D_null_space = V0[:len(F), :] # Base dello spazio nullo proiettata su F
                         if D_null_space.size > 0:
                             # Cerca la direzione di massima discesa nel null space
                             c = D_null_space.T @ g_F
                             d_F_desc = -D_null_space @ c
                         else: d_F_desc = np.zeros(len(F))
                     else: d_F_desc = np.zeros(len(F))
                else: d_F_desc = np.zeros(len(F))

                if norm(d_F_desc) > TOL: # Direzione di discesa trovata
                    # Esegue un passo (passo 1)
                    d_k[F_np] = d_F_desc
                    alpha, i_blocking, is_upper = _calculate_step_size(x, d_k, u, F) 
                    
                    if np.isinf(alpha) and i_blocking is None:
                        # Problema illimitato (non capita per Q positive semi-definite)
                        print(f"Iter {k}: Rilevata direzione illimitata.")
                        total_time = time.process_time() - start_time
                        iter_times.append(time.process_time() - iter_start_time)
                        return x, -np.inf, total_time, iterations, iter_times

                    alpha = min(1.0, alpha)
                    x = x + alpha * d_k
                    x = np.clip(x, 0.0, u) # Assicura che x rimanga nei limiti [0, u]
                    F, L, U = _update_sets_add(F, L, U, i_blocking, is_upper) # Aggiorna i set
                    
                    # Continua con l'iterazione successiva (non controlla l'ottimalità subito)
                    iter_end_time = time.process_time()
                    iter_times.append(iter_end_time - iter_start_time)
                    continue
                else: 
                    # 2b. Soluzione tramite Pseudoinversa (Least Squares Solution)
                    # Se non si trova una direzione di discesa, si usa la soluzione 
                    # di norma minima del sistema KKT (approssimazione pinv).
                    try:
                        threshold = 1e-10 
                        # Inversione modificata per EVD (0 per autovalori quasi nulli)
                        Lmbda_inv_mod = np.where(np.abs(Lmbda) > threshold, 1.0 / Lmbda, 0.0)

                        rhs_prime = V.T @ rhs 
                        z = V @ (rhs_prime * Lmbda_inv_mod) # z = V * diag(Lmbda_inv_mod) * V.T * rhs
                        if len(z) >= len(F):
                            x_F_sol = z[:len(F)]
                            mu = z[len(F):] if len(z) > len(F) else mu
                            d_k[F_np] = x_F_sol - x[F_np]
                        else: d_k[F_np] = 0
                    except np.linalg.LinAlgError:
                         d_k[F_np] = 0


        # 3. Controllo di Ottimalità o Esecuzione del Passo

        if norm(d_k) <= TOL:
            # Caso 1: La direzione di ricerca è nulla (x è la soluzione KKT nel sottospazio)
            
            # Calcolo del gradiente del Lagrangiano (moltiplicatore per vincoli di box)
            g_lagrangian = Q @ x + q
            if E.size > 0 and mu.shape[0] == E.shape[0]:
                g_lagrangian += E.T @ mu
            
            # Controllo dei moltiplicatori per rilasciare i vincoli
            F_new, L_new, U_new, released = _check_and_release(g_lagrangian, F, L, U) 

            if not released: 
                # Se non è stato rilasciato nessun vincolo, la soluzione è ottima
                f_val = 0.5 * x.T @ Q @ x + q.T @ x
                total_time = time.process_time() - start_time
                iter_end_time = time.process_time() 
                iter_times.append(iter_end_time - iter_start_time)
                return x, f_val, total_time, iterations, iter_times

            # Se è stato rilasciato un vincolo, aggiorna i set e ricomincia l'iterazione
            F, L, U = F_new, L_new, U_new 
            iter_end_time = time.process_time()
            iter_times.append(iter_end_time - iter_start_time)
            continue

        # Caso 2: Direzione di ricerca non nulla (d_k)
        
        # Calcolo del passo massimo (alpha_max)
        alpha_max, i_blocking, is_upper = _calculate_step_size(x, d_k, u, F) 
        alpha = min(1.0, alpha_max)

        # Esecuzione del passo
        x = x + alpha * d_k
        x = np.clip(x, 0.0, u) 

        if alpha < 1.0 - TOL: 
            # Se alpha < 1, un vincolo di box è stato bloccato.
            F, L, U = _update_sets_add(F, L, U, i_blocking, is_upper) 
        else: 
             # Se alpha = 1 (passo intero), la soluzione KKT del sottospazio è stata raggiunta.
             # Si ricontrolla se alcune variabili libere (F) sono accidentalmente finite 
             # sui limiti a causa di errori numerici o per la natura della soluzione KKT.
             newly_L = [i for i in F if x[i] <= TOL and i not in L]
             newly_U = [i for i in F if u[i] - x[i] <= TOL and i not in U]

             if newly_L or newly_U:
                  F = [i for i in F if i not in newly_L and i not in newly_U]
                  L.extend(newly_L)
                  U.extend(newly_U)

        iter_end_time = time.process_time()
        iter_times.append(iter_end_time - iter_start_time)

    # Uscita per massimo numero di iterazioni
    f_val = 0.5 * x.T @ Q @ x + q.T @ x
    total_time = time.process_time() - start_time
    print(f"\nMassimo numero di iterazioni ({max_iter}) .")
    print(f"Valore F(x): {f_val:.4f}")
    return x, f_val, total_time, iterations, iter_times


def solve_single_qp_problem_with_comparison(Q, q_lin, E, b, u_cap):


    # Controllo delle dimensioni di input
    if Q.shape[0] != len(q_lin) or Q.shape[0] != E.shape[1] or Q.shape[0] != len(u_cap) or E.shape[0] != len(b):
        print("\n ERRORE: Le dimensioni delle matrici/vettori di input non sono coerenti.")
        return {}
    
    n_arcs = Q.shape[0]
    lb = np.zeros(n_arcs) # Limite inferiore (lower bound) sempre a zero
    results = {}


    # --- 1. Fase 1: Trova x_start ---
    print("\n--- Fase 1: Ricerca di un Punto Iniziale Ammissibile ---")
    phase1_status, x_start = find_feasible_start(E, b, u_cap)
    
    if phase1_status == "infeasible":
        print("\nIl problema è INAMMISSIBILE. Risoluzione interrotta.")
        return results

    if phase1_status == "solver_failed":
        print("\nIl solutore della Fase 1 non ha funzionato. Tentativo con punto iniziale nullo (x=0).")
        x_start = np.zeros(n_arcs) 
    
    # --- 2. Solver Personalizzato (solve_qp_evd) ---
    print("\n--- 1. Esecuzione Solver Custom (Active-Set EVD) ---")
    try:
        x_s_c = x_start.copy()
        # Esegue il solutore Active Set implementato
        x_opt, f_opt, t_total, iters, t_iters = solve_qp_evd(
            Q, q_lin, E, b, u_cap, x_s_c
        )
        
        results['Custom'] = {
            'fopt': f_opt,
            'time': t_total,
            'iters': iters,
            'x_opt': x_opt,
            'success': np.isfinite(f_opt)
        }
        print(f"  RISULTATO CUSTOM: Valore Obiettivo={f_opt:.6e}, Tempo={t_total:.6f}s, Iterazioni={iters}")

    except Exception as e:
        results['Custom'] = {'fopt': np.nan, 'time': np.nan, 'iters': np.nan, 'x_opt': None, 'success': False}
        print(f"  ERRORE CUSTOM SOLVER: {e}")

    # --- 3. HiGHS Solver (per confronto) ---
    print("\n--- 2. Esecuzione HiGHS Solver ---")
    try:
        if 'highs' in available_solvers:
            # Converte le matrici in formato sparso (richiesto da qpsolvers per HiGHS)
            P_sparse = csc_matrix(Q)
            A_sparse = csc_matrix(E) 
            q_h = q_lin.copy()

            start_h = time.process_time()
            # Risolve il QP con HiGHS (un solutore moderno e performante)
            x_sol_highs = solve_qp(
                P=P_sparse, 
                q=q_h, 
                A=A_sparse, # Matrice dei vincoli di uguaglianza
                b=b, 
                lb=lb, # Lower bound (0)
                ub=u_cap, # Upper bound (u)
                solver='highs'
            )
            time_h = time.process_time() - start_h

            success_highs = x_sol_highs is not None
            if success_highs:
                # Calcola il valore obiettivo con la soluzione di HiGHS
                f_highs = 0.5 * x_sol_highs.T @ Q @ x_sol_highs + q_h.T @ x_sol_highs
            else:
                f_highs = np.nan
            
            results['HiGHS'] = {
                'fopt': f_highs,
                'time': time_h,
                'iters': np.nan, 
                'x_opt': x_sol_highs,
                'success': success_highs
            }
            print(f"  RISULTATO HiGHS: Valore Obiettivo={f_highs:.6e}, Tempo={time_h:.6f}s, Status={'Success' if success_highs else 'Failed'}")
        else:
            print("  ⚠️ HiGHS solver non trovato da qpsolvers. Saltato.")
            results['HiGHS'] = {'fopt': np.nan, 'time': np.nan, 'iters': np.nan, 'x_opt': None, 'success': False}
    except Exception as e:
        results['HiGHS'] = {'fopt': np.nan, 'time': np.nan, 'iters': np.nan, 'x_opt': None, 'success': False}
        print(f"   ERRORE HiGHS: {e}")

    # --- 4. Confronto Finale ---
    print("\n==================================================")
    print("📋 Riassunto dei Risultati")
    print("==================================================")
    
    f_custom = results.get('Custom', {}).get('fopt', np.nan)
    f_highs = results.get('HiGHS', {}).get('fopt', np.nan)
    
    # Prende come riferimento il risultato Custom, altrimenti HiGHS, se finito
    f_ref = f_custom if np.isfinite(f_custom) else f_highs

    print(f"{'Solver':<15}{'Successo':<10}{'Obiettivo (f(x))':<25}{'Tempo CPU (s)':<15}{'Iterazioni':<15}{'Errore Abs (vs Ref)':<25}")
    print("-" * 105)

    for solver, res in results.items():
        fopt = res.get('fopt', np.nan)
        time_s = res.get('time', np.nan)
        iters = res.get('iters', np.nan)
        success = res.get('success', False)
        
        # Calcola l'errore se il riferimento è finito
        error_abs = np.nan
        if np.isfinite(f_ref) and np.isfinite(fopt):
            error_abs = abs(fopt - f_ref)
        elif not np.isfinite(f_ref) and not np.isfinite(fopt):
             error_abs = 0.0 # Se entrambi non finiti, l'errore è 0

        # Formattazione per la stampa
        fopt_str = f"{fopt:.6e}" if np.isfinite(fopt) else "N/A"
        time_str = f"{time_s:.6f}" if np.isfinite(time_s) else "N/A"
        iters_str = f"{iters:.0f}" if np.isfinite(iters) else "N/A"
        error_str = f"{error_abs:.6e}" if np.isfinite(error_abs) else "N/A"
        success_str = "Sì" if success else "No"
        
        print(f"{solver:<15}{success_str:<10}{fopt_str:<25}{time_str:<15}{iters_str:<15}{error_str:<25}")

    print("==================================================")
    
    return results

# --- Blocco Main con Input Definito Matrice per Matrice ---
if __name__ == "__main__":
    
    # Matrice Q (Hessiana, deve essere semi-definita positiva per un problema QP convesso)
    Q_input = np.array([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # <--- Singolarità potenziale: il costo quadratico 
                               #      per x[2] è zero.
        [0.0, 0.0, 0.0, 4.0]
    ])
    
    # Vettore q (Costi lineari)
    q_input = np.array([-1.0, -2.0, 0.0, -3.0])
    
    # Matrice E (Vincoli di uguaglianza, es. vincoli di bilancio in una rete)
    E_input = np.array([
        [1.0, 0.0, -1.0, 1.0],
        [0.0, 1.0, 1.0, -1.0]
    ])
    
    # Vettore b (Bilanci o valori target per i vincoli di uguaglianza)
    b_input = np.array([0.5, 0.5])
    
    # Vettore u (Limiti superiori, lower bound è implicito a 0)
    u_input = np.array([1.0, 1.0, 1.0, 1.0])
    
    # 2. Risoluzione del Singolo Problema con Confronto
    final_results = solve_single_qp_problem_with_comparison(
        Q_input, q_input, E_input, b_input, u_input, 
    )
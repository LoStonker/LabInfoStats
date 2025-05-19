import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.cost import ExtendedBinnedNLL
from iminuit.cost import UnbinnedNLL
from iminuit.cost import BinnedNLL
from scipy.stats import chi2
from scipy.integrate import quad
from scipy.stats import norm, chi2, t
import sys
import inspect
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
from matplotlib.ticker import AutoMinorLocator


import numpy as np
from scipy.stats import norm, t # Necessario per la funzione

def Tstudent(val1, val2, sigma1, sigma2, val1_name="Valore 1", val2_name="Valore 2", use_ttest=False, custom_df=None, significance_level=0.05):
    """
    Esegue un test di compatibilità tra due valori con le loro incertezze.
    """
    # Inizializzazione di default
    T_score = np.nan
    p_value = np.nan
    compatibili = False # Assumiamo non compatibili fino a prova contraria
    dist_type = "N/A"
    df_eff = np.nan # Gradi di libertà effettivi usati

    if sigma1 < 0 or sigma2 < 0:
        print(f"Errore nel test tra {val1_name} e {val2_name}: Le incertezze (sigma) non possono essere negative.")
        # T_score, p_value, compatibili rimangono ai loro valori di default (nan, nan, False)
        return T_score, p_value, compatibili
        
    denominatore_T = np.sqrt(sigma1**2 + sigma2**2)

    if denominatore_T < 1e-15: # Denominatore quasi zero
        if np.abs(val1 - val2) < 1e-15: # Anche differenza quasi zero
            T_score = 0.0
            p_value = 1.0
            compatibili = True
            dist_type = "N/A (Valori identici con sigma denominatore zero)"
        else: # Differenza non zero
            T_score = np.inf
            p_value = 0.0
            # compatibili rimane False
            dist_type = "N/A (Valori diversi con sigma denominatore zero)"
    else: # Denominatore_T è valido
        T_score = np.abs(val1 - val2) / denominatore_T

        if use_ttest:
            if custom_df is not None:
                df_eff = custom_df
                if df_eff <= 0:
                    print(f"Errore nel test tra {val1_name} e {val2_name}: custom_df per t-test ({df_eff}) deve essere > 0.")
                    # T_score è calcolato, p_value e compatibili rimangono ai default
                    return T_score, p_value, compatibili 
            else: # Calcola df approssimato
                s1_sq = sigma1**2
                s2_sq = sigma2**2
                if s1_sq < 1e-15 and s2_sq < 1e-15:
                    df_eff = np.inf # Se entrambe le sigma sono ~0, la t si approssima alla normale
                else:
                    df_num = (s1_sq + s2_sq)**2
                    df_den = s1_sq**2 + s2_sq**2 
                    if df_den < 1e-15:
                        print(f"Attenzione nel test tra {val1_name} e {val2_name}: Denominatore per df è zero. Si userà df=infinito.")
                        df_eff = np.inf
                    else:
                        df_eff = df_num / df_den
            
            # Calcolo p_value per t-test
            if np.isinf(df_eff):
                p_value = 2 * (1 - norm.cdf(T_score))
                dist_type = f"t di Student (df=infinito, equivale a Normale)"
            elif df_eff < 1:
                print(f"Attenzione nel test tra {val1_name} e {val2_name}: df calcolato ({df_eff:.2f}) < 1. Il p-value del t-test potrebbe non essere affidabile.")
                p_value = np.nan # Non calcoliamo p_value per df < 1 in questo esempio
                dist_type = f"t di Student (df={df_eff:.2f} - problematico)"
            else:
                p_value = 2 * (1 - t.cdf(T_score, df_eff))
                dist_type = f"t di Student (df={df_eff:.2f})"
        
        else: # Usa la distribuzione Normale standard (use_ttest = False)
            df_eff = np.inf # Gradi di libertà per la normale
            p_value = 2 * (1 - norm.cdf(T_score))
            dist_type = "Normale Standard"

        # Determina compatibilità basata sul p_value calcolato (se non è NaN)
        if not np.isnan(p_value):
            compatibili = p_value > significance_level
        # Se p_value è NaN, compatibili rimane False (dall'inizializzazione)

    # Stampa finale dei risultati
    print(f"\nTest di compatibilità tra {val1_name} ({val1:.3e} ± {sigma1:.2e}) e {val2_name} ({val2:.3e} ± {sigma2:.2e}):")
    if not np.isnan(T_score):
        print(f"  Differenza: {np.abs(val1 - val2):.2e}")
        print(f"  Incertezza sulla differenza (denominatore di T): {denominatore_T:.2e}" if denominatore_T >= 1e-15 else "N/A (denominatore T zero)")
        print(f"  Statistica del test (T o Z): {T_score:.2f}")
    else: # T_score potrebbe essere NaN se si esce prima
         print("  Statistica del test non calcolata a causa di errore precedente.")

    print(f"  Distribuzione usata: {dist_type}")
    # print(f"  Gradi di libertà effettivi (df): {df_eff if not np.isnan(df_eff) else 'N/A'}") # Opzionale
    
    if not np.isnan(p_value):
        print(f"  P-value (due code): {p_value:.4f}")
        if compatibili:
            print(f"  I due valori sono COMPATIBILI (p > {significance_level}).")
        else:
            print(f"  I due valori NON sono compatibili (p <= {significance_level}).")
    else:
        print(f"  P-value non calcolato (probabilmente a causa di df < 1 o altri errori).")
        print(f"  Impossibile determinare la compatibilità.")
        
    return T_score, p_value, compatibili

# --- ESEMPIO DI UTILIZZO ---
if __name__ == '__main__':
    print("--- Esempio 1: Valori compatibili (Normale) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1")

    print("\n--- Esempio 2: Valori non compatibili (Normale) ---")
    test_compatibilita(10.0, 10.8, 0.1, 0.1, "Val A2", "Val B2")

    print("\n--- Esempio 3: Valori compatibili (t-test, df approssimato) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True)

    print("\n--- Esempio 4: Valori compatibili (t-test, df custom alto) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True, custom_df=100)

    print("\n--- Esempio 5: Valori identici, sigma zero ---")
    test_compatibilita(10.0, 10.0, 0.0, 0.0, "Val C1", "Val C2")

    print("\n--- Esempio 6: Valori diversi, sigma zero ---")
    test_compatibilita(10.0, 10.1, 0.0, 0.0, "Val D1", "Val D2")
    
    print("\n--- Esempio 7: Una sigma zero (t-test forzato) ---")
    test_compatibilita(10.0, 10.1, 0.5, 0.0, "Val E1", "Val E2", use_ttest=True) # df sarà inf

    print("\n--- Esempio 8: custom_df non valido ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True, custom_df=0)

    print("\n--- Esempio 9: df calcolato < 1 (molto diverse sigma) ---")
    # df_num = (0.01^2 + 1^2)^2 ~ 1
    # df_den = (0.01^4 + 1^4) ~ 1
    # df ~ 1
    # Proviamo sigma molto diverse per vedere se df_num/df_den diventa < 1
    # (sigma1^2+sigma2^2)^2 / (sigma1^4+sigma2^4)
    # Se sigma1 -> 0, df -> sigma2^4 / sigma2^4 = 1
    # Questa formula per df non scenderà facilmente sotto 1 se almeno una sigma è non nulla.
    # Ma se forzassimo df<1 con custom_df:
    test_compatibilita(10.0, 10.1, 0.5, 0.4, "Val G1", "Val G2", use_ttest=True, custom_df=0.5)



""" Funzione unica per fare i fit 
ESEMPIO DI UTILIZZO
    def exp_model(x, A, B):
        return A * np.exp(B * x)

    x_exp = np.linspace(1, 5, 15)
    y_exp_true = exp_model(x_exp, 100, -0.8)
    y_exp_noise = np.random.normal(0, y_exp_true * 0.2, len(x_exp)) # Errore proporzionale
    y_exp_data = y_exp_true + y_exp_noise
    sigma_y_exp = np.abs(y_exp_true * 0.2) # Stima dell'errore

    # Rendi positivi i dati y per scala log
    y_exp_data[y_exp_data <= 0] = 1e-3
    sigma_y_exp[sigma_y_exp <= 0] = 1e-3

    data_exp = {'x': x_exp, 'y': y_exp_data, 'sigma_y': sigma_y_exp}
    init_exp = {'A': 90, 'B': -0.7}

    fit_log = FitBomberone(exp_model, data_exp, init_exp, fit_method='Scipy',
                        title="Fit Esponenziale con Scala Log", ylabel = 'Dio stronzo', xlabel = 'Dio merda')
    fit_log.perform_fit().print_results()
    fit_log.plot_results(log_scale_y=True, log_scale_x=True, info_box_pos='upper right')"""


class FitBomberone:
    """
    Classe unificata per eseguire fit di dati con diversi metodi.

    Permette di scegliere tra:
    - 'LeastSquares': Minimi quadrati usando iminuit.cost.LeastSquares (richiede iminuit).
    - 'Chi2': Minimizzazione del Chi-quadro usando Minuit (richiede iminuit).
    - 'Scipy': Minimi quadrati usando scipy.optimize.curve_fit (richiede scipy).
    - 'ODR': Orthogonal Distance Regression (richiede scipy). Gestisce errori su x e y.

    Offre opzioni per personalizzare il plot dei risultati, inclusa la posizione
    del box informativo e l'uso della scala logaritmica sull'asse y.
    """
    def __init__(self, model_func, data_arrays, initial_params,
                 fit_method='LeastSquares', xlabel="x", ylabel="y", title="Risultati del fit"):
        """

        Args:
            model_func (callable): La funzione modello da fittare.
                Deve accettare come primo argomento l'array delle x e poi i parametri
                del fit come argomenti nominali (keyword arguments), es: def my_model(x, a, b, c): ...
            data_arrays (dict): Dizionario contenente i dati. Deve avere le chiavi:
                'x': array dei valori x.
                'y': array dei valori y.
                'sigma_y': array degli errori su y (deviazioni standard).
                Può opzionalmente contenere:
                'sigma_x': array degli errori su x (deviazioni standard). Necessario solo per il metodo 'ODR'.
                           Se non fornito e si usa 'ODR', verrà assunto come zero.
            initial_params (dict): Dizionario con i valori iniziali per i parametri del fit.
                                   Le chiavi devono corrispondere ai nomi dei parametri in model_func.
            fit_method (str, optional): Il metodo di fit da utilizzare.
                                        Opzioni: 'LeastSquares', 'Chi2', 'Scipy', 'ODR'.
                                        Default: 'LeastSquares'.
            xlabel (str, optional): Etichetta per l'asse x del grafico. Default: "x".
            ylabel (str, optional): Etichetta per l'asse y del grafico. Default: "y".
            title (str, optional): Titolo del grafico. Default: "Risultati del fit".
        """
        # --- Validazione Input Iniziale ---
        valid_methods = ['LeastSquares', 'Chi2', 'Scipy', 'ODR']
        if fit_method not in valid_methods:
            raise ValueError(f"Metodo di fit '{fit_method}' non valido. Scegliere tra: {valid_methods}")

        if not callable(model_func):
             raise TypeError("model_func deve essere una funzione eseguibile.")
        if not isinstance(data_arrays, dict):
             raise TypeError("data_arrays deve essere un dizionario.")
        if not isinstance(initial_params, dict):
             raise TypeError("initial_params deve essere un dizionario.")

        # --- Memorizzazione Attributi ---
        self.model = model_func
        self.fit_method = fit_method
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        # --- Estrazione Dati ---
        self.x = np.asarray(data_arrays['x'])
        self.y = np.asarray(data_arrays['y'])
        self.sigma_y = np.asarray(data_arrays.get('sigma_y', np.ones_like(self.y)))

        # Gestione sigma_x (rilevante solo per ODR)
        if 'sigma_x' in data_arrays:
             self.sigma_x = np.asarray(data_arrays['sigma_x'])
        elif fit_method == 'ODR':
             print("Attenzione: sigma_x non fornito per il metodo ODR. Verrà assunto nullo.")
             self.sigma_x = np.zeros_like(self.x)
        else:
             self.sigma_x = None # Non necessario per altri metodi

        # --- Estrazione Parametri ---
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        if not all_param_names:
             raise ValueError("La funzione modello non sembra accettare argomenti.")
        # Il primo argomento è assunto essere la variabile indipendente (x)
        self.param_names = all_param_names[1:]
        if not self.param_names:
             raise ValueError("La funzione modello deve accettare almeno un parametro oltre alla variabile indipendente.")

        self.initial_params = initial_params

        # --- Inizializzazione Risultati ---
        self.fit_result = None    # Dizionario {nome: (valore, errore)}
        self.m = None             # Oggetto Minuit (per LeastSquares, Chi2)
        self.odr_output = None    # Oggetto Output di ODR
        self.popt = None          # Parametri ottimizzati da Scipy
        self.pcov = None          # Matrice di covarianza da Scipy
        self.chi2_val = None      # Valore del Chi-quadro
        self.dof = None           # Gradi di libertà
        self.p_value = None       # p-value del fit
        self.is_fit_valid = False # Flag per la validità del fit (es. da Minuit)

        # --- Validazione Input Dettagliata ---
        self._validate_inputs()

    def _validate_inputs(self):
        """Controlla la consistenza degli array di input e dei parametri."""
        if not (len(self.x) == len(self.y) == len(self.sigma_y)):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza.")
        if self.sigma_x is not None and len(self.x) != len(self.sigma_x):
             raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza.")

        # Verifica che sigma_y non contenga zeri o valori non positivi se usati come errori
        if np.any(self.sigma_y <= 0):
            print("Attenzione: sigma_y contiene valori nulli o negativi. Questo può causare problemi nel fit.")
            # Potresti voler sollevare un errore o sostituire questi valori a seconda del contesto
            # raise ValueError("sigma_y non può contenere valori nulli o negativi.")

        # Verifica parametri iniziali
        missing_params = set(self.param_names) - set(self.initial_params.keys())
        extra_params = set(self.initial_params.keys()) - set(self.param_names)
        if missing_params:
            raise ValueError(f"Parametri iniziali mancanti per: {missing_params}")
        if extra_params:
            raise ValueError(f"Parametri iniziali non richiesti dalla funzione modello: {extra_params}")

    # --- Funzioni Ausiliarie per Fit Specifici ---

    def _odr_model_wrapper(self, B, x):
        """Wrapper per la funzione modello richiesta da scipy.odr."""
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)

    def _chi2_func_wrapper(self, *args):
        """Wrapper per la funzione Chi-quadro richiesta da Minuit."""
        params_dict = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params_dict)
        # Evita divisione per zero se sigma_y è zero (anche se _validate_inputs avverte)
        # Si potrebbe usare np.where o aggiungere un piccolo epsilon, ma qui usiamo mask
        valid_sigma = self.sigma_y > 0
        residuals = np.zeros_like(self.y)
        residuals[valid_sigma] = (self.y[valid_sigma] - y_model[valid_sigma]) / self.sigma_y[valid_sigma]
        return np.sum(residuals**2)

    # --- Metodi di Fit ---

    def _fit_odr(self):
        """Esegue il fit usando Orthogonal Distance Regression (ODR)."""
        if self.sigma_x is None:
             # Questo non dovrebbe accadere grazie al check in __init__, ma per sicurezza
             print("Avviso: ODR richiede sigma_x. Assumendo sigma_x = 0.")
             self.sigma_x = np.zeros_like(self.x)

        beta0 = [self.initial_params[name] for name in self.param_names]
        odr_model = Model(self._odr_model_wrapper)
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        odr = ODR(data, odr_model, beta0=beta0)
        self.odr_output = odr.run()

        if self.odr_output.info > 0: # Controlla se il fit è terminato con successo
            self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.odr_output.beta, self.odr_output.sd_beta)}
            self.chi2_val = self.odr_output.sum_square # ODR chiama chi2 'sum_square'
            self.dof = len(self.x) - len(self.param_names)
            self.p_value = 1 - chi2.cdf(self.chi2_val, self.dof) if self.dof > 0 else 0
            self.is_fit_valid = True
        else:
             print(f"Errore ODR: Il fit non è converguto (codice info: {self.odr_output.info}).")
             # Potresti voler popolare self.fit_result con NaN o valori iniziali
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             self.is_fit_valid = False


    def _fit_least_squares(self):
        """Esegue il fit usando iminuit.cost.LeastSquares."""
        try:
             # LeastSquares accetta direttamente la funzione modello con keyword args
             least_squares_cost = LeastSquares(self.x, self.y, self.sigma_y, self.model)
             self.m = Minuit(least_squares_cost, **self.initial_params)
             self.m.migrad()  # Esegui minimizzazione
             self.m.hesse()   # Calcola errori accurati (matrice di Hesse)

             self.is_fit_valid = self.m.valid
             if not self.is_fit_valid:
                  print("Attenzione: La covarianza del fit (LeastSquares) non è valida. Gli errori potrebbero essere inaffidabili.")

             self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
             self.chi2_val = self.m.fval # Minuit chiama chi2 'fval'
             self.dof = self.m.ndof
             self.p_value = self.m.fval / self.dof if self.dof > 0 else 0 # Chi2 p-value
             self.p_value = chi2.sf(self.m.fval, self.m.ndof) # Modo corretto con scipy.stats.chi2.sf (survival function)

        except Exception as e:
             print(f"Errore durante il fit LeastSquares con Minuit: {e}")
             self.is_fit_valid = False
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}


    def _fit_chi2(self):
        """Esegue il fit minimizzando il Chi-quadro con Minuit."""
        try:
            # Minuit richiede che la funzione da minimizzare accetti parametri posizionali
            initial_values_list = [self.initial_params[name] for name in self.param_names]
            # Assegna nomi ai parametri per Minuit per migliore output
            self.m = Minuit(self. _chi2_func_wrapper, *initial_values_list, name=self.param_names)
            self.m.errordef = Minuit.LEAST_SQUARES # Imposta errordef a 1 per Chi2/Least Squares
            self.m.migrad()
            self.m.hesse()

            self.is_fit_valid = self.m.valid
            if not self.is_fit_valid:
                 print("Attenzione: La covarianza del fit (Chi2) non è valida. Gli errori potrebbero essere inaffidabili.")

            # Recupera i risultati usando i nomi dei parametri
            self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
            self.chi2_val = self.m.fval
            self.dof = len(self.x) - len(self.param_names) # m.ndof potrebbe non essere corretto qui
            # self.p_value = self.m.fcn(self.m.values) / self.dof if self.dof > 0 else 0
            self.p_value = chi2.sf(self.m.fval, self.dof) if self.dof > 0 else 0

        except Exception as e:
            print(f"Errore durante il fit Chi2 con Minuit: {e}")
            self.is_fit_valid = False
            self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}

    def _fit_scipy(self):
        """Esegue il fit usando scipy.optimize.curve_fit."""
        try:
            # curve_fit preferisce parametri posizionali, ma può gestire keyword se la firma corrisponde
            # Per sicurezza e coerenza, potremmo creare un wrapper, ma proviamo direttamente
            # La funzione modello DEVE avere la forma func(x, p1, p2, ...) per curve_fit

            # Assicurati che l'ordine dei parametri iniziali corrisponda a self.param_names
            initial_params_list = [self.initial_params[name] for name in self.param_names]

            self.popt, self.pcov = curve_fit(
                self.model,
                self.x,
                self.y,
                p0=initial_params_list,
                sigma=self.sigma_y,
                absolute_sigma=True # Tratta sigma come deviazioni standard assolute
            )

            errors = np.sqrt(np.diag(self.pcov))
            self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.popt, errors)}

            # Calcola Chi2 manualmente
            residuals = self.y - self.model(self.x, *self.popt)
            # Evita divisione per zero se sigma_y è zero
            valid_sigma = self.sigma_y > 0
            chisq_terms = np.zeros_like(self.y)
            chisq_terms[valid_sigma] = (residuals[valid_sigma] / self.sigma_y[valid_sigma])**2
            self.chi2_val = np.sum(chisq_terms)

            self.dof = len(self.x) - len(self.popt)
            self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0
            self.is_fit_valid = True # curve_fit non ha un flag di validità diretto come Minuit, ma se non lancia eccezioni...

        except RuntimeError as e:
             print(f"Errore durante il fit con Scipy (curve_fit): {e}. Il fit potrebbe non essere converguto.")
             self.is_fit_valid = False
             self.popt = [self.initial_params[name] for name in self.param_names]
             self.pcov = np.full((len(self.param_names), len(self.param_names)), np.nan)
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             self.chi2_val = np.nan
             self.dof = len(self.x) - len(self.param_names)
             self.p_value = np.nan
        except Exception as e:
             print(f"Errore imprevisto durante il fit con Scipy: {e}")
             self.is_fit_valid = False
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             # Inizializza gli altri attributi a NaN o valori predefiniti
             self.popt, self.pcov, self.chi2_val, self.dof, self.p_value = None, None, None, None, None


    # --- Metodo Principale per Eseguire il Fit ---

    def perform_fit(self):
        """Esegue il fit utilizzando il metodo specificato in __init__."""
        print(f"--- Esecuzione Fit con Metodo: {self.fit_method} ---")
        if self.fit_method == 'ODR':
            self._fit_odr()
        elif self.fit_method == 'LeastSquares':
            self._fit_least_squares()
        elif self.fit_method == 'Chi2':
            self._fit_chi2()
        elif self.fit_method == 'Scipy':
            self._fit_scipy()
        else:
            # Questo non dovrebbe accadere grazie al check in __init__
            raise ValueError(f"Metodo di fit '{self.fit_method}' non riconosciuto.")

        if self.fit_result is None:
             print("Errore: Il fit non ha prodotto risultati.")
             self.is_fit_valid = False
        elif not self.is_fit_valid:
             print("Avviso: Il fit potrebbe non essere valido o non essere converguto.")
        else:
            print("Fit completato.")

        # Restituisce self per permettere il chaining, es. fit.perform_fit().print_results()
        return self

    # --- Metodi per Visualizzare i Risultati ---

    def print_results(self):
        """Stampa i risultati del fit (parametri, chi2, p-value)."""
        if self.fit_result is None:
            print("Errore: Nessun risultato del fit disponibile. Eseguire prima perform_fit().")
            return

        print(f"\n--- Risultati del Fit ({self.fit_method}) ---")
        print(f"Fit Valido: {'Sì' if self.is_fit_valid else 'No'}")

        print("\nParametri Ottimizzati:")
        for name in self.param_names:
            val, err = self.fit_result.get(name, (np.nan, np.nan)) # Gestisce il caso di fit fallito
            print(f"  {name} = {val:.4g} ± {err:.2g}") # Formato più leggibile

        if self.chi2_val is not None and self.dof is not None and self.dof > 0:
            chi2_rid = self.chi2_val / self.dof
            print(f"\nStatistiche del Fit:")
            print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
            print(f"  Gradi di libertà (DoF): {self.dof}")
            print(f"  Chi-quadro Ridotto (χ²/DoF): {chi2_rid:.4f}")
            if self.p_value is not None:
                print(f"  p-value: {self.p_value:.4f}")
        elif self.dof == 0:
             print("\nStatistiche del Fit:")
             print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
             print(f"  Gradi di libertà (DoF): {self.dof}")
             print("  Attenzione: Con 0 gradi di libertà, Chi2 ridotto e p-value non sono definiti.")
        else:
            print("\nStatistiche del Fit non disponibili (Chi2/DoF potrebbero non essere stati calcolati).")

        # Stampa informazioni aggiuntive da Minuit se disponibili
        if self.m and (self.fit_method == 'LeastSquares' or self.fit_method == 'Chi2'):
             # print(f"\nMinuit Fit Status: {self.m.fmin}") # fmin contiene info dettagliate
             if hasattr(self.m, 'covariance') and self.m.covariance is not None:
                  print("\nMatrice di Covarianza (Minuit):")
                  # Stampa la matrice in modo leggibile
                  # for row in self.m.covariance:
                  #      print("  [" + ", ".join(f"{x: .2e}" for x in row) + "]")
                  pass # Potrebbe essere troppo verboso, lo lascio commentato
             else:
                  print("\nMatrice di Covarianza (Minuit): non disponibile o non valida.")
        # Stampa matrice covarianza da Scipy
        elif self.pcov is not None and self.fit_method == 'Scipy':
             print("\nMatrice di Covarianza (Scipy):")
             # for row in self.pcov:
             #      print("  [" + ", ".join(f"{x: .2e}" for x in row) + "]")
             pass # Anche qui, potenzialmente verboso


    def _get_info_box_coords(self, position='upper left', pad=0.05):
        """Restituisce coordinate e allineamento per plt.annotate."""
        positions = {
            'upper left':   {'xy': (pad, 1 - pad), 'ha': 'left', 'va': 'top'},
            'upper right':  {'xy': (1 - pad, 1 - pad), 'ha': 'right', 'va': 'top'},
            'lower left':   {'xy': (pad, pad), 'ha': 'left', 'va': 'bottom'},
            'lower right':  {'xy': (1 - pad, pad), 'ha': 'right', 'va': 'bottom'},
            'upper center': {'xy': (0.5, 1 - pad), 'ha': 'center', 'va': 'top'},
            'lower center': {'xy': (0.5, pad), 'ha': 'center', 'va': 'bottom'},
            'center left':  {'xy': (pad, 0.5), 'ha': 'left', 'va': 'center'},
            'center right': {'xy': (1 - pad, 0.5), 'ha': 'right', 'va': 'center'},
            'center':       {'xy': (0.5, 0.5), 'ha': 'center', 'va': 'center'},
        }
        # Se viene passata una tupla (x, y), usala direttamente
        if isinstance(position, (tuple, list)) and len(position) == 2:
             return {'xy': tuple(position), 'ha': 'center', 'va': 'center'} # Default alignment for custom coords

        return positions.get(position.lower().replace("_", " "), positions['upper left']) # Default a upper left

    def plot_results(self, title_fontsize=14, label_fontsize=12,
                     info_box_pos='upper right', log_scale_y=False, log_scale_x=False):
        """
        Genera un grafico dei dati fittati con la curva di fit e un box informativo.

        Args:
            title_fontsize (int, optional): Dimensione del font per il titolo. Default: 14.
            label_fontsize (int, optional): Dimensione del font per le etichette degli assi. Default: 12.
            info_box_pos (str or tuple, optional): Posizione del box informativo.
                Può essere una stringa come 'upper left', 'center right', etc.,
                o una tupla (x, y) in coordinate relative agli assi (0-1).
                Default: 'upper right'.
            log_scale_y (bool, optional): Se True, imposta la scala logaritmica per l'asse y.
                                         Default: False.
        """
        if self.fit_result is None:
            print("Errore: Nessun risultato del fit disponibile per il plot. Eseguire prima perform_fit().")
            return
        if not self.is_fit_valid:
             print("Attenzione: Si sta plottando un fit non valido o non converguto.")

        plt.figure(figsize=(10, 7)) # Leggermente più alta per dare spazio
        ax = plt.gca() # Get current axes

        # --- Plot Dati ---
        plot_kwargs = {'fmt': 'o', 'label': 'Dati', 'markersize': 6, 'capsize': 4, 'elinewidth': 1.5}
        # Includi errori x se disponibili e significativi (o se ODR)
        if self.sigma_x is not None and np.any(self.sigma_x > 1e-9):
             ax.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y, **plot_kwargs)
             print("Plotting con errori su X e Y.")
        else:
             ax.errorbar(self.x, self.y, yerr=self.sigma_y, **plot_kwargs)

        # --- Plot Curva di Fit ---
        # Genera più punti per una curva liscia
        if len(self.x) > 1:
             x_min, x_max = np.min(self.x), np.max(self.x)
             # Estendi leggermente il range per non tagliare la curva ai bordi
             range_ext = (x_max - x_min) * 0.02
             x_fit = np.linspace(x_min - range_ext, x_max + range_ext, 500)
        else:
             # Gestisce caso con un solo punto dati
             x_fit = np.array([self.x[0] - 1, self.x[0], self.x[0] + 1]) # Un piccolo range intorno

        # Prendi i valori fittati (ignora gli errori qui)
        fitted_params_dict = {name: val_err[0] for name, val_err in self.fit_result.items()}
        try:
             y_fit = self.model(x_fit, **fitted_params_dict)
             ax.plot(x_fit, y_fit, '-r', label=f'Fit ({self.fit_method})', linewidth=2)
        except Exception as e:
             print(f"Errore nel calcolare la curva di fit per il plot: {e}")
             print("La curva di fit potrebbe non essere visualizzata.")


        # --- Impostazioni Grafico ---
        ax.set_xlabel(self.xlabel, fontsize=label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=label_fontsize)
        ax.set_title(self.title, fontsize=title_fontsize, pad=15) # Aumenta pad per spazio

        if log_scale_y:
            ax.set_yscale('log')
            # Aggiusta limiti y se necessario in scala log
            # Potrebbe essere necessario gestire y <= 0 nei dati
            min_y = np.min(self.y[self.y > 0]) if np.any(self.y > 0) else 1e-9
            # ax.set_ylim(bottom=min_y * 0.5) # Esempio di aggiustamento

        if log_scale_x:
            ax.set_xscale('log')
            min_x = np.min(self.x[self.x > 0]) if np.any(self.x > 0) else 1e-9

        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)
        ax.minorticks_on() # Abilita minor ticks

        # --- Box Informativo ---
        box_text_lines = []
        for name in self.param_names:
             val, err = self.fit_result.get(name, (np.nan, np.nan))
             # Usa notazione scientifica se necessario, altrimenti float più leggibile
             if abs(val) > 1e4 or abs(val) < 1e-3:
                  val_str = f"{val:.2e}"
             else:
                  val_str = f"{val:.3f}" # Aumenta precisione per float
             err_str = f"{err:.2g}" # Usa 'g' per precisione automatica sull'errore
             box_text_lines.append(f"${name} = {val_str} \\pm {err_str}$")

        if self.chi2_val is not None and self.dof is not None and self.dof > 0:
             chi2_rid_str = f"{self.chi2_val / self.dof:.3f}"
             box_text_lines.append(f"$\\chi^2/N_{{dof}} = {chi2_rid_str}$")
             if self.p_value is not None:
                  p_val_str = f"{self.p_value:.3f}"
                  box_text_lines.append(f"$p$-value $= {p_val_str}$")
        elif self.chi2_val is not None:
             # Mostra chi2 anche se DoF=0
             chi2_str = f"{self.chi2_val:.3f}"
             box_text_lines.append(f"$\\chi^2 = {chi2_str}$ (DoF=0)")


        info_text = "\n".join(box_text_lines)
        box_props = self._get_info_box_coords(info_box_pos)

        ax.annotate(info_text,
                    xy=box_props['xy'],
                    xycoords='axes fraction',
                    va=box_props['va'],
                    ha=box_props['ha'],
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.85, edgecolor='gray'),
                    fontsize=11, # Leggermente più piccolo per non sovrapporsi troppo
                    linespacing=1.4)

        ax.legend(fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Aggiusta layout per dare spazio al titolo
        plt.show()





"""Funzione per fare il test di Student, in xname basta mettere il nome che voglio printi"""

def Tstudent2(x, y, sigma_x, sigma_y, x_name="x", y_name="y"):
    T = np.abs((x - y) / np.sqrt(sigma_x**2 + sigma_y**2))
    df = (sigma_x**2 + sigma_y**2)**2 / ((sigma_x**4) + (sigma_y**4))
    p_value = 2 * (1 - t.cdf(T, df))
    
    print(f"Il p-value del test di compatibilità tra il valore di {x_name} e {y_name} vale: {p_value:.4f}")
    if p_value > 0.05:
        print("I due valori sono compatibili")
    else:
        print("I due valori NON sono compatibili")


"""Seconda funzione per il T-test, questa utilizza array, ha come ingresso array"""
def Tstudent3(val1, val2, sigma1, sigma2, val1_name="Valore 1", val2_name="Valore 2", use_ttest=False, custom_df=None, significance_level=0.05):
    """
    Esegue un test di compatibilità tra due valori con le loro incertezze.
    Confronta la differenza con l'incertezza combinata.

    Parametri:
    val1, val2: I due valori da confrontare.
    sigma1, sigma2: Le incertezze (deviazioni standard) associate a val1 e val2.
    val1_name, val2_name: Nomi descrittivi per i valori (per la stampa).
    use_ttest: Booleano. Se True, usa la distribuzione t di Student.
                 Se False (default), usa la distribuzione Normale standard.
    custom_df: Gradi di libertà da usare se use_ttest=True.
                 Se None e use_ttest=True, tenta di calcolare df con l'approssimazione
                 di Welch-Satterthwaite (adattata per singole misure, con cautela).
                 Se use_ttest=False, custom_df è ignorato.
    significance_level: Livello di significatività alfa (default 0.05 per il 95% di confidenza).

    Ritorna:
    T_score: Il valore T (o Z per la normale).
    p_value: Il p-value a due code.
    compatibili: Booleano (True se p_value > significance_level).
    """
    # Inizializziamo le variabili che potrebbero non essere definite in tutti i percorsi
    p_value = np.nan
    compatibili = False
    dist_type = "N/A"
    T_score = np.nan # Inizializza anche T_score
    df_to_print = "N/A" # Per stampare i df usati

    if sigma1 < 0 or sigma2 < 0:
        print(f"Errore nel test tra {val1_name} e {val2_name}: Le incertezze (sigma) non possono essere negative.")
        return T_score, p_value, compatibili
        
    denominatore_T = np.sqrt(sigma1**2 + sigma2**2)

    if denominatore_T < 1e-15: # Praticamente zero
        if np.abs(val1 - val2) < 1e-15: # Anche la differenza è zero
            T_score = 0.0
            p_value = 1.0
            compatibili = True
            dist_type = "N/A (Valori identici con sigma zero)"
        else: # Differenza non zero ma sigma del denominatore sì
            T_score = np.inf
            p_value = 0.0
            compatibili = False
            dist_type = "N/A (Valori diversi con sigma denominatore zero)"
        
        print(f"\nTest di compatibilità tra {val1_name} ({val1:.3e} ± {sigma1:.2e}) e {val2_name} ({val2:.3e} ± {sigma2:.2e}):")
        print(f"  Incertezza combinata sul denominatore è quasi zero.")
        print(f"  Differenza: {np.abs(val1 - val2):.2e}")
        print(f"  T-score (o Z-score): {T_score:.2f}")
        print(f"  P-value (due code): {p_value:.4f}")
        if compatibili:
            print(f"  I due valori ({val1_name} e {val2_name}) sono considerati identici (entro la precisione numerica).")
        else:
            print(f"  I due valori ({val1_name} e {val2_name}) sono DIVERSI e/o le incertezze sul denominatore sono nulle/trascurabili.")
        return T_score, p_value, compatibili

    T_score = np.abs(val1 - val2) / denominatore_T

    if use_ttest:
        df = np.nan # Inizializza df per il t-test
        if custom_df is not None:
            df = custom_df
            if df <= 0:
                print(f"Errore nel test tra {val1_name} e {val2_name}: custom_df per t-test ({df}) deve essere > 0.")
                return T_score, p_value, compatibili # p_value e compatibili rimangono inizializzati a NaN/False
        else:
            # Approssimazione di Welch-Satterthwaite per i gradi di libertà (adattata)
            # num_df = (sigma1**2/n1 + sigma2**2/n2)**2
            # den_df_part1 = ((sigma1**2/n1)**2)/(n1-1)
            # den_df_part2 = ((sigma2**2/n2)**2)/(n2-1)
            # Per singole misure, possiamo considerare n1=1, n2=1.
            # Questo rende i denominatori n-1 uguali a 0, il che non va bene.
            # La formula che avevi tu è un'approssimazione diversa:
            # df = (sigma_x**2 + sigma_y**2)**2 / ((sigma_x**4 / (nx-1)) + (sigma_y**4 / (ny-1)))
            # Se assumiamo che le sigma siano "ben note" (come se nx e ny fossero grandi),
            # i denominatori (nx-1) diventano grandi.
            # La tua formula era: df = (sigma_x**2 + sigma_y**2)**2 / (sigma_x**4 + sigma_y**4)
            # Usiamo questa, con cautela e menzionando che è un'approssimazione.
            
            s1_sq = sigma1**2
            s2_sq = sigma2**2
            
            if s1_sq < 1e-15 and s2_sq < 1e-15: # Entrambe le varianze sono zero
                 # Questo caso dovrebbe essere già gestito da denominatore_T < 1e-15
                 df = np.inf # o gestito come errore
            else:
                df_num = (s1_sq + s2_sq)**2
                df_den = s1_sq**2 + s2_sq**2 # Se una sigma è 0, il suo termine s^4/ (n-1) non conta.
                                          # Se n-1 fosse 1, allora è s1^4 + s2^4
                if df_den < 1e-15: # Evita divisione per zero se entrambe le sigma quadrate sono zero
                    print(f"Attenzione nel test tra {val1_name} e {val2_name}: Denominatore per df è zero nel t-test. Impossibile calcolare df robustamente.")
                    return T_score, p_value, compatibili
                else:
                    df = df_num / df_den
        
        df_to_print = f"{df:.2f}"
        if np.isinf(df): # Se df è infinito (può accadere se una sigma è zero e l'altra no con la formula usata)
            p_value = 2 * (1 - norm.cdf(T_score))
            dist_type = f"t di Student (df=infinito, equivale a Normale)"
        elif df < 1:
            # scipy.stats.t.cdf generalmente si aspetta df >= 1.
            # Alcune fonti suggeriscono di usare df=1 o di non usare la t-distribuzione.
            print(f"Attenzione nel test tra {val1_name} e {val2_name}: df calcolato ({df:.2f}) < 1. Il p-value del t-test potrebbe non essere affidabile.")
            # Si potrebbe decidere di non calcolare p_value o usare un fallback
            p_value = np.nan # In questo caso, è meglio non dare un p-value
            dist_type = f"t di Student (df={df:.2f} - problematico)"
        else:
            p_value = 2 * (1 - t.cdf(T_score, df))
            dist_type = f"t di Student (df={df:.2f})"

        if np.isnan(p_value):
            print(f"Attenzione nel test tra {val1_name} e {val2_name}: p_value calcolato come NaN con t-test. Verificare df e T_score.")
            # compatibili rimane False (dall'inizializzazione)
        else:
            compatibili = p_value > significance_level

    else: # Usa la distribuzione Normale standard
        df_to_print = "infinito (Normale)"
        p_value = 2 * (1 - norm.cdf(T_score))
        dist_type = "Normale Standard"
        compatibili = p_value > significance_level
        
    print(f"\nTest di compatibilità tra {val1_name} ({val1:.3e} ± {sigma1:.2e}) e {val2_name} ({val2:.3e} ± {sigma2:.2e}):")
    print(f"  Differenza: {np.abs(val1 - val2):.2e}")
    print(f"  Incertezza sulla differenza (denominatore di T): {denominatore_T:.2e}")
    print(f"  Statistica del test (T o Z): {T_score:.2f}")
    print(f"  Distribuzione usata: {dist_type}")
    if 'df=' in dist_type and not np.isinf(df if 'df' in locals() else np.nan) : # Stampa df solo se è stato calcolato e non è infinito
         pass # df_to_print è già impostato
    print(f"  P-value (due code): {p_value:.4f}")
    
    if compatibili:
        print(f"  I due valori sono COMPATIBILI (p > {significance_level}).")
    else:
        if np.isnan(p_value):
             print(f"  Impossibile determinare la compatibilità (p-value è NaN).")
        else:
             print(f"  I due valori NON sono compatibili (p <= {significance_level}).")
        
    return T_score, p_value, compatibili




"""Funzione per fare lo Z-test"""
def Ztest(x, y, sigma_x, x_name='x', y_name='y'):
    Z = np.abs((x - y) / sigma_x)

    p_value = 2 * (1 - sc.norm.cdf(Z))
    
    print(f"Il p-value del test di compatibilità tra il valore di {x_name} e {y_name} vale: {p_value:.4f}")
    if p_value > 0.05:
        print("I due valori sono compatibili")
    else:
        print("I due valori NON sono compatibili")


"""Funzione per fare lo scatter"""

def scatter_plot_with_error(x, y, sigma_y, xlabel, ylabel, title, sigma_x=None, axhline_value=None):
    """
    Crea uno scatter plot dei dati con barre d'errore e linea connettente tra i punti.
    
    Parametri:
      x: array-like, dati dell'asse x
      y: array-like, dati dell'asse y
      sigma_y: array-like, errori associati a y
      sigma_x: array-like, errori associati a x (opzionale)
      axhline_value: float, valore y dove disegnare una linea orizzontale (default: None, nessuna linea)
      xlabel: string, etichetta per l'asse x
      ylabel: string, etichetta per l'asse y
      title: string, titolo del grafico
    """

    plt.figure(figsize=(10, 5), dpi=100)
    plt.style.use('seaborn-v0_8-notebook')

    plt.errorbar(
        x,
        y,
        xerr=sigma_x,
        yerr=sigma_y,
        fmt='-',
        color='purple',             # colore della linea connettente
        ecolor='orange',               # colore delle barre di errore
        elinewidth=1.5,               # spessore delle linee degli errori
        capsize=4,                  
        alpha=0.8,
        zorder=1
    )

    sc = plt.scatter(
        x,
        y,
        c=np.abs(y),
        cmap='viridis',
        s=45,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        zorder=3
    )

    if axhline_value is not None:
        plt.axhline(axhline_value, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)

    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Barra dei colori
    cbar = plt.colorbar(sc)
    cbar.set_label('Ampiezza (V)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.show()


def scatter_plot_log_with_error(x, y, sigma_y, xlabel, ylabel, title, sigma_x=None, axhline_value=None):
    """
    Crea uno scatter plot dei dati con barre d'errore e linea connettente tra i punti,
    con scala bilogaritmica (log-log).
    
    Parametri:
      x: array-like, dati dell'asse x
      y: array-like, dati dell'asse y
      sigma_y: array-like, errori associati a y
      sigma_x: array-like, errori associati a x (opzionale)
      axhline_value: float, valore y dove disegnare una linea orizzontale (default: None, nessuna linea)
      xlabel: string, etichetta per l'asse x
      ylabel: string, etichetta per l'asse y
      title: string, titolo del grafico
    """
    plt.figure(figsize=(10, 5), dpi=100)
    plt.style.use('seaborn-v0_8-notebook')
    
    # Imposta entrambi gli assi in scala logaritmica
    plt.xscale('log')
    plt.yscale('log')
    
    plt.errorbar(
        x,
        y,
        xerr=sigma_x,
        yerr=sigma_y,
        fmt='-',
        color='purple',             # colore della linea connettente
        ecolor='orange',            # colore delle barre di errore
        elinewidth=1.5,             # spessore delle linee degli errori
        capsize=4,                  
        alpha=0.8,
        zorder=1
    )
    sc = plt.scatter(
        x,
        y,
        c=np.abs(y),
        cmap='viridis',
        s=45,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        zorder=3
    )
    if axhline_value is not None:
        plt.axhline(axhline_value, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    
    # Configurazione della griglia specifica per i grafici logaritmici
    plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    
    # Configurazione dei tick per scale logaritmiche
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Barra dei colori
    cbar = plt.colorbar(sc)
    cbar.set_label('Ampiezza (V)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.show()



"""
Funzione per fare i fit con ODR
ESEMPIO:
def V_Ohm(I, R, q):
  return R*I + q

# Prepara il dizionario dei dati e dei parametri iniziali
data_arrays = {'x': x, 'y': y, 'sigma_y': sy * np.ones_like(y), 'sigma_x': sx * np.ones_like(x)}
initial_params = {'R': 10.0, 'q': 0.0}

# Istanzia la classe per il fit ODR
fit_odr = FitODR(V_Ohm, data_arrays, initial_params,
                               xlabel="x", ylabel="y", title="Fit lineare con ODR")

# Esegue il fit
fit_odr.perform_fit()

# Stampa i risultati
fit_odr.print_results()

# Visualizza il grafico dei dati e del fit
fit_odr.plot_results()
"""


class FitODR:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come keyword (es. def model(x, a, b): …)
        data_arrays: dizionario con 'x', 'y', 'sigma_y' e opzionalmente 'sigma_x'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma_y = data_arrays.get('sigma_y', np.ones_like(self.y))
        self.sigma_x = data_arrays.get('sigma_x', np.zeros_like(self.x))  # se non fornito, si assume zero
        
        # Estrae i nomi dei parametri dalla firma della funzione (esclude x)
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        self.param_names = all_param_names[1:]
        
        self.initial_params = initial_params
        self.fit_result = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma_y):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza")
        if len(self.x) != len(self.sigma_x):
            raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza")
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("I nomi dei parametri iniziali non corrispondono a quelli della funzione modello")
    
    def _odr_model(self, B, x):
        """
        Funzione wrapper per ODR.
        B: array dei parametri, nell'ordine definito da self.param_names
        x: array (o array 2D) delle variabili indipendenti
        """
        # Mappa l'array B in un dizionario con i nomi dei parametri
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)
    
    def perform_fit(self):
        beta0 = [self.initial_params[name] for name in self.param_names]
        
        odr_model = Model(self._odr_model)
        
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        
        odr = ODR(data, odr_model, beta0=beta0)
        output = odr.run()
        
        self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, output.beta, output.sd_beta)}
        self.odr_output = output 
        
        return output
    
    def print_results(self):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
            
        print("Risultati del fit (ODR):")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.odr_output.sum_square
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"Gradi di libertà: {dof}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
        
        plt.figure(figsize=(10, 6))

        plt.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y,
                     fmt='o', label='Dati', markersize=7, capsize=4)
        

        x_fit = np.linspace(np.min(self.x), np.max(self.x), 500)

        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        y_fit = self.model(x_fit, **params_dict)
        plt.plot(x_fit, y_fit, '-r', label='Fit (ODR)', linewidth=2.5)
        
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box con informazioni sui risultati
        text_lines = [f"${name} = {val:.2e} \\pm {err:.2e}$" 
                      for name, (val, err) in self.fit_result.items()]
        dof = len(self.x) - len(self.param_names)
        chi2_red = self.odr_output.sum_square / dof
        text_lines.append(f"$\\chi^2/NdoF = {chi2_red:.3f}$")
        text = "\n".join(text_lines)
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7', edgecolor='gray'),
                     fontsize=13, 
                     linespacing=1.5)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class FitODR_2:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come keyword (es. def model(x, a, b): …)
        data_arrays: dizionario con 'x', 'y', 'sigma_y' e opzionalmente 'sigma_x'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma_y = data_arrays.get('sigma_y', np.ones_like(self.y))
        self.sigma_x = data_arrays.get('sigma_x', np.zeros_like(self.x))  # se non fornito, si assume zero
        
        # Estrae i nomi dei parametri dalla firma della funzione (esclude x)
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        self.param_names = all_param_names[1:]
        
        self.initial_params = initial_params
        self.fit_result = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma_y):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza")
        if len(self.x) != len(self.sigma_x):
            raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza")
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("I nomi dei parametri iniziali non corrispondono a quelli della funzione modello")
    
    def _odr_model(self, B, x):
        """
        Funzione wrapper per ODR.
        B: array dei parametri, nell'ordine definito da self.param_names
        x: array (o array 2D) delle variabili indipendenti
        """
        # Mappa l'array B in un dizionario con i nomi dei parametri
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)
    
    def perform_fit(self):
        beta0 = [self.initial_params[name] for name in self.param_names]
        
        odr_model = Model(self._odr_model)
        
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        
        odr = ODR(data, odr_model, beta0=beta0)
        output = odr.run()
        
        self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, output.beta, output.sd_beta)}
        self.odr_output = output 
        
        return output
    
    def print_results(self):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
            
        print("Risultati del fit (ODR):")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.odr_output.sum_square
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"Gradi di libertà: {dof}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
        
        plt.figure(figsize=(10, 6))

        plt.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y,
                     fmt='o', label='Dati', markersize=7, capsize=4)
        

        x_fit = np.linspace(np.min(self.x), np.max(self.x), 500)

        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        y_fit = self.model(x_fit, **params_dict)
        plt.plot(x_fit, y_fit, '-r', label='Fit (ODR)', linewidth=2.5)
        
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box con informazioni sui risultati
        text_lines = [f"${name} = {val:.2e} \\pm {err:.2e}$" 
                      for name, (val, err) in self.fit_result.items()]
        dof = len(self.x) - len(self.param_names)
        chi2_red = self.odr_output.sum_square / dof
        text_lines.append(f"$\\chi^2/NdoF = {chi2_red:.3f}$")
        text = "\n".join(text_lines)
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7', edgecolor='gray'),
                     fontsize=13, 
                     linespacing=1.5)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


"""
Funzione per fittare e plottare utilizzando i minimi quadrati
ESEMPIO:
x_data = I_bobina
y_data = array_funz
sigma_y = sigma_ang

# Configurazione del fit
data_dict = {
    'x': x_data,
    'y': y_data,
    'sigma_y': sigma_y
}

initial_params = {
    'Be': 1e-8,
    'b': 0.0
}

# Esegui il fit
fit = FitMinimiQuadrati(
    bobina,
    data_dict,
    initial_params,
    xlabel="Corrente (A)",
    ylabel="sos (V)",
    title="Livio Laido"
)
fit.perform_fit()
fit.print_results()
fit.plot_results(
    title_fontsize=16,
    label_fontsize=12
)
"""

class FitMinimiQuadrati:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
    
    def perform_fit(self):
        least_squares = LeastSquares(self.x, self.y, self.sigma, self.model) 
        self.m = Minuit(least_squares, **self.initial_params)
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[name], self.m.errors[name]) 
                          for name in self.param_names}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 500)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                va='top', 
                ha='left', 
                bbox=dict(
                    facecolor='white',
                    alpha=0.9,
                    boxstyle='round,pad=0.7',  
                    edgecolor='gray'
                ),
                fontsize=13,  
                linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()


"""
Funzione per fare il fit con il chi2
"""

class FitChi2:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        #print("Parametri rilevati nella funzione:", self.param_names)
        #print("Parametri forniti:", initial_params.keys()) 

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
        
        missing = set(self.param_names) - set(self.initial_params.keys())
        if missing:
          raise ValueError(f"Parametri mancanti: {missing}")
    
    def chi2_function(self, *args):
        params = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params)
        return np.sum(((self.y - y_model) / self.sigma)**2)

    def perform_fit(self):       
        self.m = Minuit(self.chi2_function, *self.initial_params.values())
        self.m.errordef = 1.0
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[i], self.m.errors[i]) 
                          for i, name in enumerate(self.param_names)}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(
                              facecolor='white',
                              alpha=0.9,
                              boxstyle='round,pad=0.7',  
                              edgecolor='gray'
                              ),
                     fontsize=13,  
                     linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()



class FitChi2_0:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        #print("Parametri rilevati nella funzione:", self.param_names)
        #print("Parametri forniti:", initial_params.keys()) 

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
        
        missing = set(self.param_names) - set(self.initial_params.keys())
        if missing:
          raise ValueError(f"Parametri mancanti: {missing}")
    
    def chi2_function(self, *args):
        params = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params)
        return np.sum(((self.y - y_model) / self.sigma)**2)

    def perform_fit(self):       
        self.m = Minuit(self.chi2_function, *self.initial_params.values())
        self.m.errordef = 1.0
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[i], self.m.errors[i]) 
                          for i, name in enumerate(self.param_names)}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(
                              facecolor='white',
                              alpha=0.9,
                              boxstyle='round,pad=0.7',  
                              edgecolor='gray'
                              ),
                     fontsize=13,  
                     linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()



class FitScipy:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come argomenti posizionali
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        # Mappatura parametri -> ordine per curve_fit
        self.initial_params_list = [initial_params[name] for name in self.param_names]
        
        self.fit_result = None
        self.cov_matrix = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self):
        """Esegue il fit usando curve_fit"""
        popt, pcov = curve_fit(
            self.model,
            self.x,
            self.y,
            p0=self.initial_params_list,
            sigma=self.sigma,
            absolute_sigma=True
        )
        
        self.fit_result = {name: (val, np.sqrt(pcov[i,i])) 
                          for i, (name, val) in enumerate(zip(self.param_names, popt))}
        self.cov_matrix = pcov

        residuals = self.y - self.model(self.x, *popt)
        self.chi2_val = np.sum((residuals / self.sigma)**2)
        self.dof = len(self.x) - len(popt)
        
    def print_results(self):
        """Stampa i risultati del fit in formato leggibile"""
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        print(f"\nChi-quadro ridotto: {self.chi2_val/self.dof:.3f}")
        print(f"Gradi di libertà: {self.dof}")
        print(f"p-value: {1 - chi2.cdf(self.chi2_val, self.dof):.3f}")

    def plot_results(self, title_fontsize=14, label_fontsize=12):
        """Genera il plot dei risultati"""
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)

        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        params = [self.fit_result[name][0] for name in self.param_names]
        y_fit = self.model(x_fit, *params)
        
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/N_{{doF}} = {self.chi2_val/self.dof:.3f}$"
        
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7'),
                     fontsize=13)
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()




class FitScipy2_0:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come argomenti posizionali
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params_list = [initial_params[name] for name in self.param_names]
        
        self.fit_result = None
        self.cov_matrix = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self):
        """Esegue il fit usando curve_fit"""
        popt, pcov = curve_fit(
            self.model,
            self.x,
            self.y,
            p0=self.initial_params_list,
            sigma=self.sigma,
            absolute_sigma=True
        )
        
        self.fit_result = {name: (val, np.sqrt(pcov[i,i])) 
                          for i, (name, val) in enumerate(zip(self.param_names, popt))}
        self.cov_matrix = pcov
        
        # Calcola chi2 ridotto
        residuals = self.y - self.model(self.x, *popt)
        self.chi2_val = np.sum((residuals / self.sigma)**2)
        self.dof = len(self.x) - len(popt)
        
    def print_results(self):
        """Stampa i risultati del fit in formato leggibile"""
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        print(f"\nChi-quadro ridotto: {self.chi2_val/self.dof:.3f}")
        print(f"Gradi di libertà: {self.dof}")
        print(f"p-value: {1 - chi2.cdf(self.chi2_val, self.dof):.3f}")

    def plot_results(self, title_fontsize=14, label_fontsize=12):
        """Genera il plot dei risultati"""
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        
        # Genera curva di fit
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        params = [self.fit_result[name][0] for name in self.param_names]
        y_fit = self.model(x_fit, *params)
        
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/N_{{doF}} = {self.chi2_val/self.dof:.3f}$"
        
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7'),
                     fontsize=13)
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()







"""
Formule per trovare zeri, massimi e minimi e integrali per via numerica"""

def bisezione_iterativa(f, a, b, tolleranza=1e-6, max_iter=100):
    """
    Bisezione usa il metodo di bisezione per trovare lo zero di funzione sfruttando 
    il teorema degli zeri
    Argomenti = la funzione, gli estremi 'a' e 'b', la tolleranza che posso impostare

    Return = valore medio dell'interallo finale
    """
    if f(a) * f(b) >= 0:
        raise ValueError("La funzione deve avere segni opposti ai due estremi dell'intervallo [a, b].")
    iterazioni = 0
    while (b - a) / 2 > tolleranza and iterazioni < max_iter:
        c = (a + b) / 2  
        if f(c) == 0:    #abbiamo trovato lo zero esatto
            return c
        elif f(a) * f(c) < 0:
            b = c       #lo zero è nell'intervallo [a, c]
        else:
            a = c       #lo zero è nell'intervallo [c, b]
        iterazioni += 1
    return (a + b) / 2 
    

def bisezione_ricorsiva(f, a, b, tolleranza=1e-6):
    """
    Metodo ricorsivo per trovare lo zero di una funzione usando il metodo di bisezione.
    Lancia un'eccezione se l'intervallo fornito non contiene uno zero.
    """
    if f(a) * f(b) > 0:
        raise ValueError("L'intervallo fornito non contiene uno zero (segni uguali ai due estremi).")
    c = (a + b) / 2  # Punto medio
    if abs(f(c)) < tolleranza or abs(b - a) / 2 < tolleranza:
        return c
    if f(a) * f(c) < 0:
        if f(a) * f(b) > 0:
            raise ValueError("Intervallo non valido durante la ricorsione.")
        return bisezione_ricorsiva(f, a, c, tolleranza)
    else:
        if f(a) * f(b) > 0:
            raise ValueError("Intervallo non valido durante la ricorsione.")
        return bisezione_ricorsiva(f, c, b, tolleranza)


def minimi_sezioneAurea(f, a, b, tolleranza=1e-6):
  phi = (1 + np.sqrt(5)) / 2
  x1 = a + (b-a)/(phi**2)
  x2 = a + (b-a)/(phi)
  if (b-a) < tolleranza:
    min = (a + b) / 2
    minf = f(min)
    return [min, minf]
  elif f(x1) < f(x2):
    return minimi_sezioneAurea(f, a, x2, tolleranza)
  else:
    return minimi_sezioneAurea(f, x1, b, tolleranza)


def massimi_sezioneAurea(f, a, b, tolleranza=1e-6):
    phi = (1 + np.sqrt(5)) / 2
    x1 = a + (b-a)/(phi**2)
    x2 = a + (b-a)/(phi)
    if (b-a) < tolleranza:
        max = (a + b) / 2
        maxf = f(max)
        return [max, maxf]
    elif f(x1) > f(x2):
        return massimi_sezioneAurea(f, a, x2, tolleranza)
    else:
        return massimi_sezioneAurea(f, x1, b, tolleranza)


def hit_or_miss_integrate(f, a, b, m, M, N):
    """
    Argomenti: funzione f, [a, b] base rettangolo, [m, M] altezza rettangolo,
    N numero di x e y generati
    """
    x = np.random.uniform(a, b, N)
    y = np.random.uniform(m, M, N)
    hits = 0
    for i in range(N):
        if y[i] <= f(x[i]):
            hits += 1
    successi = np.sum(hits)
    A_rect = (b - a) * (M-m)
    p = successi / N
    integral = p * A_rect
    std_integral = np.sqrt(((A_rect**2) / N) * p * (1 - p))
    return integral, std_integral 


def crude_montecarlo_integrate(f, a, b, N):
    """
    Montecarlo Method stima numericamente l'integrale sfruttando E[f(x)]
    Argomenti: la funzione, gli estremi e il numero di elementi N
    """
    x = np.linspace(a, b, N)
    f_values = f(x)  
    mean = np.mean(f_values)
    std_dev = np.std(f_values)
    integral = (b - a) * mean
    err = (b - a) * std_dev / np.sqrt(N)
    return integral, err


def MC_bidimensionale(f, xmin, xmax, ymin, ymax, M, N):
  """
  La funzione calcola un integrale di una funzione in due variabili
  """
  x = np.random.uniform(xmin, xmax, N)
  y = np.random.uniform(ymin, ymax, N)
  z = np.random.uniform(0, M, N)
  hits = z <= f(x, y)
  successi = np.sum(hits)
  A_rect = (xmax - xmin) * (ymax - ymin) * M
  p = successi / N
  integral = p * A_rect
  std_integral = np.sqrt(((A_rect**2) / N) * p * (1 - p))
  return integral, std_integral


"""
Funzioni per generare numeri pseudo casuali
"""

def generatore_casual_pdf(pdf, n, xmin, xmax, N):
	"""
  	Questa funzione prende in ingresso: una pdf qualunque, 
	massimo e minimo del dominio delle x e quanto voglio grande questo dominio N
	e infine quanti numeri n piccolo voglio generare"""
	
	x = np.linspace(xmin, xmax, N)
	pdf_values = pdf(x)
	pdf_values /= np.trapz(pdf_values) #normalizzo la pdf con l'integrale

	cdf = np.cumsum(pdf_values)
	cdf /= cdf[-1]  

	random_numbers = np.random.uniform(0, 1, size=n)
	samples = np.interp(random_numbers, cdf, x)  #interp mi trasforma i dati con la cdf
	return samples, x, pdf_values


def TCL(media, sigma, Neventi):                
    """
    Funzione che genera eventi gaussiani con il TCL conosciuti media e sigma
    """
    risultati = [] 
    for i in range(Neventi):
        eventi = 0
        delta = np.sqrt(3*Neventi)*sigma
        xmin = media - delta
        xmax = media + delta
        for i in range (Neventi):
            eventi += np.random.uniform(xmin, xmax)
        eventi /= Neventi
        risultati.append(eventi) 
    return np.array(risultati) 


def inversa_exponential(t0, N):
	"""
	Funzione che mi genera N numeri distribuiti esponenzialmente con lambda=1/t0
	"""
	u = np.random.uniform(0, 1, N)
	f = -t0 * np.log(1 - u)
	return f


def generatore_poisson(t, N):                   
	#il contatore mi restituisce un numero che è sostanzialmente per quante volte moltiplico p per un numero casuale
	risultati = []
	for i in range(N):
		contatore = 0
		p = 1
		while p > np.exp(-t):
			contatore += 1
			p *= np.random.uniform(0, 1)     	#p mi accumula il prodotto di numeri casuali fino a quando questo non è minore del tempo tra eventi successivi
                                          			#di una distribuzione esponenziale, quindi sostanzialmente il numero di eventi tra due eventi successivi
		risultati.append(contatore - 1)
	return np.array(risultati)


def generatore_gaussiani(Ntoy, Neventi):      
    """
    Argomenti:
    - Ntoy (int): Numero di simulazioni da generare.
    - Neventi (int): Numero di eventi per ogni simulazione.

    Ritorna:
    - medie (list): Lista delle medie calcolate per ogni simulazione.
    - sigma (list): Lista delle stime dell'errore sulla media per ogni simulazione.
    """
    medie = []
    for _ in range(Ntoy):
        eventi = np.random.uniform(0, 1, Neventi)  # Genera eventi uniformi
        toy_stats = np.mean(eventi)               # Calcola la media
        medie.append(toy_stats)                   # Aggiungi alla lista delle medie

    sigma_valore = np.std(np.random.uniform(0, 1, Neventi)) / np.sqrt(Neventi)
    sigma = [sigma_valore] * Ntoy  # Stima costante per ogni toy

    return medie, sigma


def hit_or_miss_generator(f, a, b, m, M, N):
  """
  Argomenti: 
  f = funzione, [a, b] = estremi della base del rettangolo, 
  [m, M] = estremi altezza del rettangolo (spesso m = 0), N = numeri da generare
  """
  sample = []
  hits = 0
  x_values = np.random.uniform(a, b, N)
  y_values = np.random.uniform(m, M, N)
  for i in range(N):
    if y_values[i] <= f(x_values[i]):
      sample.append(x_values[i])
      hits += 1
  Area_rettangolo = (hits/N)*(b-a)*(M-m)
  return np.array(sample), Area_rettangolo


def try_and_catch_generator(f, a, b, M, N):
    """
    Argomenti: 
    f: la pdf, funzione da campionare
    a, b: insieme di generazione (limiti di campionamento)
    M: valore tale che la pdf è minore di M (fattore di normalizzazione)
    N: numero di eventi da generare

    Return: sample di eventi generati tramite il metodo Try-and-Catch
    """
    sample = []
    attempts = 0  
    while len(sample) < N:  
        x = np.random.uniform(a, b)  
        u = np.random.uniform(0, 1)  
        if u <= f(x)/M:  
            sample.append(x)  
        attempts += 1
        if attempts >= 100000:  # Massimo numero di tentativi
            print("Maximum attempts reached!")
            break
    return np.array(sample)


def try_and_catch_generator_v2(f, a, b, M, N):
  """
  Argomenti: pdf f, (a, b) insieme di generazione, M=valore tale per cui
  f è minore di M, N=numero eventi da generare

  Return: sample di eventi
  """
  sample = []
  x = np.random.uniform(a, b, N)
  u = np.random.uniform(0, 1, N)
  for i in range(N):
    if u[i] <= f(x[i])/M:
      sample.append(x[i])
  return sample


def TCL_pdf(f, a, b, Neventi):
  """
  Argomenti: una pdf f, (a,b) intervallo di generazione
  """
  sample = []
  n = 10
  for i in range(Neventi):
    media = np.mean(mr.generatore_casual_pdf(f, n, a, b, n))
    sample.append(media)
  return np.array(sample)


def generate_gauss_Box_Muller(Neventi, mu, sigma):
  """
  La funzione genera Neventi gaussiani con media mu e dev sigma
  """
  sample = []
  g1_tot = []
  g2_tot = []
  transform = lambda x : x * sigma + mu 
  #Moltiplicare per sigma mi trasforma la larghezza da 1 a sigma e aggiungo mu per traslare
  for i in range(N):
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    p = np.sqrt(-2*np.log(x1))
    g1 = p*np.cos(2*np.pi*x2)
    g2 = p*np.sin(2*np.pi*x2)
    g1_tot.append(transform(g1))
    g2_tot.append(transform(g2))
  sample = np.concatenate([g1_tot, g2_tot])
  return sample


def generate_gauss_bm(Neventi):
  """
  La funzione genera Neventi gaussiani
  """
  sample = []
  g1_tot = []
  g2_tot = []
  for i in range(N):
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    p = np.sqrt(-2*np.log(x1))
    g1 = p*np.cos(2*np.pi*x2)
    g2 = p*np.sin(2*np.pi*x2)
    g1_tot.append(g1)
    g2_tot.append(g2)
  sample = np.concatenate([g1_tot, g2_tot])
  return sample

def uniform(media, sigma, N):
    
    l = sigma*(np.sqrt(12))         
    a = media - l/2           
    b = media + l/2
    x = np.random.uniform(a, b, N)
    return x

def sturges (n):		#STURGES PER IL NUMERO DI BIN
	return int(np.ceil(1+3.322*np.log(n)))


def two_data_binning(data1, data2):
	"""Questa funzione, dati due set di dati diversi, mi trova
	il binning ottimale per plottare l'istogramma"""

	xMin = floor (min (min (data1), min (data2)))
	xMax = ceil (max (max (data1), max (data2)))
	N_bins = sturges (min (len (data1), len (data2)))
	bin_edges = np.linspace (xMin, xMax, N_bins)
	return bin_edges


def one_data_binning(data):
  	"""
	Preso un array di dati, mi produce i binedges e il bin_content;
  	Se voglio i primi N elementi, devo fare data_new = data[:N]
  	"""
  	N = len(data)
  	Nbins = sturges(N)
  	binnaggio, binedges = np.histogram(data, bins = Nbins, range = (min(data), max(data)))
  	return binnaggio, binedges

def loglikelihood(pdf, theta, sample):
    """
    Calcola il logaritmo della likelihood dato un parametro theta (es. media) e un set di dati (sample).
    Argomenti:
      - pdf: funzione di densità di probabilità.
      - theta: parametro singolo (es. media).
      - sample: array di dati osservati.
    Ritorna:
      - log-likelihood (float).
    """
    logL = 0
    for x in sample:
      if (pdf(x, theta)) > 0:
        logL = logL + np.log(pdf(x, theta))
      else:
        raise ValueError(f"PDF value is zero or negative at x={x}, theta={theta}.")
    return logL


def loglikelihood(pdf, theta, sample):    #Logaritmo della likelihood
	risultato = 0
	for x in sample:
		if (pdf(x, theta) > 0):
			risultato += np.log(pdf(x, theta))
	return (risultato)


def maximumlikelihood(loglikelihood, pdf, sample, a, b, tolleranza=1e-6):
	phi = (1 + np.sqrt(5)) / 2
	x1 = a + (b - a) / phi**2
	x2 = a + (b - a) / phi

	while (b - a) > tolleranza:
		L1 = loglikelihood(pdf, x1, sample)
		L2 = loglikelihood(pdf, x2, sample)
		if L1 > L2:
			b = x2
			x2 = x1
			x1 = a + (b - a) / phi**2
		else:
			a = x1
			x1 = x2
			x2 = a + (b - a) / phi
	x_max = (a + b) / 2
	return x_max

# Calcola integrale [Scipy]
def integral_scipy(f, a, b) : 
  integral = quad(f, a,b)
  return integral[0], integral[1]

def media_pesata(x, sigma) :
  m = np.sum(x/sigma**2)/np.sum(1/sigma**2)
  sigma_m = 1/np.sqrt(np.sum(1/sigma**2))
  return m, sigma_m

# PDF's & CDF's
def Gaussian(x, mu = 0, sigma = 1) :
	return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Gaussiana standardizzata
def Gaussian_standard(z):
  return (1/np.sqrt(2*np.pi))*np.exp((-z**2)/2) 

def Gaussian_cdf_ext(bin_edges, s, mu, sigma) :
  return s*norm.cdf(bin_edges, mu, sigma)

def Gaussian_cdf(bin_edges, mu, sigma) :
  return norm.cdf(bin_edges, mu, sigma)


# HYPOTHESIS TESTING 
def p_value(chi_square, x, ndof) :
  s = 1-chi2.cdf(chi_square, len(x)-ndof)
  r = s*100
  return r

# z_test double sided
def z_test1(x1,x2,s1,s2) : 
  z = np.absolute(x1-x2)/np.sqrt(s1**2+s2**2)  #t di confronto
  R = quad(Gaussian_standard,-t,t) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# z test di ipotesi con un valore calcolato
def z_test2(x1,X,s) :  
  z = np.absolute(x1-X)/s  #t di confronto
  R = quad(Gaussian_standard,-t,t) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# t test con 1 vincolo
def t_test1(x1, X, err_media) :  
	t = np.absolute(x1-X)/err_media
	R = t.cdf(-t, df=len(x1)-1)
	p_value = R*2
	return p_value
	


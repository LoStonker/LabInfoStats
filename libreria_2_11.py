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



"""Funzione per fare lo scatter"""

def scatter_plot_with_error(x, y, sigma_y, xlabel, ylabel, title, sigma_x=None, axhline_value=None):
    """
    Crea uno scatter plot dei dati con barre d'errore e opzioni di personalizzazione.
    
    Parametri:
      x: array-like, dati dell'asse x
      y: array-like, dati dell'asse y
      sigma_y: array-like, errori associati a y
      xlabel: string, etichetta per l'asse x
      ylabel: string, etichetta per l'asse y
      title: string, titolo del grafico
      sigma_x: array-like, errori associati a x (opzionale)
      axhline_value: float, valore y dove disegnare una linea orizzontale (default: None, nessuna linea) (opzionale)
    """
    
    # Creazione della figura
    plt.figure(figsize=(10, 5), dpi=100)
    plt.style.use('seaborn-v0_8-notebook')
    
    # Plot delle barre di errore con maggiore visibilità, includendo anche sigma_x se fornito
    plt.errorbar(
        x,
        y,
        xerr=sigma_x,
        yerr=sigma_y,
        fmt='o',                    # marker circolare per evidenziare i dati
        markersize=5,
        markerfacecolor='purple',
        markeredgecolor='black',
        ecolor='orange',               # colore rosso per le barre di errore
        elinewidth=1.5,               # spessore maggiore della linea degli errori
        capsize=4,                  # dimensione dei "tappi" delle barre d'errore
        alpha=0.8,
        zorder=1
    )
    
    # Scatter plot colorato: il colore è basato sul valore assoluto di y
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
    
    # Aggiunta della linea orizzontale se specificata
    if axhline_value is not None:
        plt.axhline(axhline_value, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    
    # Aggiunta della griglia
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)
    
    # Titolo ed etichette
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    
    # Personalizzazione degli assi
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
	
# t test con 2 vincoli
def t_test2(x1, x2, err1, err2) : 
	t = np.absolute(x1-x2)/np.sqrt(err1**2+err2**2)
	R = t.cdf(-t, df=len(x1)-1)
	p_value = R*2
	return p_value

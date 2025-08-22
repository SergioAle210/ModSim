import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import numpy as np


def load_data():
    """Returns synthetic discrete-time and continuous-time simulation outputs"""
    np.random.seed(42)  # For reproducibility
    
    # Discrete-time data (weekly reporting)
    discrete = {
        'timestamps': np.arange(0, 21),  # Daily timestamps (integer days)
        'infections': [10, 12, 8, 9, 11, 15, 120,  # Weekly spike
                       9, 11, 7, 8, 10, 125,       # Next week
                       8, 10, 6, 9, 12, 118],      # Final week
        'agent_data': pd.DataFrame({
            'age': np.random.choice(['0-18','19-65','65+'], 1000),
            'occupation': np.random.choice(['healthcare','education','other'], 1000),
            'vaccinated': np.random.choice([True, False], 1000, p=[0.6, 0.4])
        })
    }
    
    # Continuous-time data (event-driven)
    t_continuous = np.linspace(0, 21, 500)
    outbreaks = (80 * np.exp(-(t_continuous-3.5)**2/1.5) + 
                100 * np.exp(-(t_continuous-8.2)**2/2) + 
                90 * np.exp(-(t_continuous-14.1)**2/1.8))
    background = 10 * np.sin(0.3*t_continuous) + 15
    
    continuous = {
        'timestamps': t_continuous,
        'infections': outbreaks + background + np.random.normal(0, 3, 500),
        'agent_data': pd.DataFrame({
            'age': np.random.choice(['0-18','19-65','65+'], 1000),
            'mobility': np.random.gamma(2, 1.5, 1000),  # Continuous trait
            'vaccinated': np.random.choice([True, False], 1000, p=[0.6, 0.4])
        })
    }
    
    return {'discrete': discrete, 'continuous': continuous}


# Código para ejecutar cuando se llama el script directamente
if __name__ == "__main__":
    print("Cargando datos de simulación...")
    data = load_data()
    
    print("=== Datos Discretos ===")
    print(f"Timestamps: {len(data['discrete']['timestamps'])} puntos (días 0-20)")
    print(f"Infecciones: {len(data['discrete']['infections'])} valores")
    print(f"Datos de agentes: {len(data['discrete']['agent_data'])} agentes")
    print(f"Primeros 10 valores de infecciones: {data['discrete']['infections'][:10]}")
    
    print("\n=== Datos Continuos ===")
    print(f"Timestamps: {len(data['continuous']['timestamps'])} puntos")
    print(f"Infecciones: {len(data['continuous']['infections'])} valores")
    print(f"Datos de agentes: {len(data['continuous']['agent_data'])} agentes")
    print(f"Rango de timestamps: {data['continuous']['timestamps'][0]:.2f} - {data['continuous']['timestamps'][-1]:.2f}")
    
    print("\n=== Vista previa de datos de agentes (discreto) ===")
    print(data['discrete']['agent_data'].head())
    
    print("\n=== Distribución de edad ===")
    print(data['discrete']['agent_data']['age'].value_counts())
    
    print("\n=== Distribución de ocupación ===")
    print(data['discrete']['agent_data']['occupation'].value_counts())
    
    print("\n=== Estado de vacunación ===")
    print(data['discrete']['agent_data']['vaccinated'].value_counts())
    
    print("\n¡Datos cargados exitosamente!")
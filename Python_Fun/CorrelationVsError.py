#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:47:56 2024

@author: isaac
"""

import numpy as np
from scipy.stats import pearsonr

# Semilla para reproducibilidad
np.random.seed(42)

# Número de elementos
n = 156

# Rango de correlación deseado
correlation_range = np.arange(0, 1.01, 0.01)

# Listas para almacenar los resultados
X_values = []
Y_values = []
correlations = []

# Generar datos iterativamente
for target_correlation in correlation_range:
    # Generar datos X e Y con distribución uniforme
    X = np.random.uniform(18, 90, n)
    Y = np.random.uniform(18, 90, n)

    # Ajustar la correlación entre X e Y
    Y = target_correlation * X + np.sqrt(1 - target_correlation**2) * np.random.uniform(18, 90, n)

    # Almacenar los valores y la correlación
    X_values.append(X)
    Y_values.append(Y)
    correlations.append(pearsonr(X, Y)[0])

# Imprimir los resultados
for i in range(len(correlation_range)):
    print(f'Correlación deseada: {correlation_range[i]:.2f}, Correlación real: {correlations[i]:.4f}')

# Puedes acceder a los valores específicos de X e Y para cada iteración usando X_values[i] y Y_values[i].

import math
import numpy as np # Necessário para o sorteio de sucesso

"""
Módulo para simular conectividade e RSU em VFL.
"""

# --- Parâmetros Padrão do Modelo de Conectividade (Sigmoide PDR) ---
# Estes podem ser sobrescritos por argumentos de linha de comando em fedl.py
DEFAULT_PDR_PARAMS = {
    'd_mid': 300.0,                # Distância (metros) onde PDR é ~0.5
    'k_steepness': 0.015,          # Quão rápido o PDR cai (k>0 => PDR diminui com dist)
    'max_effective_range': 600.0   # Além desta distância, PDR é efetivamente 0
}
DEFAULT_MIN_PDR_FOR_CANDIDACY = 0.3 # PDR mínimo para um veículo ser considerado

def calculate_pdr(vehicle_pos, rsu_pos, pdr_params):
    """
    Calcula a Probabilidade de Entrega de Pacotes (PDR) entre duas posições.

    Args:
        vehicle_pos (tuple): Coordenadas (x, y) do veículo.
        rsu_pos (tuple): Coordenadas (x, y) da RSU.
        pdr_params (dict): Dicionário com parâmetros do modelo PDR:
                           'd_mid', 'k_steepness', 'max_effective_range'

    Returns:
        float: PDR (entre 0.0 e 1.0).
    """
    if vehicle_pos is None or rsu_pos is None:
        return 0.0

    # Calcula a distância Euclidiana 2D
    try:
        distance = math.dist(vehicle_pos, rsu_pos)
    except AttributeError: # Fallback para Python < 3.8
        distance = math.sqrt((vehicle_pos[0] - rsu_pos[0])**2 + (vehicle_pos[1] - rsu_pos[1])**2)

    d_mid = pdr_params['d_mid']
    k = pdr_params['k_steepness']
    max_range = pdr_params['max_effective_range']

    if distance > max_range:
        return 0.0

    # Modelo Sigmoide PDR
    try:
        # PDR = 1.0 / (1.0 + exp(k * (distance - d_mid)))
        pdr = 1.0 / (1.0 + math.exp(k * (distance - d_mid)))
    except OverflowError:
        # Lida com casos onde o argumento de exp() é muito grande/pequeno
        if k * (distance - d_mid) > 700: # Limite prático antes de exp() estourar
            pdr = 0.0
        else: # Argumento muito negativo -> exp() -> 0 -> PDR -> 1.0
            pdr = 1.0

    return max(0.0, min(1.0, pdr)) # Garante PDR entre 0 e 1

def simulate_transmission_success(pdr):
    """
    Simula se uma transmissão foi bem-sucedida baseado no PDR.

    Args:
        pdr (float): A probabilidade de entrega de pacotes (0.0 a 1.0).

    Returns:
        bool: True se a transmissão simulada foi bem-sucedida, False caso contrário.
    """
    return np.random.rand() < pdr

# Você pode adicionar uma classe RSU aqui se planeja ter múltiplas RSUs no futuro,
# mas por enquanto, passar a posição e os parâmetros PDR para as funções é suficiente.
# class RSU:
#     def __init__(self, rsu_id, position, pdr_params):
#         self.id = rsu_id
#         self.position = position
#         self.pdr_params = pdr_params
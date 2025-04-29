import traci
import numpy as np
import torch
import torch.utils.data # Para DataLoader
import math # <--- Importação necessária para cálculos de distância

class Vehicle:
    """Representa um veículo na simulação SUMO que atua como cliente FL."""

    def __init__(self, vehicle_id, cifar_subset):
        """
        Inicializa o veículo.

        Args:
            vehicle_id (str): O ID do veículo no SUMO.
            cifar_subset (torch.utils.data.Subset): O subconjunto de dados CIFAR-10 atribuído a este veículo.
        """
        self.id = vehicle_id
        self.cifar_data = cifar_subset # Armazena o Subset do CIFAR-10

        # Cria um DataLoader para este veículo UMA VEZ (mais eficiente)
        if self.cifar_data and len(self.cifar_data) > 0:
             # Ajuste BATCH_SIZE conforme necessário para treino local
             local_batch_size = 16 if len(self.cifar_data) >= 16 else len(self.cifar_data)
             # drop_last=True pode evitar batches com tamanho 1, que podem causar problemas em alguns casos (ex: BatchNorm)
             self.dataloader = torch.utils.data.DataLoader(self.cifar_data, batch_size=local_batch_size, shuffle=True, drop_last=(len(self.cifar_data) > local_batch_size))
        else:
             self.dataloader = None # Sem dados, sem dataloader
             print(f"Aviso: Veículo {self.id} criado sem dados CIFAR ou com dataset vazio.")

    def get_position(self):
        """Retorna a posição (x, y) atual do veículo ou None se não encontrado/erro."""
        try:
            # Verifica se o veículo ainda existe na simulação antes de pegar a posição
            # getIDList é rápido, getPosition pode lançar exceção se o veículo sumiu
            if self.id in traci.vehicle.getIDList():
                return traci.vehicle.getPosition(self.id)
            else:
                # Veículo não está mais na simulação
                return None
        except traci.TraCIException as e:
            # Log específico para erro TraCI, pode acontecer se a conexão cair
            # print(f"Debug: TraCIException em get_position para {self.id}: {e}")
            return None
        except Exception as e:
            # Captura outros erros inesperados
            print(f"Erro não TraCI em get_position para {self.id}: {e}")
            return None

    def is_in_range(self, rsu_pos, max_range):
        """
        Verifica se o veículo está dentro do alcance especificado da RSU.

        Args:
            rsu_pos (tuple): Coordenadas (x, y) da RSU.
            max_range (float): O raio máximo de comunicação em metros.

        Returns:
            bool: True se o veículo está dentro do alcance, False caso contrário.
        """
        current_pos = self.get_position()
        if current_pos is None:
            # print(f"Debug: Posição não encontrada para veículo {self.id} em is_in_range.") # Log opcional
            return False # Veículo não está na simulação ou erro ao obter posição

        # Calcula a distância Euclidiana 2D
        try:
            # Use math.dist (Python 3.8+)
            distance = math.dist(current_pos, rsu_pos)
        except AttributeError:
            # Fallback para Python < 3.8
            distance = math.sqrt((current_pos[0] - rsu_pos[0])**2 + (current_pos[1] - rsu_pos[1])**2)

        # print(f"Debug: Veículo {self.id} - Posição: {current_pos}, Distância da RSU {rsu_pos}: {distance:.2f}m") # Log opcional
        return distance <= max_range

# Nota: A função create_vehicle_list foi integrada/adaptada no loop principal de fedl.py
# para lidar com a entrada dinâmica de veículos e atribuição de datasets.
# Se você precisar dela separada por algum motivo, ajuste conforme necessário.
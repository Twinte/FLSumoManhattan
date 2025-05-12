import traci
# import numpy as np # Não mais necessário aqui
import torch
import torch.utils.data
# import math # Não mais necessário aqui

class Vehicle:
    """Representa um veículo na simulação SUMO que atua como cliente FL."""

    def __init__(self, vehicle_id, cifar_subset):
        """
        Inicializa o veículo.

        Args:
            vehicle_id (str): O ID do veículo no SUMO.
            cifar_subset (torch.utils.data.Subset): O subconjunto de dados CIFAR-10 atribuído.
        """
        self.id = vehicle_id
        self.cifar_data = cifar_subset # Armazena o Subset do CIFAR-10

        # Cria um DataLoader para este veículo (se tiver dados)
        if self.cifar_data and len(self.cifar_data) > 0:
             local_batch_size = 16 if len(self.cifar_data) >= 16 else len(self.cifar_data)
             self.dataloader = torch.utils.data.DataLoader(
                 self.cifar_data,
                 batch_size=local_batch_size,
                 shuffle=True,
                 drop_last=(len(self.cifar_data) > local_batch_size)
             )
        else:
             self.dataloader = None

    def get_position(self):
        """Retorna a posição (x, y) atual do veículo ou None se não encontrado/erro."""
        # Esta função é crucial pois será chamada pela lógica em connectivity.py
        try:
            # Verifica se o veículo ainda existe na simulação
            if self.id in traci.vehicle.getIDList():
                return traci.vehicle.getPosition(self.id)
            else:
                return None
        except traci.TraCIException:
            return None
        except Exception as e:
            print(f"Erro não TraCI em get_position para {self.id}: {e}")
            return None

    # O método calculate_pdr_to_rsu FOI REMOVIDO daqui e sua lógica movida para connectivity.py

    # Mantenha outras funções se precisar (get_state, etc.)
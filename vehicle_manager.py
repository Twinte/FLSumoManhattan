import traci
import numpy as np
import torch
import torch.utils.data # Para DataLoader
import math # Importação para cálculos de distância

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
             # Batch size local pode ser diferente do global
             local_batch_size = 16 if len(self.cifar_data) >= 16 else len(self.cifar_data)
             # drop_last=True evita batches de tamanho 1 que podem dar erro em BatchNorm
             self.dataloader = torch.utils.data.DataLoader(
                 self.cifar_data,
                 batch_size=local_batch_size,
                 shuffle=True,
                 drop_last=(len(self.cifar_data) > local_batch_size) # Só descarta se tiver mais q 1 batch
             )
        else:
             self.dataloader = None # Sem dados, sem dataloader
             # print(f"Aviso: Veículo {self.id} criado sem dados CIFAR ou com dataset vazio.") # Log Opcional

    def get_position(self):
        """Retorna a posição (x, y) atual do veículo ou None se não encontrado/erro."""
        try:
            # Verifica se o veículo ainda existe na simulação
            # Usar getIDList pode ser menos eficiente em loops grandes,
            # mas é mais seguro que tentar pegar a posição diretamente.
            if self.id in traci.vehicle.getIDList():
                return traci.vehicle.getPosition(self.id)
            else:
                return None
        except traci.TraCIException:
            # Erro comum se a conexão com SUMO cair ou o veículo sumir entre comandos
            # print(f"Debug: TraCIException em get_position para {self.id}") # Log opcional
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

    # Se precisar das funções get_state, get_labels, update_buffer para
    # coletar dados do SUMO (não usadas no treino CIFAR atual), mantenha-as aqui.
    # Exemplo:
    # def get_state(self):
    #     """Retorna o estado normalizado do veículo (ex: pos, vel)."""
    #     pos = self.get_position()
    #     if pos is None: return None
    #     try:
    #         speed = traci.vehicle.getSpeed(self.id)
    #         max_speed = traci.vehicle.getAllowedSpeed(self.id)
    #         speed_ratio = min(speed / max_speed, 1.0) if max_speed > 0 else 0.0
    #         # Exemplo de normalização - ajuste conforme necessário
    #         return [pos[0]/1000.0, pos[1]/1000.0, speed_ratio]
    #     except traci.TraCIException:
    #         return None
    #     except Exception as e:
    #          print(f"Erro em get_state para {self.id}: {e}")
    #          return None

# A função create_vehicle_list não é mais necessária aqui,
# pois a criação e atribuição de dados são feitas dinamicamente em fedl.py.
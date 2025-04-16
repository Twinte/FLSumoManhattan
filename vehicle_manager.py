import traci
import numpy as np
import torch
import torch.utils.data # Para DataLoader

class Vehicle:
    def __init__(self, vehicle_id, cifar_subset): # Recebe um Subset
        self.id = vehicle_id
        # self.data_buffer = []  # Removido se não for usar estado do SUMO para treino
        # self.label_buffer = [] # Removido se não for usar estado do SUMO para treino
        # self._prev_speed = 0   # Removido se não for usar estado do SUMO para treino
        self.cifar_data = cifar_subset # Armazena o Subset do CIFAR-10
        
        # Cria um DataLoader para este veículo UMA VEZ (mais eficiente)
        if len(self.cifar_data) > 0:
             # Ajuste BATCH_SIZE conforme necessário para treino local
             local_batch_size = 16 if len(self.cifar_data) >= 16 else len(self.cifar_data) 
             self.dataloader = torch.utils.data.DataLoader(self.cifar_data, batch_size=local_batch_size, shuffle=True)
        else:
             self.dataloader = None # Sem dados, sem dataloader
             print(f"Aviso: Veículo {self.id} criado sem dados CIFAR.")

    # --- Funções get_state, get_labels, update_buffer ---
    # Mantenha estas funções se você precisar do estado do SUMO para *outros* fins
    # Se não precisar, pode removê-las para simplificar.
    def get_state(self):
        # ... (seu código como antes, com try/except) ...
        try:
            if self.id not in traci.vehicle.getIDList():
                # print(f"Debug: Veículo {self.id} não encontrado em get_state") # Debug
                return None
                
            pos = traci.vehicle.getPosition(self.id)
            speed = traci.vehicle.getSpeed(self.id)
            # Cuidado com max_speed sendo 0 se o veículo estiver parado no início
            max_speed = traci.vehicle.getAllowedSpeed(self.id) 
            speed_ratio = min(speed / max_speed, 1.0) if max_speed > 0 else 0.0
            
            # Normalize features
            return [
                pos[0]/1000,        # X position
                pos[1]/1000,        # Y position
                speed_ratio
            ]
        except traci.TraCIException as e:
            # print(f"Debug: TraCIException em get_state para {self.id}: {e}") # Debug
            return None
        except Exception as e: # Captura outros erros
            # print(f"Debug: Exception em get_state para {self.id}: {e}") # Debug
            return None
            
    def get_labels(self):
        # ... (seu código como antes, com try/except) ...
        try:
            if self.id not in traci.vehicle.getIDList():
                 return None

            speed = traci.vehicle.getSpeed(self.id)
            max_speed = traci.vehicle.getAllowedSpeed(self.id)
            speed_ratio = min(speed/max_speed, 1.0) if max_speed > 0 else 0.0
            
            if speed_ratio < 0.1: return 2    # Stopped
            elif speed_ratio < 0.4: return 1  # Slow
            else: return 0                   # Normal
        except traci.TraCIException:
             return None
        except Exception:
             return None
    
    # def update_buffer(self): # Remova ou mantenha se necessário
    #     state = self.get_state()
    #     label = self.get_labels()
    #     # ... (resto da lógica do buffer) ...

    # get_cifar_batch foi substituído pelo DataLoader pré-calculado
    # def get_cifar_batch(self, batch_size):
    #     # ... (código antigo removido) ...


# --- Função create_vehicle_list (MODIFICADA) ---
# Nota: Esta função pode não ser mais necessária se a criação for feita dinamicamente
# no loop principal de fedl.py como sugerido. Se for usada, deve receber os subsets.
def create_vehicle_list(all_client_datasets):
     """Cria uma lista inicial de veículos a partir do SUMO."""
     vehicles = []
     try:
         vehicle_ids = traci.vehicle.getIDList()
         num_vehicles_in_sumo = len(vehicle_ids)
         num_datasets_available = len(all_client_datasets)
        
         num_to_create = min(num_vehicles_in_sumo, num_datasets_available)
        
         print(f"Criando lista inicial de {num_to_create} veículos...")
         for i in range(num_to_create):
             vehicle_id = vehicle_ids[i]
             vehicle_data_subset = all_client_datasets[i]
             vehicles.append(Vehicle(vehicle_id, vehicle_data_subset))
    
     except traci.TraCIException as e:
          print(f"Erro TraCI ao criar lista inicial de veículos: {e}")
         
     return vehicles
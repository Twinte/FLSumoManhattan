import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from fed_model import FedModel
from vehicle_manager import Vehicle, create_vehicle_list # Importe Vehicle também se necessário
import traci
import sys # Para sys.exit e checagem de SUMO_HOME
import os  # Para checagem de SUMO_HOME

# Adicione SUMO_HOME/tools ao PYTHONPATH - ESSENCIAL
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Erro: Declare a variável de ambiente 'SUMO_HOME'")


NUM_ROUNDS = 10
# NUM_CLIENTS = 20 # Torna-se dinâmico ou um máximo
CLIENTS_PER_ROUND = 100 # Exemplo: Selecionar um subconjunto a cada rodada
BATCH_SIZE = 32
SIMULATION_STEPS_PER_ROUND = 100 # Exemplo: Fazer uma rodada FL a cada 100 passos de simulação

# --- Datasets (como antes) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False) # Loader para validação

# --- FLMetrics (como antes) ---
class FLMetrics:
    # ... (seu código FLMetrics aqui) ...
    def __init__(self):
        self.round_losses = []
        self.round_accuracies = []
    
    def add_round_metrics(self, loss, accuracy):
        self.round_losses.append(loss)
        self.round_accuracies.append(accuracy)
    
    def plot_metrics(self):
        """Plots the training loss and accuracy over rounds."""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.round_losses, marker='o', color='tab:red')
        plt.title('Training Loss per Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.round_accuracies, marker='o', color='tab:green')
        plt.title('Validation Accuracy per Round')
        plt.xlabel('Round')
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()


# --- Função de Validação ---
def validate_model(model, dataloader):
    """Valida o modelo no dataset de teste."""
    model.eval()  # Coloca o modelo em modo de avaliação
    correct = 0
    total = 0
    with torch.no_grad(): # Desabilita cálculo de gradientes
        for data, labels in dataloader:
            # Mova dados para a GPU se estiver usando
            # data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # model.train() # Opcional: volta ao modo de treino se necessário logo após
    return accuracy

# --- Função para dividir dados (CORRIGIDA) ---
def create_client_datasets(num_clients, dataset):
    """Cria subconjuntos de dados para cada cliente."""
    if num_clients == 0:
        return []
    # Garante que não tentamos dividir mais do que temos dados
    num_samples = len(dataset)
    if num_clients > num_samples:
        print(f"Aviso: Número de clientes ({num_clients}) maior que o número de amostras ({num_samples}). Reduzindo.")
        num_clients = num_samples
        
    data_per_client = num_samples // num_clients
    client_data_indices = {} # Dicionário para mapear ID do cliente para índices
    indices = list(range(num_samples))
    np.random.shuffle(indices) # Embaralha os índices para distribuição aleatória
    
    all_client_subsets = []
    for i in range(num_clients):
        start_idx = i * data_per_client
        # Garante que o último cliente pegue o restante, se a divisão não for perfeita
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else num_samples 
        client_indices = indices[start_idx:end_idx]
        if len(client_indices) > 0: # Só cria subset se houver índices
             all_client_subsets.append(data.Subset(dataset, client_indices))
        
    # Retorna a lista de Subsets criados
    return all_client_subsets

# --- Função da Rodada FL (MODIFICADA para receber clientes selecionados) ---
def federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader):
    """Executa uma rodada de FL com os veículos selecionados."""
    client_weights = []
    round_losses = []

    print(f"  Iniciando treino local para {len(selected_vehicles)} veículos...")
    model.train() # Garante que o modelo está em modo de treino
    
    for vehicle in selected_vehicles:
        # vehicle.update_buffer() # Removido - buffer não usado para CIFAR
        
        # Tenta treinar localmente
        try:
            # Passa o modelo atual para treino local (ou instancia um novo)
            weights, loss = model.train_local(vehicle, epochs=1) 
            
            if loss is not None and loss > 0: # Verifica se o treino ocorreu e a loss é válida
                client_weights.append(weights)
                round_losses.append(loss)
                print(f"    Veículo {vehicle.id} treinou. Loss: {loss:.4f}")
            elif loss == 0.0:
                 print(f"    Veículo {vehicle.id} treinou, mas loss foi 0.0 (verificar dados/treino).")
            # Se loss for None, train_local já deve ter indicado o problema (ex: sem dados)

        except Exception as e:
            print(f"    Erro ao treinar veículo {vehicle.id}: {e}")

    # Agrega pesos se houver atualizações válidas
    if client_weights:
        print(f"  Agregando pesos de {len(client_weights)} clientes.")
        # FedAvg simples
        avg_weights = [
            np.mean(np.stack(layer_weights), axis=0) # Usa np.mean para média
            for layer_weights in zip(*client_weights)
        ]
        model.set_weights(avg_weights)
        
        # Valida o modelo global APÓS a agregação
        accuracy = validate_model(model, test_loader) 
        avg_loss = np.mean(round_losses) # Calcula a média das losses dos que treinaram
        metrics.add_round_metrics(avg_loss, accuracy)
        print(f"  Rodada {round_num+1} Métricas Globais - Loss Média (Clientes): {avg_loss:.4f}, Acurácia (Teste): {accuracy*100:.2f}%")
    
    else:
        print("  Nenhum cliente contribuiu com pesos nesta rodada. Modelo global não atualizado.")
        # Opcional: Adicionar métricas 'vazias' ou repetir as anteriores
        # accuracy = validate_model(model, test_loader) # Validar mesmo assim?
        # metrics.add_round_metrics(np.nan, accuracy) # Ou algum valor padrão

# --- Loop Principal (REESTRUTURADO) ---
def run_simulation_and_training():
    """Loop principal que integra simulação SUMO e treino FL."""
    model = FedModel(input_dim=3, output_dim=10) 
    metrics = FLMetrics()
    
    sumo_binary = "sumo-gui" 
    sumo_config = "manhattan.sumocfg" # Verifique o caminho correto

    active_vehicles_map = {} # Dicionário para mapear vehicle_id -> Objeto Vehicle

    try:
        print("Iniciando SUMO e TraCI...")
        traci.start([sumo_binary, "-c", sumo_config])
        print("TraCI iniciado.")

        step = 0
        round_num = 0
        
        # Divide os dados UMA VEZ no início (ou redivida periodicamente se necessário)
        # Precisamos saber quantos clientes *potenciais* teremos no máximo
        # Ou podemos atribuir dinamicamente, o que é mais complexo
        
        # Vamos supor que queremos treinar com até NUM_CLIENTS_MAX veículos
        NUM_CLIENTS_MAX = 50 # Exemplo: Número máximo de clientes com dados pré-alocados
        all_client_datasets = create_client_datasets(NUM_CLIENTS_MAX, train_dataset)
        dataset_assignment_index = 0 # Para pegar o próximo dataset disponível

        print(f"Simulação iniciada. Rodada FL a cada {SIMULATION_STEPS_PER_ROUND} passos.")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # --- Gerenciamento Dinâmico de Veículos ---
            current_vehicle_ids = set(traci.vehicle.getIDList())
            existing_vehicle_ids = set(active_vehicles_map.keys())

            # Veículos que saíram
            departed_ids = existing_vehicle_ids - current_vehicle_ids
            for v_id in departed_ids:
                print(f"  Veículo {v_id} saiu da simulação.")
                del active_vehicles_map[v_id]

            # Veículos que entraram
            arrived_ids = current_vehicle_ids - existing_vehicle_ids
            for v_id in arrived_ids:
                 if dataset_assignment_index < len(all_client_datasets):
                     print(f"  Veículo {v_id} entrou. Atribuindo dataset #{dataset_assignment_index}.")
                     # Cria um novo objeto Vehicle com seu subconjunto de dados
                     vehicle_dataset = all_client_datasets[dataset_assignment_index]
                     active_vehicles_map[v_id] = Vehicle(v_id, vehicle_dataset)
                     dataset_assignment_index += 1
                 else:
                     print(f"  Veículo {v_id} entrou, mas não há mais datasets pré-alocados.")


            # --- Lógica para iniciar Rodada FL ---
            if step > 0 and step % SIMULATION_STEPS_PER_ROUND == 0:
                print(f"\n--- Passo {step}: Iniciando Rodada FL {round_num + 1}/{NUM_ROUNDS} ---")
                
                active_vehicle_list = list(active_vehicles_map.values())
                
                if len(active_vehicle_list) > 0:
                    # Seleciona um subconjunto de clientes ativos para a rodada
                    num_to_select = min(CLIENTS_PER_ROUND, len(active_vehicle_list))
                    selected_vehicles = np.random.choice(active_vehicle_list, num_to_select, replace=False)
                    
                    # Executa a rodada FL
                    federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader)
                    round_num += 1 # Incrementa o número da rodada apenas se ela ocorreu
                else:
                     print(f"--- Passo {step}: Pular rodada FL (sem veículos ativos).")

                # Para o loop se atingiu o número de rodadas desejado
                if round_num >= NUM_ROUNDS:
                    print("\nNúmero alvo de rodadas FL atingido.")
                    break 
            
            # --- Outras lógicas por passo (opcional) ---
            # Ex: Coletar dados de SUMO para outros propósitos
            # for vehicle in active_vehicles_map.values():
            #     vehicle.update_buffer() # Chame se precisar do buffer de estado

            step += 1

        print("\nSimulação SUMO terminada.")

    except traci.TraCIException as e:
        print(f"Erro fatal no TraCI: {e}")
    except Exception as e:
        print(f"Erro inesperado no script: {e}")
    finally:
        print("Fechando conexão TraCI...")
        try:
            traci.close()
            print("Conexão TraCI fechada.")
        except Exception as e:
            print(f"Erro ao fechar TraCI: {e}")

    # Mostra resultados
    print("\nPlotando métricas de treino...")
    metrics.plot_metrics()

if __name__ == "__main__":
    run_simulation_and_training()
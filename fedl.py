import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from fed_model import FedModel
from vehicle_manager import Vehicle # Importa a classe Vehicle
import traci
import sys
import os
import math # <--- Importação necessária para cálculos de distância
import traceback # Para imprimir detalhes de erros

# Adicione SUMO_HOME/tools ao PYTHONPATH - ESSENCIAL
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Erro: Declare a variável de ambiente 'SUMO_HOME'")


# --- Constantes e Configurações ---
NUM_ROUNDS = 10                 # Número total de rodadas FL
CLIENTS_PER_ROUND = 10          # Número de clientes a selecionar por rodada (se disponíveis)
BATCH_SIZE = 32                 # Batch size global (usado para test_loader)
SIMULATION_STEPS_PER_ROUND = 50 # Executar uma rodada FL a cada X passos de simulação
RSU_POSITION = (450.0, 450.0)   # <--- Posição da RSU (x, y) - CENTRO DO MAPA
MAX_COMMUNICATION_RANGE = 1000.0 # <--- Alcance de 1 KM (1000 metros)
NUM_CLIENTS_MAX = 50            # Número máximo de clientes para pré-dividir dados
SUMO_CONFIG_FILE = "manhattan.sumocfg" # <--- Verifique este caminho!
SUMO_BINARY = "sumo-gui"        # Use "sumo" para rodar sem interface gráfica

# --- Datasets e Transformações ---
print("Configurando datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalização padrão para CIFAR-10
])

try:
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # DataLoader para o conjunto de teste (usado na validação)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    print("Datasets CIFAR-10 carregados/baixados.")
except Exception as e:
     print(f"Erro ao carregar/baixar CIFAR-10: {e}")
     sys.exit("Falha no carregamento do dataset.")

# --- Classe para Métricas FL ---
class FLMetrics:
    def __init__(self):
        self.round_losses = []
        self.round_accuracies = []

    def add_round_metrics(self, loss, accuracy):
        self.round_losses.append(loss)
        self.round_accuracies.append(accuracy)
        # Salva as métricas a cada rodada (opcional)
        # self.save_metrics()

    def plot_metrics(self):
        """Plota loss e acurácia ao longo das rodadas."""
        if not self.round_losses and not self.round_accuracies:
            print("Nenhuma métrica foi registrada para plotar.")
            return

        plt.figure(figsize=(12, 5))

        if self.round_losses:
            plt.subplot(1, 2, 1)
            # Remove NaNs para plotagem, se houver rodadas sem treino
            losses_to_plot = [l for l in self.round_losses if not np.isnan(l)]
            rounds_for_loss = [i + 1 for i, l in enumerate(self.round_losses) if not np.isnan(l)]
            if losses_to_plot:
                 plt.plot(rounds_for_loss, losses_to_plot, marker='o', color='tab:red')
            plt.title('Loss Média (Clientes Treinados) por Rodada FL')
            plt.xlabel('Rodada FL')
            plt.ylabel('Loss')
            plt.grid(True)

        if self.round_accuracies:
            plt.subplot(1, 2, 2)
            # Remove NaNs para plotagem
            acc_to_plot = [a for a in self.round_accuracies if not np.isnan(a)]
            rounds_for_acc = [i + 1 for i, a in enumerate(self.round_accuracies) if not np.isnan(a)]
            if acc_to_plot:
                plt.plot(rounds_for_acc, acc_to_plot, marker='o', color='tab:green')
            plt.title('Acurácia Global (Teste) por Rodada FL')
            plt.xlabel('Rodada FL')
            plt.ylim(0, 1.0)
            plt.ylabel('Acurácia')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig("fl_metrics_plot.png") # Salva o gráfico
        print("Gráfico de métricas salvo como fl_metrics_plot.png")
        plt.show()

    # def save_metrics(self, filename="fl_metrics.csv"): # Opcional: Salvar em CSV
    #     import pandas as pd
    #     df = pd.DataFrame({
    #         'Round': range(1, len(self.round_losses) + 1),
    #         'Loss': self.round_losses,
    #         'Accuracy': self.round_accuracies
    #     })
    #     df.to_csv(filename, index=False)
    #     print(f"Métricas salvas em {filename}")


# --- Função de Validação Global ---
def validate_model(model, dataloader):
    """Valida o modelo global no dataset de teste."""
    model.eval()  # Modo de avaliação
    correct = 0
    total = 0
    device = next(model.parameters()).device # Obtem o device do modelo (cpu/cuda)
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device) # Move para o device
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# --- Função para Dividir Dados entre Clientes ---
def create_client_datasets(num_clients, dataset):
    """Cria subconjuntos (Subset) de dados para cada cliente potencial."""
    if num_clients == 0:
        print("Aviso: num_clients é 0 em create_client_datasets.")
        return []
    num_samples = len(dataset)
    if num_clients > num_samples:
        print(f"Aviso: Número de clientes ({num_clients}) > amostras ({num_samples}). Usando {num_samples} clientes.")
        num_clients = num_samples

    if num_clients == 0: # Checa novamente se num_samples era 0
        print("Erro: Não há amostras no dataset para dividir.")
        return []

    data_per_client = num_samples // num_clients
    indices = list(range(num_samples))
    np.random.shuffle(indices) # Embaralha para distribuição IID (ou pseudo-não-IID)

    all_client_subsets = []
    print(f"Distribuindo {data_per_client} amostras por cliente (aprox)...")
    for i in range(num_clients):
        start_idx = i * data_per_client
        # O último cliente pega o restante para evitar perda de dados
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        if len(client_indices) > 0:
             all_client_subsets.append(data.Subset(dataset, client_indices))
        else:
             print(f"Aviso: Cliente potencial {i} ficaria sem dados.")

    print(f"{len(all_client_subsets)} subconjuntos de dados criados.")
    return all_client_subsets

# --- Função da Rodada FL ---
def federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader):
    """Executa uma rodada de FL com os veículos selecionados."""
    client_weights = []
    client_losses = [] # Guarda a loss de cada cliente que treinou
    num_samples_list = [] # Opcional: para FedAvg ponderado

    print(f"  Iniciando treino local para {len(selected_vehicles)} veículos selecionados...")
    model.train() # Garante modo de treino

    for vehicle in selected_vehicles:
        # O modelo já deve ter os pesos globais mais recentes (set_weights foi chamado antes)
        # Passamos o veículo para a função de treino local
        try:
            # Importante: train_local retorna pesos atualizados e loss média
            weights, loss = model.train_local(vehicle, epochs=1)

            if loss is not None and not np.isnan(loss): # Verifica se treino ocorreu e loss é válida
                if loss >= 0: # Verifica se a loss não é negativa (pode indicar problema)
                    client_weights.append(weights)
                    client_losses.append(loss)
                    # Opcional: Obter número de amostras para FedAvg ponderado
                    # num_samples_list.append(len(vehicle.cifar_data))
                    # print(f"    Veículo {vehicle.id} treinou. Loss: {loss:.4f}") # Log verboso
                else:
                    print(f"    Aviso: Veículo {vehicle.id} retornou loss negativa ({loss:.4f}). Descartando.")
            # else: # Log verboso opcional
                 # print(f"    Veículo {vehicle.id} não retornou loss válida (None ou NaN).")

        except Exception as e:
            print(f"    Erro CRÍTICO ao treinar veículo {vehicle.id}: {e}")
            traceback.print_exc() # Imprime stack trace do erro

    # Agrega pesos se houver atualizações válidas
    if client_weights:
        num_contributing_clients = len(client_weights)
        print(f"  Agregando pesos de {num_contributing_clients} clientes.")

        # --- Agregação FedAvg Simples ---
        try:
            avg_weights = [
                np.mean(np.stack(layer_weights, axis=0), axis=0)
                for layer_weights in zip(*client_weights) # zip(*...) agrupa pesos da mesma camada
            ]
            model.set_weights(avg_weights)
            print("  Modelo global atualizado.")

            # Calcula métricas APÓS a agregação
            avg_contributing_loss = np.mean(client_losses) if client_losses else np.nan
            global_accuracy = validate_model(model, test_loader)
            metrics.add_round_metrics(avg_contributing_loss, global_accuracy) # Adiciona métricas válidas
            print(f"  Rodada {round_num+1} Métricas - Loss Média (Clientes): {avg_contributing_loss:.4f}, Acurácia (Teste): {global_accuracy*100:.2f}%")

        except Exception as e_agg:
            print(f"  Erro durante a agregação ou validação: {e_agg}")
            traceback.print_exc()
            # Adiciona NaN para indicar falha na rodada
            metrics.add_round_metrics(np.nan, np.nan)

    else:
        print("  Nenhum cliente contribuiu com pesos válidos nesta rodada. Modelo global não atualizado.")
        # Adiciona NaN ou repete métrica anterior para manter consistência nos plots
        last_acc = metrics.round_accuracies[-1] if metrics.round_accuracies else np.nan
        metrics.add_round_metrics(np.nan, last_acc) # Ou (np.nan, np.nan)

# --- Loop Principal ---
def run_simulation_and_training():
    """Loop principal que integra simulação SUMO e treino FL com conectividade."""
    print("Inicializando modelo FL...")
    # Detecta se CUDA está disponível e usa, senão usa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    model = FedModel(input_dim=3, output_dim=10).to(device) # Move o modelo para o device
    metrics = FLMetrics()

    sumoCmd = [SUMO_BINARY, "-c", SUMO_CONFIG_FILE, "--start", "--quit-on-end"]

    active_vehicles_map = {} # Dicionário: vehicle_id -> Objeto Vehicle

    try:
        print("Iniciando SUMO e TraCI...")
        traci.start(sumoCmd)
        print("TraCI iniciado.")

        step = 0
        round_num = 0

        print(f"Dividindo dataset CIFAR-10 para até {NUM_CLIENTS_MAX} clientes potenciais...")
        all_client_datasets = create_client_datasets(NUM_CLIENTS_MAX, train_dataset)
        dataset_assignment_index = 0

        if not all_client_datasets:
             print("Erro: Falha ao criar subconjuntos de dados. Encerrando.")
             return # Sai se não conseguiu criar datasets

        print(f"Simulação iniciada. RSU em {RSU_POSITION}, Alcance: {MAX_COMMUNICATION_RANGE}m.")
        print(f"Rodada FL a cada {SIMULATION_STEPS_PER_ROUND} passos. Total de {NUM_ROUNDS} rodadas.")

        # Loop principal da simulação
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # --- Gerenciamento Dinâmico de Veículos ---
            current_vehicle_ids = set(traci.vehicle.getIDList())
            existing_vehicle_ids = set(active_vehicles_map.keys())

            # Remove veículos que saíram
            departed_ids = existing_vehicle_ids - current_vehicle_ids
            for v_id in departed_ids:
                # print(f"  Veículo {v_id} saiu.") # Log opcional
                if v_id in active_vehicles_map:
                    del active_vehicles_map[v_id]

            # Adiciona veículos que entraram e atribui dataset
            arrived_ids = current_vehicle_ids - existing_vehicle_ids
            for v_id in arrived_ids:
                 if dataset_assignment_index < len(all_client_datasets):
                     # print(f"  Veículo {v_id} entrou. Atribuindo dataset #{dataset_assignment_index}.") # Log opcional
                     vehicle_dataset = all_client_datasets[dataset_assignment_index]
                     active_vehicles_map[v_id] = Vehicle(v_id, vehicle_dataset)
                     dataset_assignment_index += 1
                 # else: # Log opcional
                     # print(f"  Aviso: Veículo {v_id} entrou, mas não há mais datasets pré-alocados.")


            # --- Lógica para iniciar Rodada FL ---
            if step > 0 and step % SIMULATION_STEPS_PER_ROUND == 0 and round_num < NUM_ROUNDS:
                print(f"\n--- Passo {step}: Verificando para Rodada FL {round_num + 1}/{NUM_ROUNDS} ---")

                # 1. Obter TODOS os veículos ATIVOS no mapa e gerenciados pelo script
                current_active_ids = set(active_vehicles_map.keys())
                vehicles_to_check = [active_vehicles_map[vid] for vid in current_active_ids]

                # 2. Filtrar veículos que estão DENTRO DO ALCANCE da RSU
                vehicles_in_range = []
                if not vehicles_to_check:
                     print("  Nenhum veículo ativo gerenciado no momento.")
                else:
                     print(f"  Verificando conectividade para {len(vehicles_to_check)} veículos ativos...")
                     for vehicle in vehicles_to_check:
                         if vehicle.is_in_range(RSU_POSITION, MAX_COMMUNICATION_RANGE):
                             vehicles_in_range.append(vehicle)
                         # else: # Log Opcional
                         #     pass # print(f"    Veículo {vehicle.id} fora de alcance.")

                     print(f"  {len(vehicles_in_range)} veículos estão DENTRO do alcance ({MAX_COMMUNICATION_RANGE}m) da RSU em {RSU_POSITION}.")

                # 3. Prosseguir com a rodada FL APENAS se houver veículos conectados
                if vehicles_in_range:
                    # 4. Seleciona um subconjunto de clientes DENTRE OS VEÍCULOS EM ALCANCE
                    num_available_clients = len(vehicles_in_range)
                    num_to_select = min(CLIENTS_PER_ROUND, num_available_clients)

                    if num_to_select > 0:
                         print(f"  Selecionando {num_to_select} clientes dentre os {num_available_clients} em alcance.")
                         selected_vehicles = np.random.choice(vehicles_in_range, num_to_select, replace=False)

                         # 5. Executa a rodada FL
                         print(f"--- Iniciando Rodada FL {round_num + 1}/{NUM_ROUNDS} ---")
                         federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader)
                         round_num += 1 # Incrementa contador de rodadas bem-sucedidas
                    else:
                         print("  Não há clientes suficientes em alcance para selecionar.")

                else:
                     print(f"--- Passo {step}: Pular rodada FL (sem veículos em alcance).")
                     # Adiciona métricas vazias se nenhuma rodada ocorreu para manter o tamanho dos arrays
                     metrics.add_round_metrics(np.nan, np.nan)


                # Verifica se completou todas as rodadas FL planejadas
                if round_num >= NUM_ROUNDS:
                    print(f"\nNúmero alvo de {NUM_ROUNDS} rodadas FL atingido.")
                    break # Sai do loop while da simulação

            step += 1
            # Adiciona uma pequena pausa para evitar uso excessivo de CPU (opcional)
            # time.sleep(0.01)

        print("\nSimulação SUMO terminada ou número de rodadas FL atingido.")

    except traci.TraCIException as e:
        print(f"Erro fatal no TraCI: {e}")
        traceback.print_exc()
    except KeyboardInterrupt: # Permite parar com Ctrl+C
        print("\nSimulação interrompida pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"Erro inesperado no script: {e}")
        traceback.print_exc()
    finally:
        print("Fechando conexão TraCI...")
        try:
            traci.close()
            print("Conexão TraCI fechada.")
        except Exception as e_close:
            # Pode dar erro se a conexão já foi perdida ou nunca iniciou
            print(f"Erro ao fechar TraCI (pode ser normal se já houve erro antes): {e_close}")

    # Mostra e salva resultados
    print("\nPlotando e salvando métricas de treino...")
    metrics.plot_metrics()
    # metrics.save_metrics() # Descomente se quiser salvar em CSV


if __name__ == "__main__":
    # Verifica se o arquivo de configuração do SUMO existe
    if not os.path.exists(SUMO_CONFIG_FILE):
         print(f"Erro: Arquivo de configuração do SUMO não encontrado em '{SUMO_CONFIG_FILE}'")
         print("Verifique o caminho na variável SUMO_CONFIG_FILE no script.")
         sys.exit(1)

    run_simulation_and_training()
    print("\nExecução do script concluída.")
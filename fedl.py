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
import math # Para cálculos de distância
import argparse # Para argumentos de linha de comando
import traceback # Para imprimir detalhes de erros
import time # Opcional, para pausas

# Adicione SUMO_HOME/tools ao PYTHONPATH - ESSENCIAL
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print(f"Adicionado ao sys.path: {tools}")
else:
    sys.exit("Erro: Variável de ambiente 'SUMO_HOME' não declarada.")


# --- Constantes e Configurações (padrões, podem ser sobrescritas por args) ---
NUM_ROUNDS = 10
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 32                 # Usado apenas para test_loader agora
SIMULATION_STEPS_PER_ROUND = 50
RSU_POSITION = (450.0, 450.0)
MAX_COMMUNICATION_RANGE = 300.0 # Default realista, pode ser sobrescrito
NUM_CLIENTS_MAX = 50            # Max clientes para pré-dividir dados
SUMO_CONFIG_FILE = "manhattan.sumocfg" # Verifique este caminho!
SUMO_BINARY = "sumo-gui"        # Mude para "sumo" para rodar sem GUI

# --- Datasets e Transformações ---
print("Configurando datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

try:
    # Garante que o diretório ./data exista
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Diretório './data' criado.")

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2) # num_workers acelera carregamento
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

    def plot_metrics(self, filename="fl_metrics_plot.png"):
        """Plota loss e acurácia e salva em arquivo."""
        if not self.round_losses and not self.round_accuracies:
            print("Nenhuma métrica foi registrada para plotar.")
            return

        plt.figure(figsize=(12, 5))
        rounds_available = range(1, len(self.round_losses) + 1)

        # Plota Loss se houver dados válidos
        valid_losses = [(r, l) for r, l in zip(rounds_available, self.round_losses) if l is not None and not np.isnan(l)]
        if valid_losses:
            rounds_l, losses_l = zip(*valid_losses)
            plt.subplot(1, 2, 1)
            plt.plot(rounds_l, losses_l, marker='o', linestyle='-', color='tab:red', label='Loss Média (Clientes)')
            plt.title('Loss Média por Rodada FL')
            plt.xlabel('Rodada FL')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

        # Plota Acurácia se houver dados válidos
        valid_accuracies = [(r, a) for r, a in zip(rounds_available, self.round_accuracies) if a is not None and not np.isnan(a)]
        if valid_accuracies:
            rounds_a, accs_a = zip(*valid_accuracies)
            plt.subplot(1, 2, 2)
            plt.plot(rounds_a, accs_a, marker='^', linestyle='--', color='tab:green', label='Acurácia Global (Teste)')
            plt.title('Acurácia Global por Rodada FL')
            plt.xlabel('Rodada FL')
            plt.ylim(0, 1.0)
            plt.ylabel('Acurácia')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        print(f"Gráfico de métricas salvo como {filename}")
        try:
            plt.show()
        except Exception as e_plot:
            print(f"Não foi possível exibir o gráfico interativamente: {e_plot}")

    # def save_metrics(self, filename="fl_metrics.csv"): # Opcional: Salvar em CSV
    #     try:
    #         import pandas as pd
    #         df = pd.DataFrame({
    #             'Round': range(1, len(self.round_losses) + 1),
    #             'Avg_Client_Loss': self.round_losses,
    #             'Global_Accuracy': self.round_accuracies
    #         })
    #         df.to_csv(filename, index=False, float_format='%.5f')
    #         print(f"Métricas salvas em {filename}")
    #     except ImportError:
    #         print("Biblioteca pandas não encontrada. Não foi possível salvar métricas em CSV.")
    #     except Exception as e_csv:
    #         print(f"Erro ao salvar métricas em CSV: {e_csv}")


# --- Função de Validação Global ---
def validate_model(model, dataloader, device):
    """Valida o modelo global no dataset de teste."""
    model.eval()  # Modo de avaliação
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device) # Move para o device
            outputs = model(data_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# --- Funções de Distribuição de Dados ---
def create_iid_client_datasets(num_clients, dataset):
    """Cria subconjuntos IID de dados (Shuffle & Split)."""
    print(f"Criando {num_clients} datasets IID...")
    num_samples = len(dataset)
    if num_clients == 0 or num_samples == 0: return []

    if num_clients > num_samples:
        print(f"Aviso: Número de clientes ({num_clients}) > amostras ({num_samples}). Limitando a {num_samples}.")
        num_clients = num_samples

    indices = list(range(num_samples))
    np.random.shuffle(indices)

    data_per_client = num_samples // num_clients
    all_client_subsets = []
    print(f"  Distribuindo ~{data_per_client} amostras por cliente...")
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        if client_indices: # Verifica se a lista não está vazia
             all_client_subsets.append(data.Subset(dataset, client_indices))

    print(f"  {len(all_client_subsets)} subconjuntos IID criados.")
    return all_client_subsets

def create_dirichlet_client_datasets(num_clients, dataset, alpha):
    """Cria subconjuntos Não-IID usando a Distribuição de Dirichlet."""
    print(f"Criando {num_clients} datasets Não-IID com Dirichlet (alpha={alpha})...")
    if num_clients == 0: return []

    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        print("Erro: Dataset precisa do atributo 'targets' para Dirichlet por classe.")
        return create_iid_client_datasets(num_clients, dataset) # Fallback

    num_samples = len(labels)
    if num_samples == 0: return []
    num_classes = len(np.unique(labels))

    if num_clients > num_samples:
        print(f"Aviso: Número de clientes ({num_clients}) > amostras ({num_samples}). Limitando a {num_samples}.")
        num_clients = num_samples
    if num_clients == 0: return []

    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    min_samples_per_class = min(len(idx) for idx in idx_by_class)

    if min_samples_per_class < 1 :
        print("Erro: Uma ou mais classes não têm amostras no dataset.")
        return create_iid_client_datasets(num_clients, dataset) # Fallback

    # Garante que não tentamos pegar mais clientes do que samples na menor classe (para evitar erros na dirichlet/split)
    # Isso é uma simplificação; abordagens mais complexas poderiam lidar com isso de forma diferente.
    # Ou podemos permitir clientes com poucas classes / samples, a lógica de treino precisa ser robusta.
    # Vamos prosseguir, mas estar cientes que alguns clientes podem ter muito poucos dados com alpha baixo.

    client_indices = [[] for _ in range(num_clients)]
    print(f"  Distribuindo amostras de {num_classes} classes...")

    for k in range(num_classes):
        class_indices = idx_by_class[k]
        np.random.shuffle(class_indices)
        num_samples_in_class = len(class_indices)

        # Amostra proporções da Dirichlet para esta classe
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Calcula quantos samples desta classe cada cliente recebe
        target_samples_per_client = (proportions * num_samples_in_class).astype(int)

        # Ajusta para garantir que a soma seja igual ao total (corrige arredondamentos)
        diff = num_samples_in_class - target_samples_per_client.sum()
        # Adiciona/subtrai a diferença aleatoriamente (com reposição se diff > num_clients)
        adjust_indices = np.random.choice(num_clients, abs(diff), replace=(abs(diff) > num_clients))
        target_samples_per_client[adjust_indices] += np.sign(diff)
        # Garante que nenhuma contagem ficou negativa após o ajuste
        target_samples_per_client = np.maximum(0, target_samples_per_client)
        # Reajuste final (raro, mas para garantir a soma exata)
        final_diff = num_samples_in_class - target_samples_per_client.sum()
        if final_diff != 0:
            target_samples_per_client[np.random.choice(num_clients)] += final_diff

        # Distribui os índices da classe k
        current_idx = 0
        for client_id in range(num_clients):
            num_assigned = target_samples_per_client[client_id]
            assigned_indices = class_indices[current_idx : current_idx + num_assigned]
            client_indices[client_id].extend(assigned_indices)
            current_idx += num_assigned

        # Verifica se todos os índices da classe foram distribuídos
        if current_idx != num_samples_in_class:
            print(f"Aviso: Nem todos os samples da classe {k} foram distribuídos ({current_idx}/{num_samples_in_class}). Verifique a lógica de ajuste.")


    # Cria os Subsets
    all_client_subsets = []
    print("  Criando Subsets para cada cliente...")
    total_assigned_samples = 0
    for i in range(num_clients):
        np.random.shuffle(client_indices[i]) # Embaralha dentro do dataset do cliente
        subset = data.Subset(dataset, client_indices[i])
        all_client_subsets.append(subset)
        total_assigned_samples += len(subset)
        # Log opcional para ver o tamanho de cada dataset de cliente
        # print(f"  Cliente {i}: {len(subset)} amostras")

    print(f"  {len(all_client_subsets)} subconjuntos Não-IID (Dirichlet) criados.")
    if total_assigned_samples != num_samples:
         print(f"  Aviso: Número total de amostras distribuídas ({total_assigned_samples}) não bate com o original ({num_samples}).")

    return all_client_subsets


# --- Função da Rodada FL ---
def federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader, device):
    """Executa uma rodada de FL com os veículos selecionados."""
    client_weights = []
    client_losses = []
    num_contributing_clients = 0

    print(f"  Iniciando treino local para {len(selected_vehicles)} veículos selecionados...")
    model.train() # Garante modo de treino

    for vehicle in selected_vehicles:
        if vehicle.dataloader is None: # Pula veículos que não receberam dados
            # print(f"    Veículo {vehicle.id} pulado (sem dataloader).") # Log opcional
            continue

        # Clona o modelo global para treino local (opcional, mas boa prática se houver concorrência)
        # local_model = copy.deepcopy(model) # Requer import copy
        # weights, loss = local_model.train_local(vehicle, epochs=1)
        # Alternativa: treina diretamente no objeto 'model' (mais simples aqui)
        try:
            weights, loss = model.train_local(vehicle, epochs=1) # train_local usa o device do modelo

            if loss is not None and not np.isnan(loss) and loss >= 0:
                client_weights.append(weights)
                client_losses.append(loss)
                num_contributing_clients += 1
                # print(f"    Veículo {vehicle.id} treinou. Loss: {loss:.4f}") # Log verboso
            # else: # Log opcional
            #    print(f"    Veículo {vehicle.id} não retornou loss válida (Loss: {loss}).")

        except Exception as e:
            print(f"    Erro CRÍTICO ao treinar veículo {vehicle.id}: {e}")
            traceback.print_exc()

    # Agrega pesos se houver contribuições válidas
    if client_weights:
        print(f"  Agregando pesos de {num_contributing_clients} clientes.")
        try:
            # --- Agregação FedAvg Simples ---
            with torch.no_grad(): # Desativa gradientes durante agregação/set_weights
                avg_weights = [
                    np.mean(np.stack(layer_weights, axis=0), axis=0)
                    for layer_weights in zip(*client_weights)
                ]
                model.set_weights(avg_weights)
            print("  Modelo global atualizado.")

            # Calcula métricas APÓS a agregação
            avg_contributing_loss = np.mean(client_losses) if client_losses else np.nan
            global_accuracy = validate_model(model, test_loader, device) # Passa o device
            metrics.add_round_metrics(avg_contributing_loss, global_accuracy)
            print(f"  Rodada FL {round_num+1} Métricas - Loss Média (Clientes): {avg_contributing_loss:.4f}, Acurácia (Teste): {global_accuracy*100:.2f}%")

        except Exception as e_agg:
            print(f"  Erro durante a agregação ou validação pós-rodada: {e_agg}")
            traceback.print_exc()
            metrics.add_round_metrics(np.nan, np.nan) # Indica falha

    else:
        print("  Nenhum cliente contribuiu com pesos válidos nesta rodada. Modelo global não atualizado.")
        last_acc = metrics.round_accuracies[-1] if metrics.round_accuracies else np.nan
        metrics.add_round_metrics(np.nan, last_acc)


# --- Loop Principal ---
def run_simulation_and_training(args):
    """Loop principal que integra simulação SUMO e treino FL."""
    print("=" * 50)
    print("Iniciando Simulação VFL com SUMO")
    print("=" * 50)
    print(f"Distribuição de Dados: {args.distribution}")
    if args.distribution == 'dirichlet':
        print(f"  Parâmetro Alpha (Dirichlet): {args.alpha}")
    print(f"Número de Rodadas FL: {args.num_rounds}")
    print(f"Clientes por Rodada: {args.clients_per_round}")
    print(f"Passos SUMO por Rodada FL: {args.steps_per_round}")
    print(f"RSU Posição: {args.rsu_pos}")
    print(f"RSU Alcance: {args.rsu_range}m")
    print(f"Máximo Clientes com Dados: {args.max_clients}")
    print("-" * 50)

    # Configura device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    model = FedModel(input_dim=3, output_dim=10).to(device)
    metrics = FLMetrics()

    sumoCmd = [args.sumo_binary, "-c", args.sumo_cfg, "--start", "--quit-on-end", "--step-length", "0.1"] # Exemplo step-length

    active_vehicles_map = {}
    all_client_datasets = None # Inicializa

    try:
        # Distribui os dados ANTES de iniciar SUMO
        print(f"Distribuindo dataset CIFAR-10 ({args.distribution}) para até {args.max_clients} clientes potenciais...")
        if args.distribution == 'iid':
            all_client_datasets = create_iid_client_datasets(args.max_clients, train_dataset)
        elif args.distribution == 'dirichlet':
            all_client_datasets = create_dirichlet_client_datasets(args.max_clients, train_dataset, args.alpha)

        if not all_client_datasets or len(all_client_datasets) < args.max_clients:
             print(f"Aviso: Não foi possível criar datasets para todos os {args.max_clients} clientes potenciais (criados: {len(all_client_datasets) if all_client_datasets else 0}).")
             if not all_client_datasets:
                 print("Erro fatal: Nenhum dataset de cliente criado. Encerrando.")
                 return

        dataset_assignment_index = 0

        print("Iniciando SUMO e TraCI...")
        traci.start(sumoCmd)
        print("TraCI iniciado.")

        step = 0
        round_num = 0

        print(f"Simulação iniciada. RSU em {args.rsu_pos}, Alcance: {args.rsu_range}m.")
        print(f"Rodada FL a cada {args.steps_per_round} passos. Total de {args.num_rounds} rodadas.")

        # Loop principal da simulação
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # --- Gerenciamento Dinâmico de Veículos ---
            current_vehicle_ids = set(traci.vehicle.getIDList())
            existing_vehicle_ids = set(active_vehicles_map.keys())

            departed_ids = existing_vehicle_ids - current_vehicle_ids
            for v_id in departed_ids:
                if v_id in active_vehicles_map:
                    del active_vehicles_map[v_id]

            arrived_ids = current_vehicle_ids - existing_vehicle_ids
            for v_id in arrived_ids:
                 if dataset_assignment_index < len(all_client_datasets):
                     vehicle_dataset = all_client_datasets[dataset_assignment_index]
                     if vehicle_dataset and len(vehicle_dataset) > 0:
                          active_vehicles_map[v_id] = Vehicle(v_id, vehicle_dataset)
                          dataset_assignment_index += 1
                     else: # Dataset vazio, apenas incrementa
                          # print(f"Aviso: Dataset {dataset_assignment_index} vazio para novo veículo {v_id}.") # Log Opcional
                          dataset_assignment_index += 1
                 else: # Sem mais datasets pré-alocados
                     #print(f"Aviso: Veículo {v_id} entrou, sem mais datasets.") # Log Opcional
                     pass


            # --- Lógica para iniciar Rodada FL ---
            if step > 0 and step % args.steps_per_round == 0 and round_num < args.num_rounds:
                print(f"\n--- Passo SUMO {step}: Verificando para Rodada FL {round_num + 1}/{args.num_rounds} ---")

                current_active_ids = set(active_vehicles_map.keys()) # Pega IDs dos veículos *gerenciados*
                vehicles_to_check = [active_vehicles_map[vid] for vid in current_active_ids if active_vehicles_map[vid].dataloader is not None] # Verifica apenas os com dados

                if not vehicles_to_check:
                     print("  Nenhum veículo ativo com dados no momento.")
                     metrics.add_round_metrics(np.nan, np.nan) # Adiciona métrica vazia
                     # Não incrementa round_num se a rodada não pode acontecer
                     step += 1
                     continue # Pula para o próximo passo SUMO

                # Filtra por conectividade
                vehicles_in_range = []
                print(f"  Verificando conectividade para {len(vehicles_to_check)} veículos com dados...")
                for vehicle in vehicles_to_check:
                     if vehicle.is_in_range(args.rsu_pos, args.rsu_range):
                         vehicles_in_range.append(vehicle)

                print(f"  {len(vehicles_in_range)} veículos estão DENTRO do alcance ({args.rsu_range}m).")

                if vehicles_in_range:
                    num_available_clients = len(vehicles_in_range)
                    num_to_select = min(args.clients_per_round, num_available_clients)

                    if num_to_select > 0:
                         print(f"  Selecionando {num_to_select} clientes dentre os {num_available_clients} em alcance.")
                         selected_vehicles = np.random.choice(vehicles_in_range, num_to_select, replace=False)

                         print(f"--- Iniciando Rodada FL {round_num + 1}/{args.num_rounds} ---")
                         # Passa o device para a função da rodada
                         federated_learning_round(model, selected_vehicles, metrics, round_num, test_loader, device)
                         round_num += 1 # Incrementa SÓ SE a rodada foi executada
                    else:
                         print("  Seleção resultou em 0 clientes. Pulando treino.")
                         metrics.add_round_metrics(np.nan, np.nan)
                else:
                     print(f"--- Passo {step}: Pular rodada FL (sem veículos em alcance).")
                     metrics.add_round_metrics(np.nan, np.nan)

                # Verifica se completou todas as rodadas FL planejadas
                if round_num >= args.num_rounds:
                    print(f"\nNúmero alvo de {args.num_rounds} rodadas FL atingido.")
                    break # Sai do loop while da simulação

            step += 1
            # time.sleep(0.01) # Pausa opcional

        print("\nSimulação SUMO terminada ou número de rodadas FL atingido.")

    except traci.TraCIException as e:
        print(f"\nErro fatal no TraCI: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"\nErro inesperado no script: {e}")
        traceback.print_exc()
    finally:
        print("Tentando fechar conexão TraCI...")
        try:
            if traci.isLoaded():
                traci.close()
                print("Conexão TraCI fechada.")
            else:
                print("Conexão TraCI já estava fechada ou não foi iniciada.")
        except Exception as e_close:
            print(f"Erro ao tentar fechar TraCI: {e_close}")

    # Mostra e salva resultados
    print("\nPlotando e salvando métricas de treino...")
    metrics.plot_metrics()
    # metrics.save_metrics()


if __name__ == "__main__":
    # --- Configuração do Argument Parser ---
    parser = argparse.ArgumentParser(description="Executa simulação VFL com SUMO e CIFAR-10.")

    # Argumentos da Simulação SUMO
    parser.add_argument('--sumo_cfg', type=str, default=SUMO_CONFIG_FILE, help='Caminho para o arquivo .sumocfg')
    parser.add_argument('--sumo_binary', type=str, default=SUMO_BINARY, choices=['sumo', 'sumo-gui'], help='Executável do SUMO a usar (sumo ou sumo-gui)')

    # Argumentos do Federated Learning
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS, help='Número total de rodadas FL.')
    parser.add_argument('--clients_per_round', type=int, default=CLIENTS_PER_ROUND, help='Número de clientes a selecionar por rodada.')
    parser.add_argument('--steps_per_round', type=int, default=SIMULATION_STEPS_PER_ROUND, help='Número de passos SUMO entre cada rodada FL.')
    parser.add_argument('--max_clients', type=int, default=NUM_CLIENTS_MAX, help='Número máximo de clientes para os quais pré-dividir dados.')

    # Argumentos da Distribuição de Dados
    parser.add_argument('-d', '--distribution', type=str, default='iid', choices=['iid', 'dirichlet'], help='Tipo de distribuição de dados (iid ou dirichlet).')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='Alpha da Dirichlet (se distribution=dirichlet). Menor alpha = mais Não-IID.')

    # Argumentos da Conectividade
    # Permite especificar RSU_POS como dois floats
    parser.add_argument('--rsu_pos', type=float, nargs=2, default=[RSU_POSITION[0], RSU_POSITION[1]], help='Coordenadas X Y da RSU.')
    parser.add_argument('--rsu_range', type=float, default=MAX_COMMUNICATION_RANGE, help='Raio de comunicação da RSU em metros.')

    args = parser.parse_args()

    # Converte a lista de rsu_pos de volta para tupla
    args.rsu_pos = tuple(args.rsu_pos)

    # Verifica se o arquivo de configuração do SUMO existe
    if not os.path.exists(args.sumo_cfg):
         print(f"Erro: Arquivo de configuração do SUMO não encontrado em '{args.sumo_cfg}'")
         sys.exit(1)

    # Chama a função principal passando o objeto 'args'
    run_simulation_and_training(args)

    print("\nExecução do script principal concluída.")
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from fed_model import FedModel       # Importa do modelo FL
from vehicle_manager import Vehicle  # Importa a classe Vehicle
import connectivity                  # <--- Importa o novo módulo de conectividade
import traci
import sys
import os
# import math # Não mais necessário aqui diretamente, pois está em connectivity.py
import argparse
import traceback
import time # Opcional, para pausas

# Adicione SUMO_HOME/tools ao PYTHONPATH
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path: # Evita adicionar múltiplas vezes
        sys.path.append(tools)
    # print(f"Adicionado ao sys.path: {tools}") # Log opcional
else:
    sys.exit("Erro: Variável de ambiente 'SUMO_HOME' não declarada.")


# --- Constantes e Configurações (padrões, podem ser sobrescritas por args) ---
# Constantes relacionadas ao FL e SUMO
NUM_ROUNDS_DEFAULT = 10
CLIENTS_PER_ROUND_DEFAULT = 10
# BATCH_SIZE_DEFAULT = 32 # Usado apenas para test_loader agora, que é fixo
SIMULATION_STEPS_PER_ROUND_DEFAULT = 50
NUM_CLIENTS_MAX_DEFAULT = 50
SUMO_CONFIG_FILE_DEFAULT = "manhattan.sumocfg" # Verifique este caminho!
SUMO_BINARY_DEFAULT = "sumo-gui"        # Mude para "sumo" para rodar sem GUI
# Constantes relacionadas à RSU/Conectividade agora vêm dos defaults em connectivity.py ou args
RSU_POSITION_DEFAULT = (450.0, 450.0)


# --- Datasets e Transformações ---
print("Configurando datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

try:
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Diretório './data' criado.")

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
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
        if not self.round_losses and not self.round_accuracies:
            print("Nenhuma métrica foi registrada para plotar.")
            return

        plt.figure(figsize=(12, 5))
        rounds_available = range(1, len(self.round_losses) + 1)

        valid_losses = [(r, l) for r, l in zip(rounds_available, self.round_losses) if l is not None and not np.isnan(l)]
        if valid_losses:
            rounds_l, losses_l = zip(*valid_losses)
            plt.subplot(1, 2, 1)
            plt.plot(rounds_l, losses_l, marker='o', linestyle='-', color='tab:red', label='Loss Média (Clientes Tx Sucesso)')
            plt.title('Loss Média por Rodada FL')
            plt.xlabel('Rodada FL')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

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
            plt.show(block=False) # block=False para não travar o script em alguns ambientes
            plt.pause(1) # Pequena pausa para renderizar
        except Exception as e_plot:
            print(f"Não foi possível exibir o gráfico interativamente: {e_plot}")


# --- Função de Validação Global ---
def validate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
            outputs = model(data_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    return correct / total if total > 0 else 0.0

# --- Funções de Distribuição de Dados ---
def create_iid_client_datasets(num_clients, dataset):
    print(f"Criando {num_clients} datasets IID...")
    num_samples = len(dataset)
    if num_clients == 0 or num_samples == 0: return []
    if num_clients > num_samples: num_clients = num_samples

    indices = list(range(num_samples))
    np.random.shuffle(indices)
    data_per_client = num_samples // num_clients
    all_client_subsets = []
    # print(f"  Distribuindo ~{data_per_client} amostras por cliente...") # Log verboso
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        if client_indices:
             all_client_subsets.append(data.Subset(dataset, client_indices))
    print(f"  {len(all_client_subsets)} subconjuntos IID criados.")
    return all_client_subsets

def create_dirichlet_client_datasets(num_clients, dataset, alpha):
    print(f"Criando {num_clients} datasets Não-IID com Dirichlet (alpha={alpha})...")
    if num_clients == 0: return []
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        print("Erro: Dataset precisa do atributo 'targets' para Dirichlet por classe. Usando IID como fallback.")
        return create_iid_client_datasets(num_clients, dataset)

    num_samples = len(labels)
    if num_samples == 0: return []
    num_classes = len(np.unique(labels))
    # print(f"  Dataset tem {num_samples} amostras e {num_classes} classes.") # Log verboso

    if num_clients > num_samples:
        # print(f"Aviso: Número de clientes ({num_clients}) > amostras ({num_samples}). Limitando a {num_samples}.") # Log verboso
        num_clients = num_samples
    if num_clients == 0: return []

    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    min_samples_per_class_actual = min(len(idx) for idx in idx_by_class if len(idx) > 0) # Considera classes que podem estar vazias
    if min_samples_per_class_actual < 1:
        print("Aviso: Uma ou mais classes não têm amostras suficientes no dataset. A distribuição Dirichlet pode ser instável. Tentando IID.")
        return create_iid_client_datasets(num_clients, dataset)

    client_indices = [[] for _ in range(num_clients)]
    # print(f"  Distribuindo amostras de {num_classes} classes...") # Log verboso

    for k in range(num_classes):
        class_indices = idx_by_class[k]
        if not class_indices.size > 0: continue # Pula classes vazias
        np.random.shuffle(class_indices)
        num_samples_in_class = len(class_indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        target_samples_per_client = (proportions * num_samples_in_class).astype(int)

        diff = num_samples_in_class - target_samples_per_client.sum()
        if diff != 0:
            adjust_indices = np.random.choice(num_clients, abs(diff), replace=(abs(diff) > num_clients))
            target_samples_per_client[adjust_indices] += np.sign(diff)
        target_samples_per_client = np.maximum(0, target_samples_per_client) # Garante não negativo
        final_diff = num_samples_in_class - target_samples_per_client.sum() # Re-checa soma
        if final_diff != 0: target_samples_per_client[np.random.choice(num_clients)] += final_diff

        current_idx = 0
        for client_id in range(num_clients):
            num_assigned = target_samples_per_client[client_id]
            if num_assigned > 0 : # So atribui se houver samples
                assigned_indices = class_indices[current_idx : current_idx + num_assigned]
                client_indices[client_id].extend(assigned_indices)
                current_idx += num_assigned
        # if current_idx != num_samples_in_class: # Log verboso
            # print(f"Aviso: Discrepância na distribuição da classe {k} ({current_idx}/{num_samples_in_class}).")

    all_client_subsets = []
    # print("  Criando Subsets para cada cliente...") # Log verboso
    total_assigned_samples = 0
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        subset = data.Subset(dataset, client_indices[i]) # Pode ser vazio se o cliente não recebeu dados
        all_client_subsets.append(subset)
        total_assigned_samples += len(subset)

    print(f"  {len(all_client_subsets)} subconjuntos Não-IID (Dirichlet) criados.")
    # if total_assigned_samples != num_samples: # Log verboso
        # print(f"  Aviso: Número total de amostras distribuídas ({total_assigned_samples}) não bate com o original ({num_samples}).")
    return all_client_subsets

# --- Função da Rodada FL ---
def federated_learning_round(model, selected_vehicles_with_pdr, metrics, round_num, test_loader, device):
    """Executa uma rodada de FL, simulando sucesso de transmissão baseado em PDR."""
    client_weights = []
    client_losses = []
    num_successful_transmissions = 0

    # print(f"  Iniciando treino local para {len(selected_vehicles_with_pdr)} veículos selecionados...") # Log verboso
    model.train()

    for vehicle, pdr_at_selection in selected_vehicles_with_pdr:
        if vehicle.dataloader is None: # Veículo pode não ter dados (ex: Dirichlet com alpha baixo)
            continue

        try:
            weights, loss = model.train_local(vehicle, epochs=1)

            if loss is not None and not np.isnan(loss) and loss >= 0:
                # Usa a função do módulo connectivity para simular sucesso da transmissão
                if connectivity.simulate_transmission_success(pdr_at_selection):
                    client_weights.append(weights)
                    client_losses.append(loss)
                    num_successful_transmissions += 1
                    # print(f"    Veículo {vehicle.id}: Treino (Loss {loss:.3f}), Tx OK (PDR {pdr_at_selection:.2f})") # Log
                # else: # Log opcional
                    # print(f"    Veículo {vehicle.id}: Treino (Loss {loss:.3f}), Tx FALHOU (PDR {pdr_at_selection:.2f})")
        except Exception as e:
            print(f"    Erro CRÍTICO ao treinar veículo {vehicle.id}: {e}")
            traceback.print_exc()

    if client_weights:
        print(f"  Agregando pesos de {num_successful_transmissions} clientes (Tx bem-sucedidas).")
        try:
            with torch.no_grad():
                avg_weights = [ np.mean(np.stack(layer_weights, axis=0), axis=0) for layer_weights in zip(*client_weights) ]
                model.set_weights(avg_weights)
            # print("  Modelo global atualizado.") # Log verboso

            avg_contributing_loss = np.mean(client_losses) if client_losses else np.nan
            global_accuracy = validate_model(model, test_loader, device)
            metrics.add_round_metrics(avg_contributing_loss, global_accuracy)
            print(f"  Rodada FL {round_num+1} Métricas - Loss Média (Tx Sucesso): {avg_contributing_loss:.4f}, Acurácia (Teste): {global_accuracy*100:.2f}%")

        except Exception as e_agg:
            print(f"  Erro durante agregação/validação: {e_agg}")
            traceback.print_exc(); metrics.add_round_metrics(np.nan, np.nan)
    else:
        print("  Nenhum cliente transmitiu pesos com sucesso nesta rodada.")
        last_acc = metrics.round_accuracies[-1] if metrics.round_accuracies else np.nan
        metrics.add_round_metrics(np.nan, last_acc)


# --- Loop Principal ---
def run_simulation_and_training(args):
    """Loop principal que integra simulação SUMO e treino FL."""
    print("=" * 60); print(" Iniciando Simulação VFL com SUMO ".center(60, "=")); print("=" * 60)
    print(f"  Distribuição: {args.distribution}" + (f" (Alpha: {args.alpha})" if args.distribution == 'dirichlet' else ""))
    print(f"  Rodadas FL: {args.num_rounds}, Clientes/Rodada: {args.clients_per_round}, Passos/Rodada: {args.steps_per_round}")
    # Cria o dicionário de parâmetros PDR a partir dos args
    pdr_params = {'d_mid': args.pdr_dmid, 'k_steepness': args.pdr_k, 'max_effective_range': args.pdr_max_range}
    print(f"  RSU Posição: {args.rsu_pos}, Params PDR: {pdr_params}, Min PDR: {args.min_pdr}")
    print(f"  Máximo Clientes com Dados: {args.max_clients}, SUMO Bin: {args.sumo_binary}")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Usando device: {device}")
    model = FedModel(input_dim=3, output_dim=10).to(device); metrics = FLMetrics()
    sumoCmd = [args.sumo_binary, "-c", args.sumo_cfg, "--start", "--quit-on-end", "--step-length", "0.1"]
    active_vehicles_map = {}

    try:
        print(f"Distribuindo dataset CIFAR-10 ({args.distribution}) para até {args.max_clients} clientes...")
        if args.distribution == 'iid': all_client_datasets = create_iid_client_datasets(args.max_clients, train_dataset)
        elif args.distribution == 'dirichlet': all_client_datasets = create_dirichlet_client_datasets(args.max_clients, train_dataset, args.alpha)
        else: # Fallback
            print(f"Erro: Distribuição '{args.distribution}' não reconhecida. Usando IID."); all_client_datasets = create_iid_client_datasets(args.max_clients, train_dataset)

        if not all_client_datasets: print("Erro fatal: Nenhum dataset de cliente criado."); return
        if len(all_client_datasets) < args.max_clients : print(f"Aviso: Criados apenas {len(all_client_datasets)}/{args.max_clients} datasets.")
        dataset_assignment_index = 0

        print("Iniciando SUMO e TraCI..."); traci.start(sumoCmd); print("TraCI iniciado.")
        step = 0; round_num = 0
        print(f"Simulação iniciada. Rodada FL a cada {args.steps_per_round} passos. Total: {args.num_rounds} rodadas.")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            current_vehicle_ids = set(traci.vehicle.getIDList()); existing_vehicle_ids = set(active_vehicles_map.keys())
            for v_id in (existing_vehicle_ids - current_vehicle_ids): # Departed
                if v_id in active_vehicles_map: del active_vehicles_map[v_id]
            for v_id in (current_vehicle_ids - existing_vehicle_ids): # Arrived
                 if dataset_assignment_index < len(all_client_datasets):
                     vehicle_dataset = all_client_datasets[dataset_assignment_index]
                     if vehicle_dataset and len(vehicle_dataset) > 0: active_vehicles_map[v_id] = Vehicle(v_id, vehicle_dataset)
                     dataset_assignment_index += 1

            if step > 0 and step % args.steps_per_round == 0 and round_num < args.num_rounds:
                print(f"\n--- Passo SUMO {step}: Verificando para Rodada FL {round_num + 1}/{args.num_rounds} ---")
                current_managed_vehicles = [v for v_id, v in active_vehicles_map.items() if v.dataloader is not None]

                if not current_managed_vehicles:
                    print("  Nenhum veículo ativo com dados no momento."); metrics.add_round_metrics(np.nan, np.nan)
                    step += 1; continue

                candidate_vehicles_with_pdr = []
                # print(f"  Calculando PDR para {len(current_managed_vehicles)} veículos (RSU: {args.rsu_pos})...") # Log verboso
                for vehicle in current_managed_vehicles:
                    pos = vehicle.get_position()
                    if pos:
                        pdr = connectivity.calculate_pdr(pos, args.rsu_pos, pdr_params)
                        if pdr >= args.min_pdr:
                            candidate_vehicles_with_pdr.append((vehicle, pdr))

                print(f"  {len(candidate_vehicles_with_pdr)} veículos são candidatos (PDR >= {args.min_pdr}).")

                if candidate_vehicles_with_pdr:
                    num_to_select = min(args.clients_per_round, len(candidate_vehicles_with_pdr))
                    if num_to_select > 0:
                         # print(f"  Selecionando {num_to_select} clientes...") # Log verboso
                         indices_selecionados = np.random.choice(len(candidate_vehicles_with_pdr), num_to_select, replace=False)
                         selected_vehicles_and_their_pdrs = [candidate_vehicles_with_pdr[i] for i in indices_selecionados]

                         # print(f"--- Iniciando Rodada FL {round_num + 1}/{args.num_rounds} ---") # Log verboso
                         federated_learning_round(model, selected_vehicles_and_their_pdrs, metrics, round_num, test_loader, device)
                         round_num += 1
                    else: print("  Seleção resultou em 0 clientes."); metrics.add_round_metrics(np.nan, np.nan)
                else: print(f"--- Passo {step}: Pular rodada FL (sem candidatos com PDR suficiente)."); metrics.add_round_metrics(np.nan, np.nan)

                if round_num >= args.num_rounds: print(f"\nNúmero alvo de {args.num_rounds} rodadas FL atingido."); break
            step += 1
            # time.sleep(0.001) # Pausa muito pequena opcional

        print("\nSimulação SUMO terminada ou número de rodadas FL atingido.")

    except traci.TraCIException as e: print(f"\nErro fatal no TraCI: {e}"); traceback.print_exc()
    except KeyboardInterrupt: print("\nSimulação interrompida (Ctrl+C).")
    except Exception as e: print(f"\nErro inesperado: {e}"); traceback.print_exc()
    finally:
        print("Tentando fechar conexão TraCI...");
        try:
            if traci.isLoaded(): traci.close(); print("Conexão TraCI fechada.")
            else: print("Conexão TraCI já fechada/não iniciada.")
        except Exception as e_close: print(f"Erro ao fechar TraCI: {e_close}")

    print("\nPlotando e salvando métricas de treino..."); metrics.plot_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa simulação VFL com SUMO e CIFAR-10.")
    # Simulação SUMO
    parser.add_argument('--sumo_cfg', type=str, default=SUMO_CONFIG_FILE_DEFAULT, help='Caminho .sumocfg')
    parser.add_argument('--sumo_binary', type=str, default=SUMO_BINARY_DEFAULT, choices=['sumo', 'sumo-gui'], help='Executável SUMO')
    # Federated Learning
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS_DEFAULT, help='Número de rodadas FL.')
    parser.add_argument('--clients_per_round', type=int, default=CLIENTS_PER_ROUND_DEFAULT, help='Clientes por rodada.')
    parser.add_argument('--steps_per_round', type=int, default=SIMULATION_STEPS_PER_ROUND_DEFAULT, help='Passos SUMO por rodada FL.')
    parser.add_argument('--max_clients', type=int, default=NUM_CLIENTS_MAX_DEFAULT, help='Max clientes para dividir dados.')
    # Distribuição de Dados
    parser.add_argument('-d', '--distribution', type=str, default='iid', choices=['iid', 'dirichlet'], help='Distribuição (iid ou dirichlet).')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='Alpha da Dirichlet (se -d dirichlet).')
    # Conectividade (usando defaults de connectivity.py se não especificado)
    parser.add_argument('--rsu_pos', type=float, nargs=2, default=list(RSU_POSITION_DEFAULT), help='Coords X Y da RSU.')
    parser.add_argument('--pdr_dmid', type=float, default=connectivity.DEFAULT_PDR_PARAMS['d_mid'], help='PDR Model: d_mid (distância para PDR 0.5).')
    parser.add_argument('--pdr_k', type=float, default=connectivity.DEFAULT_PDR_PARAMS['k_steepness'], help='PDR Model: k_steepness (inclinação da sigmoide).')
    parser.add_argument('--pdr_max_range', type=float, default=connectivity.DEFAULT_PDR_PARAMS['max_effective_range'], help='PDR Model: max_effective_range (m).')
    parser.add_argument('--min_pdr', type=float, default=connectivity.DEFAULT_MIN_PDR_FOR_CANDIDACY, help='PDR mínimo para veículo ser candidato.')

    args = parser.parse_args()
    args.rsu_pos = tuple(args.rsu_pos)

    if not os.path.exists(args.sumo_cfg): print(f"Erro: SUMO config não encontrado em '{args.sumo_cfg}'"); sys.exit(1)

    run_simulation_and_training(args)
    print("\nExecução do script principal concluída.")
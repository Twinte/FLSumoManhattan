# Simulação de Federated Learning Veicular com SUMO e CIFAR-10

Este projeto simula um cenário de Aprendizado Federado (Federated Learning - FL) em um ambiente veicular utilizando o simulador de tráfego SUMO. Veículos na simulação atuam como clientes que colaboram para treinar um modelo de classificação de imagens (CIFAR-10) sem compartilhar seus dados brutos, utilizando o algoritmo Federated Averaging (FedAvg).

## Visão Geral

O objetivo é integrar uma simulação de tráfego realista com um processo de treinamento de FL distribuído. A simulação permite estudar como a dinâmica veicular (movimento, entrada/saída de veículos) interage com o processo de FL.

**Componentes Principais:**

* **Simulação de Tráfego (SUMO):** Utiliza o SUMO para simular o movimento de veículos em uma malha viária (baseado no cenário "Manhattan"). Arquivos de configuração incluem a rede (`.net.xml`), rotas/fluxos (`.rou.xml`/`.flow.xml`) e a configuração geral (`.sumocfg`).
* **Orquestrador FL (`fedl.py`):** Script Python principal que:
    * Inicia e controla a simulação SUMO via TraCI.
    * Gerencia o ciclo de vida dos veículos/clientes FL.
    * Coordena as rodadas de treinamento federado.
    * Seleciona clientes para cada rodada.
    * Agrega as atualizações do modelo (FedAvg).
    * Avalia o modelo global e registra métricas.
* **Modelo de ML (`fed_model.py`):** Define a arquitetura da Rede Neural Convolucional (CNN) usada para classificação do CIFAR-10 e implementa a lógica de treinamento local (`train_local`) que é executada em cada cliente/veículo.
* **Gerenciador de Veículos (`vehicle_manager.py`):** Define a classe `Vehicle`, que representa cada cliente FL. Essa classe gerencia o subconjunto de dados local (CIFAR-10) e interage com o TraCI para obter informações do veículo no SUMO (se necessário).
* **Dataset:** CIFAR-10, um dataset padrão para classificação de imagens, dividido entre os veículos participantes.

## Pré-requisitos

Antes de executar o projeto, certifique-se de ter instalado:

1.  **SUMO:**
    * Instale o SUMO (Simulation of Urban MObility). Visite [https://sumo.dlr.de/](https://sumo.dlr.de/).
    * **Importante:** Configure a variável de ambiente `SUMO_HOME` para apontar para o diretório de instalação do SUMO. O script Python precisa disso para encontrar as ferramentas do TraCI.
2.  **Python:**
    * Python 3.8 ou superior recomendado.
3.  **Bibliotecas Python:**
    * PyTorch (`torch`)
    * Torchvision (`torchvision`)
    * NumPy (`numpy`)
    * Matplotlib (`matplotlib`)

Recomenda-se criar um ambiente virtual e instalar as dependências usando um arquivo `requirements.txt`.

## Configuração

1.  **Clone o Repositório (se aplicável):**
    ```bash
    # git clone [URL_DO_SEU_REPOSITORIO]
    # cd [NOME_DA_PASTA_DO_PROJETO]
    cd sumo-manhattan
    ```

2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    # venv\Scripts\activate
    # No Linux/macOS:
    # source venv/bin/activate
    ```

3.  **Instale as Dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo:
    ```text
    torch
    torchvision
    numpy
    matplotlib
    ```
    E então instale:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: O `traci` vem com a instalação do SUMO e será encontrado se `SUMO_HOME` estiver configurado corretamente).*

4.  **Dataset CIFAR-10:**
    O script `fedl.py` está configurado para baixar automaticamente o dataset CIFAR-10 na primeira execução e salvá-lo na pasta `./data`. Certifique-se de ter conexão com a internet na primeira vez que rodar.

5.  **Verifique a Configuração do SUMO:**
    * Confirme que a variável de ambiente `SUMO_HOME` está definida.
    * Verifique se o caminho para o arquivo de configuração do SUMO (`.sumocfg`) dentro do script `fedl.py` (na variável `sumo_config`) está correto em relação à estrutura do seu projeto. Atualmente está como `"sumo-manhattan/manhattan.sumocfg"`, ajuste se necessário.

## Execução

1.  **Navegue até o Diretório Raiz:**
    Certifique-se de que você está no diretório que contém o script `fedl.py`.

2.  **Execute o Script Principal:**
    ```bash
    python fedl.py
    ```

3.  **Acompanhe a Simulação:**
    * Se `sumo_binary` em `fedl.py` estiver configurado como `"sumo-gui"`, a interface gráfica do SUMO será aberta, permitindo visualizar o tráfego.
    * O terminal exibirá logs sobre o progresso da simulação, início/fim das rodadas de FL, treinamento dos clientes, agregação e métricas (loss e acurácia).
    * Ao final da execução, gráficos mostrando a evolução da loss e da acurácia do modelo global ao longo das rodadas de FL serão exibidos.

## Fluxo de Funcionamento

1.  O `fedl.py` inicia o SUMO (GUI ou headless) e estabelece a conexão via TraCI.
2.  A simulação SUMO começa, e veículos são adicionados conforme definido nos arquivos de rotas/fluxos.
3.  O script `fedl.py` detecta os veículos ativos na simulação a cada passo.
4.  Novos veículos recebem um subconjunto de dados do CIFAR-10 (pré-dividido).
5.  A simulação avança passo a passo (`traci.simulationStep()`).
6.  A cada `SIMULATION_STEPS_PER_ROUND` passos de simulação, uma rodada de FL é iniciada:
    * Um número (`CLIENTS_PER_ROUND`) de veículos ativos é selecionado aleatoriamente.
    * Os veículos selecionados treinam o modelo global atual em seus dados locais por um número definido de épocas (atualmente 1 época local).
    * Os pesos atualizados de cada cliente são enviados de volta ao orquestrador.
    * O orquestrador calcula a média dos pesos recebidos (FedAvg) e atualiza o modelo global.
    * O modelo global é avaliado no conjunto de teste do CIFAR-10.
    * As métricas são registradas.
7.  O processo continua até que o número alvo de rodadas (`NUM_ROUNDS`) seja atingido ou a simulação SUMO termine.
8.  A conexão TraCI é fechada.
9.  Os gráficos de métricas são exibidos.

## Estrutura do Projeto (Exemplo)

sumo-manhattan/
├── data/                     # Dados (CIFAR-10 será baixado aqui)
├── sumo-manhattan/           # Arquivos específicos do cenário SUMO
│   ├── manhattan.sumocfg     # Configuração principal do SUMO
│   ├── net.net.xml           # Definição da rede viária
│   ├── routes.xml            # Definição das rotas (ou flows.xml)
│   └── ...                   # Outros arquivos SUMO (polígonos, etc.)
├── fedl.py                   # Script principal (orquestrador FL + TraCI)
├── fed_model.py              # Definição do modelo de ML (CNN)
├── vehicle_manager.py        # Definição da classe Vehicle e interação com SUMO
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
└── venv/                     # Ambiente virtual Python (opcional)

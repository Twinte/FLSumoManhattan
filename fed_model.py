import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class FedModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=10):
        super().__init__()
        # --- Arquitetura da rede (como antes) ---
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512) # Ajuste o tamanho aqui se a saída da conv3/pool for diferente
        self.fc2 = nn.Linear(512, output_dim)
        
        # --- Otimizador e Loss (podem ser recriados a cada treino local ou mantidos) ---
        # Se mantidos, o estado do otimizador é persistente entre clientes, o que não é padrão em FL.
        # É mais comum recriar otimizador ou passar pesos e deixar cliente criar o seu.
        # Vamos manter por simplicidade, mas esteja ciente disso.
        self.optimizer = optim.Adam(self.parameters(), lr=0.001) 
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # --- Forward pass (como antes) ---
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        # Verifique se o pooling adaptativo para (4, 4) está correto para o tamanho da imagem CIFAR (32x32)
        # Após 2 max pools (32->16->8), a saída da conv3 é 8x8. 
        # AdaptiveAvgPool2d( (4,4) ) redimensionará para 4x4.
        x = F.adaptive_avg_pool2d(x, (4, 4)) 
        x = x.view(x.size(0), -1) # Flatten: 128 * 4 * 4 = 2048. Ajuste fc1 se necessário.
        # ***** CORREÇÃO IMPORTANTE: O tamanho achatado é 128 * 4 * 4 = 2048, não 128*4*4. Atualize fc1. *****
        # self.fc1 = nn.Linear(2048, 512) # Deveria ser inicializado assim no __init__
        x = F.relu(self.fc1(x)) # Se fc1 espera 2048, o init deve refletir isso.
        x = self.fc2(x)
        return x

    def train_local(self, vehicle, epochs=1):
        """Treina o modelo localmente usando o DataLoader do veículo."""
        
        # Verifica se o veículo tem um dataloader (tem dados)
        if vehicle.dataloader is None:
            print(f"    Aviso: Veículo {vehicle.id} não possui dataloader (sem dados). Pulando treino.")
            return self.get_weights(), None # Retorna None para loss indicar que não treinou

        # *** Importante: Para treino local FL, o cliente geralmente começa 
        # *** com os pesos GLOBAIS atuais. A função `set_weights` já deve ter sido
        # *** chamada no objeto `model` *antes* de chamar `train_local`.
        # *** Esta função treina o modelo *deste objeto* (que deve ter os pesos globais).

        # Coloca o modelo em modo de treino (caso estivesse em eval)
        self.train() 
        
        total_loss = 0.0
        total_samples = 0

        # Loop de épocas locais
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_data, batch_labels in vehicle.dataloader:
                # Mova dados para a GPU se estiver usando
                # batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                # Verifica se há NaNs nos dados (precaução)
                if torch.isnan(batch_data).any():
                    print(f"      Aviso: NaN detectado no batch de dados do veículo {vehicle.id}. Pulando batch.")
                    continue
                    
                if len(batch_data) == 0:
                     continue # Pula batch vazio

                self.optimizer.zero_grad()
                outputs = self(batch_data)
                
                try:
                    loss = self.loss_fn(outputs, batch_labels)

                    # Verifica se a loss é válida
                    if torch.isnan(loss):
                        print(f"      Aviso: Loss NaN encontrada para veículo {vehicle.id}. Pulando backward/step.")
                        continue # Não propaga gradientes NaN

                    loss.backward()
                    # Opcional: Gradient Clipping para evitar explosão de gradientes
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) 
                    self.optimizer.step()

                    epoch_loss += loss.item() * batch_data.size(0)
                    epoch_samples += batch_data.size(0)
                
                except Exception as e_train:
                     print(f"      Erro durante o treino do batch para veículo {vehicle.id}: {e_train}")
                     # Decide se quer continuar a época ou parar
                     break # Para a época atual para este cliente

            if epoch_samples > 0:
                avg_epoch_loss = epoch_loss / epoch_samples
                # print(f"    Época {epoch+1}/{epochs} para Veículo {vehicle.id} - Loss: {avg_epoch_loss:.4f}")
                total_loss += epoch_loss
                total_samples += epoch_samples
            else:
                 # Nenhuma amostra válida foi treinada na época
                 break # Sai do loop de épocas se não houver amostras válidas


        # Calcula a loss média de todas as épocas bem-sucedidas
        avg_total_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Retorna os pesos ATUALIZADOS do modelo local e a loss média
        return self.get_weights(), avg_total_loss

    def get_weights(self):
        # Retorna pesos como arrays numpy na CPU (como antes)
        return [param.data.cpu().numpy() for param in self.parameters()]

    def set_weights(self, weights):
        # Define pesos a partir de arrays numpy (como antes)
        # Garante que os pesos sejam movidos para o device correto (CPU/GPU)
        device = next(self.parameters()).device 
        with torch.no_grad():
            for param, weight_array in zip(self.parameters(), weights):
                 # Converte numpy array para tensor, garante tipo float e move para o device
                 weight_tensor = torch.from_numpy(np.array(weight_array)).float().to(device)
                 # Verifica se as formas coincidem antes de atribuir
                 if param.data.shape == weight_tensor.shape:
                     param.data.copy_(weight_tensor)
                 else:
                      print(f"Erro set_weights: Discrepância de forma! Param: {param.data.shape}, Peso recebido: {weight_tensor.shape}")
                      # Lança um erro ou lida com a situação
                      raise ValueError(f"Discrepância de forma ao definir pesos para o parâmetro {param.name}")
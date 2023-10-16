# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Cargar el conjunto de datos Red Karate Club
dataset = dgl.data.KarateClubDataset()

# Obtener el grafo y las etiquetas
g = dataset[0]
labels = g.ndata['label']

# Visualizar el grafo con NetworkX
nx_G = g.to_networkx()
pos = nx.spring_layout(nx_G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(nx_G, pos, with_labels=True, node_color=labels, cmap=plt.cm.Set1)
plt.title("Red Karate Club Graph")
plt.show()

# Obtener la matriz de adyacencia
adjacency_matrix = np.array(g.adjacency_matrix().to_dense())

# Visualizar la matriz de adyacencia
plt.figure(figsize=(8, 6))
plt.imshow(adjacency_matrix, cmap='viridis', origin='upper', aspect='equal')
plt.colorbar()
plt.title("Matriz de Adyacencia")
plt.show()

# Definir una capa de propagación de grafos (Graph Convolution Layer - GCN)
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Aplicar la capa lineal a los nodos del grafo
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum(msg='m', out='h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h = self.linear(h_neigh)
            return h

# Definir la red GCN
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_size)
        self.layer2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

# Parámetros de la red
input_dim = 1  # Dimensión de entrada (no hay características)
hidden_dim = 16  # Dimensión oculta
num_classes = 2  # Número de clases de salida

# Crear la red GCN
net = GCN(input_dim, hidden_dim, num_classes)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(100):
    logits = net(g, torch.ones(g.number_of_nodes(), 1))  # Usar un tensor de unos como entrada
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch + 1} | Loss: {loss.item():.4f}')

# Evaluar el modelo
net.eval()
logits = net(g, torch.ones(g.number_of_nodes(), 1))
predicted_labels = torch.argmax(logits, dim=1)
accuracy = torch.sum(predicted_labels == labels).item() / len(labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

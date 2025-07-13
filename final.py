import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time

# using a simple regression model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output(x)
        return x

# creating a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# client declarations
class Client:
    def __init__(self, data, targets):
        self.dataset = SimpleDataset(data, targets)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.model = Model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def train(self):                                           # training
        self.model.train()
        for data, target in self.dataloader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()


    def get_parameters(self):
        return self.model.state_dict()     #  model parameters as a state dictionary

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)

"""    def print_model_parameters(self):
        print("Client model parameters:")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.data.numpy()}")
"""
# function to return client-specific information
def client_information(i):
    c1 = [20, 3, 5, 4, 1]  # [weight, particle to c2, particle to c3, anti-particle from c2, anti-particle from c3]
    c2 = [30, 4, 7, 3, 9]  # [weight, particle to c1, particle to c3, anti-particle from c1, anti-particle from c3]
    c3 = [40, 1, 9, 5, 7]  # [weight, particle to c1, particle to c2, anti-particle from c1, anti-particle from c2]

    # give the specific client's information based on index
    if i == 0:
        return c1
    if i == 1:
        return c2
    if i == 2:
        return c3

# aggregation protocol function
def protocol(client_parameter, client_number):
    particle = []
    antiparticle = []

    # calculating particle and antiparticle contributions based on client number
    if client_number == 0:
        particle.append(client_information(1)[0] * client_information(0)[1])
        particle.append(client_information(2)[0] * client_information(0)[2])
        antiparticle.append(client_information(1)[0] * client_information(0)[3])
        antiparticle.append(client_information(2)[0] * client_information(0)[4])

    if client_number == 1:
        particle.append(client_information(0)[0] * client_information(1)[1])
        particle.append(client_information(2)[0] * client_information(1)[2])
        antiparticle.append(client_information(0)[0] * client_information(1)[3])
        antiparticle.append(client_information(2)[0] * client_information(1)[4])

    if client_number == 2:
        particle.append(client_information(0)[0] * client_information(2)[1])
        particle.append(client_information(1)[0] * client_information(2)[2])
        antiparticle.append(client_information(0)[0] * client_information(2)[3])
        antiparticle.append(client_information(1)[0] * client_information(2)[4])

    return client_parameter + sum(particle) - sum(antiparticle)

# server
class Server:
    def __init__(self):
        self.model = Model()
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)

    def aggregation(self):
        global_state_dict = self.model.state_dict()
        num_clients = len(self.clients)

        for key in global_state_dict:          # parameters from all clients for the specific key
            client_params = [client.get_parameters()[key] for client in self.clients]
            aggregated_param = torch.zeros_like(client_params[0])

            for i in range(num_clients):
                aggregated_param += protocol(client_params[i], i)

            global_state_dict[key] = aggregated_param/90


        self.model.load_state_dict(global_state_dict)

    def broadcast_parameters(self):
        for client in self.clients:
            client.set_parameters(self.model.state_dict())

    def print_model_parameters(self):
        print("Server model parameters:")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.data.numpy()}")

# dummy data for each client
data_client1 = ([1], [2])
data_client2 = ([2], [4])
data_client3 = ([3], [6])

# server and clients
server = Server()
client1 = Client(*data_client1)
client2 = Client(*data_client2)
client3 = Client(*data_client3)

# add clients to the server
server.add_client(client1)
server.add_client(client2)
server.add_client(client3)

# start federated learning process
num_rounds = 20
for round in range(num_rounds):
    print(f"Round {round+1}")

    # client trains on its data and prints model parameters
    for client in server.clients:
        client.train()

    # server aggregates parameters with the custom protocol and broadcasts them
    print(f'Clients trained for round {round+1}')
    server.aggregation()
    print(f'Aggregation successful')
    server.broadcast_parameters()
    print(f'Parameters broadcasted to all clients')
    time.sleep(2)
    # print server model parameters after aggregation
    #server.print_model_parameters()

# print final model parameters at the end of 20 rounds
print("Final model parameters after all rounds:")
server.print_model_parameters()

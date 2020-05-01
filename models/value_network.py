import torch

class Value_Net(torch.nn.Module):
    def __init__(self, ob_dim, value_dim):
        if value_dim is None: value_dim = 1

        # Define the network architecture
        super().__init__()
        self.fc1 = torch.nn.Linear(ob_dim, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 25)
        self.fc4 = torch.nn.Linear(25, value_dim)
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
        self.tanh3 = torch.nn.Tanh()
        self.tanh4 = torch.nn.Tanh()


    def forward(self, obs):
        x = self.fc1(obs)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.fc4(x)
        return x

from utils import testModule
from utils import common as U
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(5,2)
        self.act1 = torch.nn.ReLU()

    def forward(self, x):
        return self.act1(self.fc1(x))


testNet = Net()
#testNet(torch.tensor([2.,3.]))
cur_param = U.get_flat_param(testNet)
print("current parameters are: ", cur_param)
testModule.update(testNet)
new_param = U.get_flat_param(testNet)
print("new parameters are: ", new_param)




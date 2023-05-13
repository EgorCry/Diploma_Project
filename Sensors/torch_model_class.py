def create_model():
    path = 'Sensors/torch_model.pth'

    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    import torch
    from torch.utils.data import Dataset

    batch_size = 64

    import torch.optim as optim

    input_dim = 4
    hidden_dim = 32
    output_dim = 1

    model = MyModel(input_dim, hidden_dim, output_dim)

    model.load_state_dict(torch.load(path))

    return model


if __name__ == '__main__':
    create_model()

import torch.cuda

from utils import CNNModel
from torchinfo import summary


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cnn = CNNModel(in_channels=3, out_features=2)
cnn = cnn.to(DEVICE)
summary(cnn, input_size=(32, 3, 300, 300))

torch.save(cnn.state_dict(), "model/test.pt")

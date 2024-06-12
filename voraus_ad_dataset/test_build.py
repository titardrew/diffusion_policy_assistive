import torch
import torch.nn.functional as F

F.conv1d(torch.zeros(3, 3, 3).to("cuda"), torch.zeros(3, 3, 3).to("cuda"))
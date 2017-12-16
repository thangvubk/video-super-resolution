import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class ModelFactory(object):
    
    def create_model(self, model_name):
        if model_name == 'VSRCNN':
            return VSRCNN()
        elif model_name == 'ESPCN':
            return ESPCN()
        elif model_name == 'VDCNN':
            return VDCNN()
        elif model_name == 'VRES':
            return VRES()
        elif model_name == 'VDSR':
            return VDSR()
        elif model_name == 'VRNET':
            return VRNET()
        elif model_name == 'MFCNN':
            return MFCNN()
        else:
            raise Exception('unknown model {}'.format(model_name))

class VSRCNN(nn.Module):
    """
    Model for SRCNN

    Low Resolution -> Conv1 -> Relu -> Conv2 -> Relu -> Conv3 -> High Resulution
    
    Args:
        - C1, C2, C3: num output channels for Conv1, Conv2, and Conv3
        - F1, F2, F3: filter size
    """
    def __init__(self,
                 C1=64, C2=32, C3=1,
                 F1=9, F2=1, F3=5):
        super(VSRCNN, self).__init__()
        self.name = 'VSRCNN'
        self.conv1 = nn.Conv2d(1, C1, F1, padding=4, bias=False) # in, out, kernel
        self.conv2 = nn.Conv2d(C1, C2, F2)
        self.conv3 = nn.Conv2d(C2, C3, F3, padding=2, bias=False)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        self.name = 'ESPCN'
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 9, 3, padding=1)
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.conv3(x)
        return x

class VDCNN(nn.Module):
    def __init__(self):
        super(VDCNN, self).__init__()
        self.name = 'VDCNN'
        self.conv_first = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1, bias=False)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _res_layer(self, x):
        res = x
        out = F.relu(self.conv_next(x))
        out = self.conv_next(out)
        out += res
        out = F.relu(out)
        return out

    def forward(self, x):
        res = x
        out = F.relu(self.conv_first(x))
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)

        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        #out = F.relu(self.conv_next(out))
        out = self.conv_last(out)
        out += res
        return out

class VRES(nn.Module):
    def __init__(self):
        super(VRES, self).__init__()
        self.name = 'VRES'
        self.conv_first = nn.Conv2d(5, 64, 3, padding=1, bias=False)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1, bias=False)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _res_layer(self, x):
        res = x
        out = F.relu(self.conv_next(x))
        out = self.conv_next(out)
        out += res
        out = F.relu(out)
        return out

    def forward(self, x):
        center = 2
        res = x[:, center, :, :]
        res = res.unsqueeze(1)
        out = F.relu(self.conv_first(x))
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self.conv_last(out)
        out += res
        return out


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.name = 'VDSR'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

class VRNET(nn.Module):
    def __init__(self):
        super(VRNET, self).__init__()
        self.name = 'VRNET'
        self.conv1 = nn.Conv2d(5, 64, 9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 1, 5, padding=2, bias=False)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class MFCNN(nn.Module):
    def __init__(self):
        super(MFCNN, self).__init__()
        self.name = 'MFCNN'
        self.conv1 = nn.Conv2d(5, 32, 9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


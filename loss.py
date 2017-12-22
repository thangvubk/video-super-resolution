import pytorch_ssim
import torch.nn as nn

MSE_and_SSIM_model = ['VRES', 'VRES5', 'VRES10', 'VRES7', 'VRES15']

def get_loss_fn(model_name):
    if model_name in MSE_and_SSIM_model:
        return MSE_and_SSIM_loss()
    else:
        return nn.MSELoss()

class MSE_and_SSIM_loss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSE_and_SSIM_loss, self).__init__()
        self.MSE = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.alpha = alpha

    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2) + (1 - self.alpha)*(1 - self.SSIM(img1, img2))
        return loss



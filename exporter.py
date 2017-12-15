import numpy as np
import scipy.misc
import os

class Exporter(object):
    def __init__(self, model_name, scale=None):
        self.model_name = model_name
        if scale is not None:
            self.scale = scale

    def _ESPCN_shuffle(self, imgs):
        
        outs = []
        for img in imgs:
            C, H, W = img.shape
            S = self.scale
            SH = S*H
            SW = S*W
            out = np.zeros((SH, SW))
            for h in range(SH):
                for w in range(SW):
                    out[h, w] = img[h%S*S + w%S, h//S, w//S]
            outs.append(out)
        return outs
            
        

    def export(self, psnrs, outputs):
        if self.model_name == 'ESPCN':
            outputs = self._ESPCN_shuffle(outputs)
        
        for i, img in enumerate(outputs):
            img_name = os.path.join('Results', self.model_name, 
                                    self.model_name+'_output%03d.png'%i)
            scipy.misc.imsave(img_name, img)
        
        with open(os.path.join('Results', self.model_name, self.model_name+'.txt'), 'w') as f:
            for i, psnr in enumerate(psnrs):
                print('Psnr img%d: %.3f' %(i, psnr))
                f.write('Psnr img%d: %.3f\n' %(i, psnr))

        print('Average test psnr: %.3f' %np.mean(psnrs))
        print('Finish!!!')

        

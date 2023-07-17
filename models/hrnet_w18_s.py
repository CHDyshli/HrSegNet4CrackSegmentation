import math

import paddle
import paddle.nn as nn

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm
import paddle.nn.functional as F
from paddleseg.models.backbones.hrnet import HRNet_W18



class SegHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super(SegHead, self).__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.relu = nn.ReLU()
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2D(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2DTranspose(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        self.conv = nn.Conv2D(in_channels=interplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out
    
@manager.MODELS.add_component
class HRNet_W18_S(nn.Layer):
    def __init__(self, 
                 num_classes=2,
                 ):
        super().__init__()
        self.hrnet = HRNet_W18()
        self.head = SegHead(inplanes=270, interplanes=64, outplanes=num_classes)

    def forward(self, x):
        logits = []
        h, w = paddle.shape(x)[2:]
        x = self.hrnet(x)
        x = x[0]
        x = self.head(x)
        logits.append(x)
        logits = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits]
        return logits
    


    def init_weights(self):
        for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)




# if __name__ == '__main__':
#     model = HRNet_W18_S()
#     input = paddle.rand([1, 3, 400, 400])
#     output  = model(input)
#     print(len(output))
#     for o in output:
#         print(o.shape)
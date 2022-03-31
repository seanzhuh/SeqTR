import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from seqtr.models import VIS_ENCODERS
from seqtr.utils import get_root_logger, is_main
from seqtr.models.utils import freeze_params, parse_yolo_weights


def darknet_conv(in_chs,
                 out_chs,
                 kernel_sizes,
                 strides,
                 norm_cfg=dict(type="BN2d"),
                 act_cfg=dict(type="LeakyReLU", negative_slope=0.1)):
    convs = []
    for i, (in_ch, out_ch, kernel_size, stride) in enumerate(zip(in_chs, out_chs, kernel_sizes, strides)):
        convs.append(ConvModule(in_ch,
                                out_ch,
                                kernel_size,
                                stride=stride,
                                padding=kernel_size // 2,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg))

    return convs


class DarknetBlock(nn.Module):
    def __init__(self,
                 ch,
                 num_block=1,
                 shortcut=True):
        super(DarknetBlock, self).__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList([
            nn.ModuleList([*darknet_conv((ch, ch // 2),
                          (ch // 2, ch), (1, 3), (1, 1))])
            for _ in range(num_block)
        ])

    def forward(self, x):
        for module in self.module_list:
            residual = x
            for conv in module:
                x = conv(x)
            if self.shortcut:
                x = x + residual
        return x


@VIS_ENCODERS.register_module()
class DarkNet53(nn.Module):
    def __init__(self,
                 freeze_layer=2,
                 pretrained='./data/weights/darknet.weights',
                 out_layer=(6, 8, 13)):
        super(DarkNet53, self).__init__()
        self.fp16_enabled = False
        assert isinstance(out_layer, tuple)
        self.out_layer = out_layer

        self.darknet = nn.ModuleList([
            *darknet_conv((3, 32), (32, 64), (3, 3), (1, 2)),
            DarknetBlock(64),
            *darknet_conv((64, ), (128, ), (3, ), (2, )),
            DarknetBlock(128, num_block=2),
            *darknet_conv((128, ), (256, ), (3, ), (2, )),
            DarknetBlock(256, num_block=8),
            *darknet_conv((256, ), (512, ), (3, ), (2, )),
            DarknetBlock(512, num_block=8),
            *darknet_conv((512, ), (1024, ), (3, ), (2, )),
            DarknetBlock(1024, num_block=4),
            DarknetBlock(1024, num_block=2, shortcut=False),
            *darknet_conv((1024, 512), (512, 1024), (1, 3), (1, 1))
        ])

        if pretrained is not None:
            parse_yolo_weights(self, pretrained, len(self.darknet))
            if is_main():
                logger = get_root_logger()
                logger.info(
                    f"load pretrained visual backbone from {pretrained}")

        self.do_train = False
        if freeze_layer is not None:
            freeze_params(self.darknet[:-freeze_layer])
        else:
            self.do_train = True

    @force_fp32(apply_to=('img', ))
    def forward(self, img, y):
        x = []
        for i, mod in enumerate(self.darknet):
            img = mod(img)
            if i in self.out_layer:
                x.append(img)

        if len(self.out_layer) == 1:
            return x[0]
        else:
            return x

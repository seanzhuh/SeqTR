import torch
import numpy


def freeze_params(model):
    if getattr(model, 'module', False):
        for child in model.module():
            for param in child.parameters():
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


def parse_conv_block(m, weights, offset, initflag):
    """Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): weight_file weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): weight_file weights data
    """
    conv_model = m.conv
    bn_model = m.bn
    param_length = m.bn.bias.numel()

    for pname in ['bias', 'weight', 'running_mean', 'running_var']:
        layerparam = getattr(bn_model, pname)

        if initflag:
            if pname == 'weight':
                weights = numpy.append(weights, numpy.ones(param_length))
            else:
                weights = numpy.append(weights, numpy.zeros(param_length))

        param = torch.from_numpy(
            weights[offset:offset + param_length]).view_as(layerparam)
        layerparam.data.copy_(param)
        offset += param_length

    param_length = conv_model.weight.numel()

    if initflag:
        n, c, k, _ = conv_model.weight.shape
        scale = numpy.sqrt(2 / (k * k * c))
        weights = numpy.append(
            weights, scale * numpy.random.normal(size=param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_block(m, weights, offset, initflag):
    """Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): weight_file weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): weight_file weights data
    """
    conv_model = m._modules['conv']
    param_length = conv_model.bias.numel()

    if initflag:
        weights = numpy.append(weights, numpy.zeros(param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.bias)
    conv_model.bias.data.copy_(param)
    offset += param_length

    param_length = conv_model.weight.numel()

    if initflag:
        n, c, k, _ = conv_model.weight.shape
        scale = numpy.sqrt(2 / (k * k * c))
        weights = numpy.append(
            weights, scale * numpy.random.normal(size=param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_weights(model, weights_path, end=-2):
    """Parse YOLO (darknet) pre-trained weights data onto the pytorch model
    Args:
        model : pytorch model object
        weights_path (str): path to the YOLO (darknet) pre-trained weights file
    """
    fp = open(weights_path, "rb")

    header = numpy.fromfile(fp, dtype=numpy.int32, count=5)
    weights = numpy.fromfile(fp, dtype=numpy.float32)
    fp.close()

    offset = 0
    initflag = False
    for m in model.darknet[:end]:
        if m._get_name() == 'ConvModule':
            if 'VGG' in model._get_name() and 'batch_norm' not in m._modules:
                offset, weights = parse_yolo_block(
                    m, weights, offset, initflag)
            else:
                offset, weights = parse_conv_block(
                    m, weights, offset, initflag)

        elif m._get_name() == 'DarknetBlock':
            for module in m._modules['module_list']:
                for block in module:
                    offset, weights = parse_conv_block(
                        block, weights, offset, initflag)
        else:
            assert NotImplemented

        initflag = (offset >= len(weights))


class ExponentialMovingAverage(object):
    """
        Be cautious of where ema is initialized in the sequence model initialization,
        gpu assignment and distributed dataparallel wrappers.
    """

    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            if name not in self.shadow:
                self.shadow[name] = state[name]
            else:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
        for name in self.buffer_keys:
            if self.buffer_ema:
                if name not in self.shadow:
                    self.shadow[name] = state[name]
                else:
                    self.shadow[name].copy_(
                        decay * self.shadow[name]
                        + (1 - decay) * state[name]
                    )
            else:
                if name not in self.shadow:
                    self.shadow[name] = state[name]
                else:
                    self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow, strict=True)

    def restore(self):
        self.model.load_state_dict(self.backup, strict=True)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

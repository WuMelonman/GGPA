import torch
import torch.nn as nn
from torch.autograd import Variable
import math

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    21 : [59, 55, 'M', 134, 142, 'M', 224, 204, 232, 252, 'M', 432, 360, 464, 144, 'M', 152, 176, 192, 456],
    20 : [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64],
}

class vgg(nn.Module):

    def __init__(self, depth=20, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:  # vgg19
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

        self.relu_outputs = [] 
        self.conv_gradients = []
        self.flattened_x = None 
        self.y_grad = None

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, use_hooks=True):
        if use_hooks:
            self.relu_outputs = []  #ReLU output
            self.conv_gradients = []  # conv layer output

            def conv_hook(module, input, output):
                output.register_hook(lambda grad: self.conv_gradients.append(grad))

            def relu_hook(module, input, output):
                self.relu_outputs.append(output)

            hooks = []
            for i, layer in enumerate(self.feature):
                if isinstance(layer, nn.Conv2d) and i!=0:
                    hooks.append(layer.register_forward_hook(conv_hook))
    
                if isinstance(layer, nn.MaxPool2d):
                    hooks.append(layer.register_forward_hook(relu_hook))

                if isinstance(layer, nn.ReLU):
                    if (i + 1 < len(self.feature) and not isinstance(self.feature[i + 1], nn.MaxPool2d)) :
                        hooks.append(layer.register_forward_hook(relu_hook))

        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        self.flattened_x = x.view(x.size(0), -1)
        y = self.classifier(self.flattened_x)

        if use_hooks:
            for hook in hooks:
                hook.remove()  # Remove hooks after forward pass
                
        if use_hooks:
            y.retain_grad()
            y.register_hook(lambda grad: setattr(self, 'y_grad', grad))

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40), requires_grad=True)
    y = net(x)
    print(y.data.shape)

    loss = y.mean()
    loss.backward()

    for i, fmap in enumerate(net.feature_maps):
        print(f'Feature map {i}: {fmap.size()}')
    for i, grad in enumerate(net.gradients):
        print(f'Gradient {i}: {grad.size()}')

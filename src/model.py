import yaml
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D


class Mish(object):
    def __call__(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))


class Swish(object):
    def __call__(self, x):
        return tf.nn.swish(x)  # tf.nn.leaky_relu(x, alpha=0.1)


class Conv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='SAME', groups=1):
        super(Conv, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, padding, groups=groups, use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))
        self.bn = BatchNormalization()
        self.activation = Mish()

    def call(self, x):
        return self.activation(self.bn(self.conv(x)))


class DWConv(Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DWConv, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, groups=1)  # Todo

    def call(self, x):
        return self.conv(x)


class Focus(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME'):
        super(Focus, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, padding)

    def call(self, x):
        return self.conv(tf.concat([x[..., ::2, ::2, :],
                                    x[..., 1::2, ::2, :],
                                    x[..., ::2, 1::2, :],
                                    x[..., 1::2, 1::2, :]],
                                   axis=-1))


class CrossConv(Layer):
    def __init__(self, filters, kernel_size, strides=1, groups=1, expansion=1, shortcut=False):
        super(CrossConv, self).__init__()
        units_e = int(filters * expansion)
        self.conv1 = Conv(units_e, (1, kernel_size), (1, strides))
        self.conv2 = Conv(filters, (kernel_size, 1), (strides, 1), groups=groups)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class MP(Layer):
    # Spatial pyramid pooling layer
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = MaxPool2D(pool_size=k, strides=k)

    def forward(self, x):
        return self.m(x)


class Bottleneck(Layer):
    def __init__(self, units, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(int(units * expansion), 1, 1)
        self.conv2 = Conv(units, 3, 1)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class BottleneckCSP(Layer):
    def __init__(self, units, n_layer=1, shortcut=True, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        units_e = int(units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(units, 1, 1)
        self.bn = BatchNormalization(momentum=0.03)
        self.activation = Mish()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))


class BottleneckCSP2(Layer):
    def __init__(self, units, n_layer=1, shortcut=False, expansion=0.5):
        super(BottleneckCSP2, self).__init__()
        units_e = int(units)  # hidden channels
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units, 1, 1)
        self.bn = BatchNormalization()
        self.activation = Mish()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        x1 = self.conv1(x)
        y1 = self.modules(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))


class VoVCSP(Layer):
    def __init__(self, units, expansion=0.5):
        super(VoVCSP, self).__init__()
        units_e = int(units * expansion)
        self.conv1 = Conv(units_e // 2, 3, 1)
        self.conv2 = Conv(units_e // 2, 3, 1)
        self.conv3 = Conv(units_e, 1, 1)

    def call(self, x):
        _, x1 = tf.split(x, 2, axis=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        return self.conv3(tf.concat([x1, x2], axis=-1))


class SPP(Layer):
    def __init__(self, units, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        units_e = units // 2  # Todo:
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv(units, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in kernels]  # Todo: padding check

    def call(self, x):
        x = self.conv1(x)
        return self.conv2(tf.concat([x] + [module(x) for module in self.modules], axis=-1))


class SPPCSP(Layer):
    # Cross Stage Partial Networks
    def __init__(self, units, n=1, shortcut=False, expansion=0.5, kernels=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        units_e = int(2 * units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units_e, 3, 1)
        self.conv4 = Conv(units_e, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='same') for x in kernels]
        self.conv5 = Conv(units_e, 1, 1)
        self.conv6 = Conv(units_e, 3, 1)
        self.bn = BatchNormalization()
        self.act = Mish()
        self.conv7 = Conv(units, 1, 1)

    def call(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(tf.concat([x1] + [module(x1) for module in self.modules], axis=-1)))
        y2 = self.conv2(x)
        return self.conv7(self.act(self.bn(tf.concat([y1, y2], axis=-1))))


class Upsample(Layer):
    def __init__(self, i=None, ratio=2, method='bilinear'):
        super(Upsample, self).__init__()
        self.ratio = ratio
        self.method = method

    def call(self, x):
        return tf.image.resize(x, (tf.shape(x)[1] * self.ratio, tf.shape(x)[2] * self.ratio), method=self.method)


class Concat(Layer):
    def __init__(self, dims=-1):
        super(Concat, self).__init__()
        self.dims = dims

    def call(self, x):
        return tf.concat(x, self.dims)
    
    
class Detect(Layer):
    def __init__(self, num_classes, anchors=()):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_scale = len(anchors)
        self.output_dims = self.num_classes + 5
        self.num_anchors = len(anchors[0])//2
        self.stride = np.array([8, 16, 32], np.float32)  # fixed here, modify if structure changes
        self.anchors = tf.cast(tf.reshape(anchors, [self.num_anchors, -1, 2]), tf.float32)
        self.modules = [Conv2D(self.output_dims * self.num_anchors, 1, use_bias=False) for _ in range(self.num_scale)]

    def call(self, x, training=True):
        res = []       
        for i in range(self.num_scale):  # number of scale layer, default=3
            y = self.modules[i](x[i])
            _, grid1, grid2, _ = y.shape
            y = tf.reshape(y, (-1, grid1, grid2, self.num_scale, self.output_dims))               
          
            grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x)
            grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2),tf.float32)  

            y_norm = tf.sigmoid(y)  # sigmoid for all dims
            xy, wh, conf, classes = tf.split(y_norm, (2, 2, 1, self.num_classes), axis=-1)

            pred_xy = (xy * 2. - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
            pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]
            
            out = tf.concat([pred_xy, pred_wh, conf, classes], axis=-1)
            res.append(out)
        return res
    
    
class Yolo(object):
    def __init__(self, yaml_dir):
        with open(yaml_dir) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.module_list = self.parse_model(yaml_dict)
        module = self.module_list[-1]
        if isinstance(module, Detect):
            # transfer the anchors to grid coordinator, 3 * 3 * 2
            module.anchors /= tf.reshape(module.stride, [-1, 1, 1])

    def __call__(self, img_size, name='yolo'):
        x = tf.keras.Input([img_size, img_size, 3])
        output = self.forward(x)
        return tf.keras.Model(inputs=x, outputs=output, name=name)

    def forward(self, x):
        y = []
        for module in self.module_list:
            if module.f != -1:  # if not from previous layer
                if isinstance(module.f, int):
                    x = y[module.f]
                else:
                    x = [x if j == -1 else y[j] for j in module.f]

            x = module(x)
            y.append(x)
        return x

    def parse_model(self, yaml_dict):
        anchors, nc = yaml_dict['anchors'], yaml_dict['nc']
        depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (nc + 5)

        layers = []
        # from, number, module, args
        for i, (f, number, module, args) in enumerate(yaml_dict['backbone'] + yaml_dict['head']):
            # all component is a Class, initialize here, call in self.forward
            module = eval(module) if isinstance(module, str) else module

            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings, like Detect(nc, anchors)
                except:
                    pass

            number = max(round(number * depth_multiple), 1) if number > 1 else number  # control the model scale

            if module in [Conv2D, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                c2 = args[0]
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]

                if module in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                    args.insert(1, number)
                    number = 1

            modules = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
            modules.i, modules.f = i, f
            layers.append(modules)
        return layers
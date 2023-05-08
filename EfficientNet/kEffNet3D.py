from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cai.util
import cai.models
import cai.layers

import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras.applications import imagenet_utils
import tensorflow
from copy import deepcopy
from keras import layers

import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding3D 


from DepthwiseConv3D import DepthwiseConv3D
import numpy as np

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def correct_pad3d(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 3D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 3 integers.
        kernel_size: An integer or tuple/list of 3 integers.
    # Returns
        A tuple.
    """
    #A string, either 'channels_first' or 'channels_last'
    img_dim = 1 # 2 if backend.image_data_format() == 'channels_first' else 1
    #Returns the shape of tensor or variable as a tuple of int or None entries.
    #извлечение чисел - d1, d2, d3
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 3)]
    print(input_size, backend.image_data_format())


    if isinstance(kernel_size, int):
        print('is_inst', isinstance(kernel_size, int))

        kernel_size = (kernel_size, kernel_size, kernel_size)
    
    if input_size[0] is None:
        adjust = (1, 1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2, 1 - input_size[2] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
            (correct[2] - adjust[2], correct[2]))

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)

def D6v3_2ch(): return 46
def D6v3_4ch(): return 45
def D6v3_8ch(): return 44
def D6v3_12ch(): return 42
def D6v3_16ch(): return 32
def D6v3_24ch(): return 43
def D6v3_32ch(): return 33
def D6v3_64ch(): return 34
def D6v3_128ch(): return 35


def conv3d_bn(x,
    filters,
    num_d1,
    num_d2,
    num_d3,
    padding='same',
    strides=(1, 1, 1),
    name=None,
    use_bias=False,
    activation='relu', 
    has_batch_norm=True,
    has_batch_scale=False,  
    groups=0,
    kernel_initializer="glorot_uniform",
    kernel_regularizer=None
    ):
    """Practical Conv3D wrapper.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        use_bias: True means that bias will be added,
        activation: activation function. None means no activation function. 
        has_batch_norm: True means that batch normalization is added.
        has_batch_scale: True means that scaling is added to batch norm.
        groups: number of groups in the convolution
        kernel_initializer: this is a very big open question.
        kernel_regularizer: a conservative L2 may be a good idea.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 4

    # groups parameter isn't available in older tensorflow implementations
    if (groups>1) :
        x = tensorflow.keras.layers.Conv3D(
            filters, (num_d1, num_d2, num_d3),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            groups=groups,
            name=conv_name,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(x)
    else:
        x = tensorflow.keras.layers.Conv3D(
            filters, (num_d1, num_d2, num_d3),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(x)

    if (has_batch_norm): x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, scale=has_batch_scale, name=bn_name)(x)
    if activation is not None: x = tensorflow.keras.layers.Activation(activation=activation, name=name)(x)
    return x

class CopyChannels3D(tensorflow.keras.layers.Layer):
    """
    This layer copies channels from channel_start the number of channels given in channel_count.
    """
    def __init__(self,
                 channel_start=0,
                 channel_count=1,
                 **kwargs):
        self.channel_start=channel_start
        self.channel_count=channel_count
        super(CopyChannels3D, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.channel_count)
    
    def call(self, x):
        return x[:, :, :, :, self.channel_start:(self.channel_start+self.channel_count)]
        
    def get_config(self):
        config = {
            'channel_start': self.channel_start,
            'channel_count': self.channel_count
        }
        base_config = super(CopyChannels3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def FitChannelCountTo3D(last_tensor, next_channel_count, has_interleaving=False, channel_axis=4):
        """
        Forces the number of channels to fit a specific number of channels.
        The new number of channels must be bigger than the number of input channels.
        The number of channels is fitted by concatenating copies of existing channels.
        """
        prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
        full_copies = next_channel_count // prev_layer_channel_count
        extra_channels = next_channel_count % prev_layer_channel_count
        output_copies = []
        for copy_cnt in range(full_copies):
            if copy_cnt == 0:
                output_copies.append( last_tensor )
            else:
                if has_interleaving:
                    output_copies.append( cai.layers.InterleaveChannels(step_size=((copy_cnt+1) % prev_layer_channel_count))(last_tensor) )
                else:
                    output_copies.append( last_tensor )
        if (extra_channels > 0):
            if has_interleaving:
                extra_tensor = cai.layers.InterleaveChannels(step_size=((full_copies+1) % prev_layer_channel_count))(last_tensor)
            else:
                extra_tensor = last_tensor
            output_copies.append( CopyChannels3D(0,extra_channels)(extra_tensor) )
        last_tensor = tensorflow.keras.layers.Concatenate(axis=channel_axis)( output_copies )
        return last_tensor

def kGroupConv3D(last_tensor, filters=32, channel_axis=3, channels_per_group=16, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """ 
    This is a grouped convolution wrapper that tries to force the number of input channels per group. You can give any number of filters and groups.
    You can also add this layer after any layer with any number of channels independently on any common divisor requirement.
    Follows an example:
        * 1020 input channels.
        * 16 channels per group.
        * 250 filters.
    This is how  kGroupConv2D works:
        * The first step is to make the number of "input channels" multiple of the "number of input channels per group". So, we'll add 4 channels to the input by copying the first 4 channels. The total number of channels will be 1024.
        * The number of groups will be 1024/16 = 64 groups with 16 input channels each.
        * 250 filters aren't divisible by 64 groups. 250 mod 64 = 58. To solve this problem, we'll create 2 paths. The first path deals with the integer division while the second path deals with the remainder (modulo).
                Path 1: 250 filters divided by 64 groups equals 3 filters per group (integer division). So, the first path has a grouped convolution with 64 groups, 16 input channels per group and 3 filters per group. The total number of filters in this path is 64*3 = 192.
                Path 2: the remaining 58 filters are included in this second path. There will be 58 groups with 1 filter each. The first 58 * 16 = 928 channels will be copied and made as input layer for this path.
        * Both paths are then concatenated. As a result, we'll have 192 + 58 = 250 filers or output channels!
    """
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    groups = prev_layer_channel_count // channels_per_group
    if prev_layer_channel_count % channels_per_group > 0:
        groups = groups + 1
    # the number of groups should never be bigger than the number of filters.
    if groups > filters:
        groups = filters
    local_channels_per_group = channels_per_group
    if groups > 1:
        # do we need to add more channels to make the number of imput channels multiple of channels_per_group?
        if groups * channels_per_group > prev_layer_channel_count:
            last_tensor = FitChannelCountTo3D(last_tensor, next_channel_count=groups * channels_per_group, has_interleaving=False, channel_axis=channel_axis)
        # if we have few filters, we might end needing less channels per group. This is the only case that we'll have more channels per group.
        if groups * channels_per_group < prev_layer_channel_count:
            local_channels_per_group = prev_layer_channel_count // groups
            if ( prev_layer_channel_count % groups > 0):
                local_channels_per_group = local_channels_per_group + 1
            if local_channels_per_group * groups > prev_layer_channel_count:
                last_tensor = FitChannelCountTo3D(last_tensor, next_channel_count=groups * local_channels_per_group, has_interleaving=False, channel_axis=channel_axis)
        prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
        extra_filters = filters % groups
        # should we create an additional path so we can fit the extra filters?
        if (extra_filters == 0):
            last_tensor = conv3d_bn(last_tensor, filters-extra_filters, kernel_size, kernel_size, name=name+'_m'+str(groups), activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=groups)
        else:
            root = last_tensor
            # the main path
            path1 = conv3d_bn(root, filters-extra_filters, kernel_size, kernel_size, name=name+'_p1_'+str(groups), activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=groups)
            # we'll create one group per extra filter.
            path2 = CopyChannels3D(0, local_channels_per_group * extra_filters)(root)
            path2 = conv3d_bn(path2, extra_filters, kernel_size, kernel_size, name=name+'_p2', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=extra_filters)
            # concats both paths.
            last_tensor =  tensorflow.keras.layers.Concatenate(axis=3, name=name+'_dc')([path1, path2]) # deep concat
    else:
        # deep unmodified.
        last_tensor = conv3d_bn(last_tensor, filters, kernel_size, kernel_size, name=name+'_dum', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size))
    return last_tensor, groups

def kConv3DType10(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same', never_intergroup=False):
    """
    Same as Type 2 but with a different groupings. This is also a D6 type.
    It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection.
    https://www.researchgate.net/figure/Graphical-representation-of-our-pointwise-convolution-replacement-At-the-left-a-classic_fig1_355214501
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    expansion = (filters > prev_layer_channel_count)
    # does it make sense to optimize this layer?
    if (prev_layer_channel_count > 2*min_channels_per_group) or (expansion and (prev_layer_channel_count > min_channels_per_group) ):
        output_tensor, group_count = kGroupConv3D(output_tensor, filters=filters, channel_axis=channel_axis, channels_per_group=min_channels_per_group, name=name+'_c1', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
        # should add a new convolution to mix group information?
        if (group_count>1) and (prev_layer_channel_count>=filters) and not(never_intergroup):
            compression_tensor = output_tensor
            # if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
            interleave_step = filters // min_channels_per_group
            # should we interleave?
            if interleave_step>1: output_tensor = cai.layers.InterleaveChannels(interleave_step, name=name+'_i'+str(interleave_step))(output_tensor)
            output_tensor, group_count = kGroupConv3D(output_tensor, filters=filters, channel_axis=channel_axis, channels_per_group=min_channels_per_group, name=name+'_c2', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=1, stride_size=1, padding='valid')
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_iga')
    else:
        # unmofied
        output_tensor = conv3d_bn(last_tensor, filters, kernel_size, kernel_size, name=name+'_um', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size))
    return output_tensor


def kConv3DType2(last_tensor, filters=32, channel_axis=4, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same'):
    """
    This ktype is composed by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to the input tensor and its parameters. This is the basic building block for the papers:
    https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks
    https://www.researchgate.net/publication/355214501_Grouped_Pointwise_Convolutions_Significantly_Reduces_Parameters_in_EfficientNet
    """
    output_tensor = last_tensor
    print("hohoho last_tensor  ", last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    print("yyy prev_layer_channel_count ", prev_layer_channel_count)
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//min_channels_per_group)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv3d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        if output_group_size > 1:
            output_tensor = cai.layers.InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
        if (prev_layer_channel_count >= output_channel_count):
            print('Has intergroup')
            output_tensor = conv3d_bn(output_tensor, output_channel_count, 1, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias)
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv3d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor


##kTYpe = 2?
def kConv3D(last_tensor, filters=32, channel_axis=4, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', kType = D6_32ch()):
    print("last_tensor input kconv3d    ", last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    print("last_tensor after keras backend  ", last_tensor)
    if kType == D6v3_4ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=4, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_8ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=8, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_12ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=12, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_16ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=16, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_24ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=24, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_32ch():
        return kConv3DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=32, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    if kType == D6_32ch():
        return kConv3DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)


def kPointwiseConv3D(last_tensor, filters=32, channel_axis=4, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kType=2):
    """
    Parameter efficient pointwise convolution as shown in these papers:
    https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks
    https://www.researchgate.net/publication/363413038_An_Enhanced_Scheme_for_Reducing_the_Complexity_of_Pointwise_Convolutions_in_CNNs_for_Image_Classification_Based_on_Interleaved_Grouped_Filters_without_Divisibility_Constraints
    """
    print("last tensor input kPointwiseConv3D   ", last_tensor)
    return kConv3D(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=1, stride_size=1, padding='same', kType=kType)


def kblock(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True, kType=1,
          dropout_all_blocks=False):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4

    # Expansion phase
    filters = filters_in * expand_ratio
    
    if expand_ratio != 1:
        #x = layers.Conv2D(filters, 1,
        #                 padding='same',
        #                  use_bias=False,
        #                  kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                  name=name + 'expand_conv')(inputs)
        #x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        #x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
        x = kPointwiseConv3D(last_tensor=inputs, filters=filters, channel_axis=bn_axis, name=name+'expand', activation=activation_fn, has_batch_norm=True, use_bias=False, kType=kType)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding3D(padding=correct_pad3d(backend, x, kernel_size),
                                 name=name + 'dwconv_pad3d')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = DepthwiseConv3D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling3D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, 1, filters), name=name + 'se_reshape')(se)
        #se = layers.Conv2D(filters_se, 1,
        #                   padding='same',
        #                   activation=activation_fn,
        #                   kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                   name=name + 'se_reduce')(se)
        print("kPointwiseConv3D 1v se = ", se)
        se = kPointwiseConv3D(last_tensor=se, filters=filters_se, channel_axis=bn_axis, name=name+'se_reduce', activation=activation_fn, has_batch_norm=False, use_bias=True, kType=kType)
        #se = layers.Conv2D(filters, 1,
        #                   padding='same',
        #                   activation='sigmoid',
        #                   kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                   name=name + 'se_expand')(se)
        print("kPointwiseConv3D 2v se = ", se)
        se = kPointwiseConv3D(last_tensor=se, filters=filters, channel_axis=bn_axis, name=name+'se_expand', activation='sigmoid', has_batch_norm=False, use_bias=True, kType=kType)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    #x = layers.Conv2D(filters_out, 1,
    #                  padding='same',
    #                  use_bias=False,
    #                  kernel_initializer=CONV_KERNEL_INITIALIZER,
    #                  name=name + 'project_conv')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    x = kPointwiseConv3D(last_tensor=x, filters=filters_out, channel_axis=bn_axis, name=name+'project_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if (drop_rate > 0)  and (dropout_all_blocks):
        x = layers.Dropout(drop_rate,
                noise_shape=(None, 1, 1, 1, 1),
                name=name + 'drop')(x)

    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if (drop_rate > 0)  and (not dropout_all_blocks):
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x

def GlobalAverageMaxPooling2D(previous_layer,  name=None):
    """
    Adds both global Average and Max poolings. This layers is known to speed up training.
    """
    if name is None: name='global_pool'
    return tensorflow.keras.layers.Concatenate(axis=1)([
      tensorflow.keras.layers.GlobalAveragePooling3D(name=name+'_avg')(previous_layer),
      tensorflow.keras.layers.GlobalMaxPooling3D(name=name+'_max')(previous_layer)
    ])


def kEffNet3D(
        width_coefficient,
        depth_coefficient,
        skip_stride_cnt=-1,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation_fn=swish,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        model_name='efficientnet',
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        kType=2,
        concat_paths=True,
        dropout_all_blocks=False,
        name_prefix='k_',
        **kwargs):
    #    """Instantiates the EfficientNet architecture using given scaling coefficients.
    #Optionally loads weights pre-trained on ImageNet.
    #Note that the data format convention used by the model is
    #the one specified in your Keras config at `~/.keras/keras.json`.
    #    # Arguments
    #    width_coefficient: float, scaling coefficient for network width.
    #    depth_coefficient: float, scaling coefficient for network depth.
    #    default_size: integer, default input image size.
    #    dropout_rate: float, dropout rate before final classifier layer.
    #    drop_connect_rate: float, dropout rate at skip connections.
    #    depth_divisor: integer, a unit of network width.
    #    activation_fn: activation function.
    #    blocks_args: list of dicts, parameters to construct block modules.
    #    model_name: string, model name.
    #    include_top: whether to include the fully-connected
    #        layer at the top of the network.
    #    input_tensor: optional Keras tensor
    #        (i.e. output of `layers.Input()`)
    #        to use as image input for the model.
    #    input_shape: optional shape tuple, only to be specified
    #        if `include_top` is False.
    #        It should have exactly 3 inputs channels.
    #    pooling: optional pooling mode for feature extraction
    #        when `include_top` is `False`.
    #        - `None` means that the output of the model will be
    #            the 4D tensor output of the
    #            last convolutional layer.
    #        - `avg` means that global average pooling
    #            will be applied to the output of the
    #            last convolutional layer, and thus
    #            the output of the model will be a 2D tensor.
    #        - `max` means that global max pooling will
    #            be applied.
    #    classes: optional number of classes to classify images
    #        into, only to be specified if `include_top` is True, and
    ## Returns
    #    A Keras model instance.
    ## Raises
    #    ValueError: in case of invalid input shape."""

    if input_tensor is None:
        #Input() используется для создания экземпляра тензора Keras.
        #shape: Кортеж фигур (целые числа), не включая размер пакета. Например, shape=(32,) указывает, что ожидаемыми входными данными будут пакеты 32-мерных векторов

        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):#Returns whether x is a Keras tensor.
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 4  #???

    #кол-во round фильтров
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    #кол-во повторов
    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    
    if isinstance(kType, (int)):#Позволяет проверить принадлежность экземпляра к классу
        kTypeList = [kType]
    else:
        kTypeList = kType
    
    # Build stem
    x = img_input
    print(x)
    x = layers.ZeroPadding3D(padding=correct_pad3d(backend, x, (3, 3, 3)),
                             name=name_prefix+'stem_conv_pad3d')(x)

    first_stride = 1 if skip_stride_cnt >= 0 else 2
    x = layers.Conv3D(round_filters(32), 3,
                      strides=first_stride,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name_prefix+'stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name_prefix+'stem_bn3d')(x)
    x = layers.Activation(activation_fn, name=name_prefix+'stem_activation3d')(x)

    root_layer = x
    output_layers = []
    path_cnt = 0
    for kType in kTypeList:
        x = root_layer
        blocks_args_cp = deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args_cp))
        #only the first branch can backpropagate to the input.
        #if path_cnt>0:
        #    x = keras.layers.Lambda(lambda x: tensorflow.stop_gradient(x))(x)
        for (i, args) in enumerate(blocks_args_cp):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                #should skip the stride
                if (skip_stride_cnt > i) and (j == 0) and (args['strides'] > 1):
                    args['strides'] = 1
                # The first block needs to take care of stride and filter size increase.
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                print("x = kblock before    ", x)
                x = kblock(x, activation_fn, drop_connect_rate * b / blocks,
                          name=name_prefix+'block{}{}_'.format(i + 1, chr(j + 97))+'_'+str(path_cnt), **args,
                          kType=kType, dropout_all_blocks=dropout_all_blocks)
                print(print("x = kblock after   ", x))
                b += 1
        if (len(kTypeList)>1):
            x = layers.Activation('relu', name=name_prefix+'end_relu'+'_'+str(path_cnt))(x)
        output_layers.append(x)
        path_cnt = path_cnt +1

    if (len(output_layers)==1):
        x = output_layers[0]
    else:
        if concat_paths:
            x = keras.layers.Concatenate(axis=bn_axis, name=name_prefix+'global_concat')(output_layers)
        else:
            x = keras.layers.add(output_layers, name=name_prefix+'global_add')

    x = kPointwiseConv3D(last_tensor=x, filters=round_filters(1280), channel_axis=bn_axis, name=name_prefix+'top_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling3D(name=name_prefix+'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling3D(name=name_prefix+'max_pool')(x)
    elif pooling == 'avgmax':
        x = GlobalAverageMaxPooling3D(x, name=name_prefix+'avgmax_pool')

    if include_top:
        if (dropout_rate > 0):
            x = layers.Dropout(dropout_rate, name=name_prefix+'top_dropout')(x)
        x = layers.Dense(classes,
            activation='softmax', # 'softmax'
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name=name_prefix+'probs')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    return model



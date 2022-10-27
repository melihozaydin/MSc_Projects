from keras_applications import get_submodules_from_kwargs
import keras.layers as layers
import keras.backend as backend
import keras.models as models
import keras.utils as keras_utils

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50, ResNet101, ResNet152

# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------
def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return


def get_submodules():
    return {
        "backend": backend,
        "models": models,
        "layers": layers,
        "utils": keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs,
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = "decoder_stage{}_upsampling".format(stage)
    conv1_name = "decoder_stage{}a".format(stage)
    conv2_name = "decoder_stage{}b".format(stage)
    concat_name = "decoder_stage{}_concat".format(stage)

    concat_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = "decoder_stage{}a_transpose".format(stage)
    bn_name = "decoder_stage{}a_bn".format(stage)
    relu_name = "decoder_stage{}a_relu".format(stage)
    conv_block_name = "decoder_stage{}b".format(stage)
    concat_name = "decoder_stage{}_concat".format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation("relu", name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


def Conv2dBn(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    use_batchnorm=False,
    **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop("name", None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + "_conv"

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + "_" + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + "_bn"

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def build_unet(
    backbone,
    decoder_block,
    skip_connection_layers,
    decoder_filters=(256, 128, 64, 32, 16),
    n_upsample_blocks=5,
    classes=1,
    activation="sigmoid",
    use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = [
        backbone.get_layer(name=i).output
        if isinstance(i, str)
        else backbone.get_layer(index=i).output
        for i in skip_connection_layers
    ]

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name="center_block1")(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name="center_block2")(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(
            x, skip
        )

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        name="final_conv",
    )(x)

    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet functions
# ---------------------------------------------------------------------


def Unet(
    backbone_name="vgg16",
    input_shape=(None, None, 3),
    classes=1,
    activation="sigmoid",
    weights=None,
    encoder_weights="imagenet",
    encoder_freeze=False,
    encoder_features="default",
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    **kwargs
):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if decoder_block_type == "upsampling":
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == "transpose":
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError(
            'Decoder block type should be in ("upsampling", "transpose"). '
            "Got: {}".format(decoder_block_type)
        )

    feature_layers = {
        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.
        # VGG
        "vgg16": (
            "block5_conv3",
            "block4_conv3",
            "block3_conv3",
            "block2_conv2",
            "block1_conv2",
        ),
        "vgg19": (
            "block5_conv4",
            "block4_conv4",
            "block3_conv4",
            "block2_conv2",
            "block1_conv2",
        ),
        # ResNets
        "resnet50": (
            "stage4_unit1_relu1",
            "stage3_unit1_relu1",
            "stage2_unit1_relu1",
            "relu0",
        ),
        "resnet101": (
            "stage4_unit1_relu1",
            "stage3_unit1_relu1",
            "stage2_unit1_relu1",
            "relu0",
        ),
        "resnet152": (
            "stage4_unit1_relu1",
            "stage3_unit1_relu1",
            "stage2_unit1_relu1",
            "relu0",
        ),
    }

    if backbone_name == "vgg16":
        backbone = VGG16(
            weights=encoder_weights, include_top=False, input_shape=input_shape
        )
    elif backbone_name == "ResNet50":
        backbone = ResNet50(
            weights=encoder_weights, include_top=False, input_shape=input_shape
        )
    elif backbone_name == "ResNet101":
        backbone = ResNet101(
            weights=encoder_weights, include_top=False, input_shape=input_shape
        )
    elif backbone_name == "ResNet152":
        backbone = ResNet152(
            weights=encoder_weights, include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(
            'Backbone name should be in ("vgg16", "ResNet50", "ResNet101", "ResNet152"). '
            "Got: {}".format(backbone_name)
        )

    if encoder_features == "default":
        encoder_features = feature_layers[backbone_name][:4]

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    GaussianNoise,
    SpatialDropout2D,
    concatenate,
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


def UNet(
        n_bands,
        n_classes,
        n1_filters=32,
        depth=5,
        kernel=(3, 3),
        activation="relu",
        padding="same",
        dilation_rate=1,
        input_highway=False,
        noise=0.0,
        get_logits=False,
        do_batch_norm=False,
        dropout_val=0.0,
        l2_reg_val=0.0,
        compile_kwargs=None,
):
    """UNet, as in https://arxiv.org/pdf/1505.04597.pdf

       n1_filters: set to 64 to get architecture in the paper
    """
    inputs = Input((None, None, n_bands))
    convolutions = list()
    x = inputs
    conv_kwargs = dict(
        kernel_size=kernel,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(l2_reg_val),
    )

    for i in range(depth):
        conv = Conv2D(n1_filters * 2 ** i, **conv_kwargs)(x)
        if do_batch_norm:
            conv = BatchNormalization()(conv)
        conv = SpatialDropout2D(dropout_val)(conv)
        conv = Conv2D(n1_filters * 2 ** i, **conv_kwargs)(conv)
        if do_batch_norm:
            conv = BatchNormalization()(conv)
        convolutions.append(conv)
        if i < depth - 1:
            x = MaxPooling2D(pool_size=(2, 2))(conv)

    for i in range(depth - 2, -1, -1):
        up = Conv2DTranspose(
            n1_filters * 2 ** i, (2, 2), strides=(2, 2), padding=padding
        )(conv)
        up = concatenate([up, convolutions[i]], axis=3)
        conv = Conv2D(n1_filters * 2 ** i, **conv_kwargs)(up)
        if do_batch_norm:
            conv = BatchNormalization()(conv)
        conv = Conv2D(n1_filters * 2 ** i, **conv_kwargs)(conv)
        if do_batch_norm:
            conv = BatchNormalization()(conv)

    if input_highway:
        conv = concatenate([conv, inputs])

    conv = Conv2D(n_classes, (1, 1), activation="linear")(conv)
    conv = GaussianNoise(noise)(conv)  # if noise==0, this does nothing
    if get_logits:
        output = conv
    else:
        output = Activation("sigmoid")(conv)

    model = Model(inputs=inputs, outputs=[output], name="UNet")
    if compile_kwargs:
        model.compile(**compile_kwargs)

    return model

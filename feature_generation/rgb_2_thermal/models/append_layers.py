from keras.layers import Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout, Dense, BatchNormalization, SeparableConv2D, Conv2DTranspose, Concatenate, Add, Subtract, Multiply, Input
from keras.models import Model
from keras.models import Sequential
def create_convolution_sequence(model, n_filters, kernel_size=(3,3), scale_pool=(4,4), strides=1, pooling=False, padding='valid', activation="relu"):
    x  = Conv2D(n_filters, kernel_size=kernel_size,activation=activation, padding=padding, strides=strides)(model)
    if pooling:
        x = MaxPooling2D(pool_size = scale_pool)(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    return x

def create_separarable_convolution_sequence(model, n_filters, kernel_size=(3,3), scale_pool=(4,4), pooling=False, padding='valid'):
    x  = SeparableConv2D(n_filters, kernel_size=kernel_size,activation='relu', padding=padding, depth_multiplier=1)(model)
    if pooling:
        x = MaxPooling2D(pool_size = scale_pool)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x


def create_deconvolution_sequence(model, n_filters, kernel_size=(3,3)):
    x = Conv2DTranspose(n_filters, kernel_size=kernel_size,activation='relu', padding='valid')(model)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

def create_fullyconnected_sequence(model, n_filters, function="tanh"):
    x = Dense(n_filters, activation=function)(model)
    x = Dropout(0.5)(x)
    return x

def extend_model(model, n_classes=4,activation = "relu", multiplier=2):
    input_layer = Input(shape=model.inputs[0].shape[1:])
    input_layer2 = Input(shape=model.outputs[0].shape[1:])
    model0 = create_convolution_sequence(input_layer,4*multiplier, kernel_size=(1,1), padding='same', activation=activation, pooling=True)
    model01 = create_convolution_sequence(model0,2*multiplier, kernel_size=(1,1), padding='same', scale_pool=(2,2), pooling=True, activation=activation)
    model02 = create_convolution_sequence(model01,5*multiplier, kernel_size=(3,3), scale_pool=(2,2), pooling=True, padding='same', activation=activation)

    model1 = create_convolution_sequence(input_layer,4*multiplier, kernel_size=(3,3), padding='same', activation=activation, pooling=True)
    model11 = create_convolution_sequence(model1,2*multiplier, kernel_size=(3,3), padding='same', scale_pool=(2,2), pooling=True, activation=activation)
    model12 = create_convolution_sequence(model11,5*multiplier, kernel_size=(3,3), scale_pool=(2,2), pooling=True, padding='same', activation=activation)

    model2 = create_convolution_sequence(input_layer,4*multiplier, kernel_size=(5,5), padding='same', activation=activation, pooling=True)
    model21 = create_convolution_sequence(model2,2*multiplier, kernel_size=(5,5), padding='same', scale_pool=(2,2), pooling=True, activation=activation)
    model22 = create_convolution_sequence(model21,5*multiplier, kernel_size=(3,3), scale_pool=(2,2), pooling=True, padding='same', activation=activation)

    model3 = create_convolution_sequence(input_layer,4*multiplier, kernel_size=(7,7), padding='same', activation=activation, pooling=True)
    model31 = create_convolution_sequence(model3,2*multiplier, kernel_size=(7,7), padding='same', scale_pool=(2,2), pooling=True, activation=activation)
    model32 = create_convolution_sequence(model31,5*multiplier, kernel_size=(3,3), scale_pool=(2,2), pooling=True, padding='same', activation=activation)

    model_4 = create_convolution_sequence(input_layer2,10*multiplier, kernel_size=(3,3), scale_pool=(2,2), pooling=False, padding='same', activation=activation)
    model_41 = create_convolution_sequence(model_4,2*multiplier, kernel_size=(7,7), padding='same', scale_pool=(4,4), pooling=True, activation=activation)
    model_42 = create_convolution_sequence(model_41,2*multiplier, kernel_size=(7,7), padding='same', scale_pool=(4,4), pooling=True, activation=activation)

    concatenated = Concatenate()([model12, model22, model32, model_42])#, model4])#, model4])
    flattened = Flatten() (concatenated)
    flattened = create_fullyconnected_sequence(flattened,128)
    flattened = create_fullyconnected_sequence(flattened,64)

    o_activation = "softmax"

    if n_classes == 1:
        o_activation = "sigmoid"
    output = Dense(n_classes,activation=o_activation)(flattened)
    model = Model(inputs=[input_layer,input_layer2], outputs=output)
    #model.summary()
    return model




def extend_model2(model, n_classes=4,activation = "relu", multiplier=2):
    x = create_convolution_sequence(model.output,4*multiplier, kernel_size=(1,1), padding='same', activation="relu", pooling=True)
    flattened = Flatten() (x)
    flattened = create_fullyconnected_sequence(flattened,multiplier*16)
    flattened = create_fullyconnected_sequence(flattened,multiplier*8)
    flattened = create_fullyconnected_sequence(flattened,multiplier*4)
    o_activation = "softmax"

    if n_classes == 1:
        o_activation = "sigmoid"
    output = Dense(n_classes,activation=o_activation)(flattened)
    model = Model(inputs=model.input, outputs=output)
    return model

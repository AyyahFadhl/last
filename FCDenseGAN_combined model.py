import time

import matplotlib
import tensorflow as tf
#keras.backend
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
#import keras_tuner as kt

#import matlab.engine
import GAN_Evaluation
import numpy as np
import math
import os

'''______________________________________________________ General Helping Functions___________________________________________ '''

'''Return image data from a raw PGM file as numpy array.
Format specification: http://netpbm.sourceforge.net/doc/pgm.html'''


# input: file name
# output: numpy matrix of image
def read_pgm(self, filename):
    image = matplotlib.pyplot.imread(filename, format='pgm')
    # Plotting read image
    # pyplot.imshow(image, pyplot.cm.gray)
    # pyplot.show()
    return image


''' Write Numpy array in PGM format to file.'''


# input: numpy matrix of image, file name
# output:no output
def write_pgm(self, StegoImage, filename):
    height, width = StegoImage.shape
    maxval = StegoImage.max()
    with open(filename, 'wb') as f:
        f.write(bytes('P5\n{} {}\n{}\n'.format(width, height, maxval), 'ascii'))
        # not sure if next line works universally, but seems to work on my mac
        StegoImage.tofile(f)
    return


'''
    call the  BinaryMatrixGenerator Function to obtain BinaryMatrix of Numpy float32 type,
    input the BinaryMatrix and probability map element-wise to the activation function using following equations
    m_(i,j)= -0.5 tanh(? (p_(i,j)-2n_(i,j)))+ 0.5 tanh(? (p_(i,j)-2?(1-n?_(i,j)))).
    tanh(x)=(e^x- e^(-x))/(e^x+ e^(-x) )'''


# input : Probability map p. (Numpy type, float32)
# output: Modification map m. (Numpy type,  float32)
def TernaryEmbeddingSimulator(p):
    # Call BinaryMatrixGenerator() to obtain n
    n = BinaryMatrixGenerator(p.shape)
    # Creating and computing  modificarionMap numpy array.
    # m = np.empty(p.shape, np.float32)
    # 1000 is the scalling factor to simulate stair case function
    m = -0.5 * tf.keras.backend.tanh(1000 * (p - 2 * n)) + 0.5 * tf.keras.backend.tanh(1000 * (p - 2 * (1 - n)))
    # Compute modificationMap
    # 1000 is the scalling factor to simulate stair case function

    '''
    for batch in range(p.shape[0]):
        for i in range(p.shape[1]):
            for j in range(p.shape[2]):
                m[batch,i, j,0] = -0.5 * tf.keras.backend.tanh(1000 * (p[batch,i, j,0] - 2 * n[batch,i, j,0])) + 0.5 * tf.keras.backend.tanh(1000 * (p[batch,i, j,0] - 2 * (1 - n[batch,i, j,0])))
    '''

    # print modificationMap
    # np.set_printoptions(threshold=np.inf)
    # print(m)

    # plotting modification matrix
    # plt.matshow(m)
    # plt.show()
    return m


'''
Produce a randomly generated matrix (CoverImage H,CoverImage Width) of  zeros and once (Uniform distribution) using Numpy random (PRNG)'''


# input: ImageSize
# output: Randomly generated matrix of zeros and once. (Numpy type, float32)
def BinaryMatrixGenerator(ProbabilityMapSize):
    # initializing the seed for PRNG based on system (Truly RNG)
    np.random.seed(list(os.urandom(1)))

    # generating  binary numpy array of size 512x512
    #Random_matrix = np.random.randint(low=0, high=2, size=ProbabilityMapSize)
    Random_matrix = np.random.randint(low=0, high=2, size=(256,256,1))
    # plotting matrix
    # cmap = ListedColormap(['k', 'w'])
    # cax = plt.matshow(Random_matrix, cmap=cmap)
    # plt.show()

    # Casting to float32
    Random_matrix = np.cast['float32'](Random_matrix)
    return Random_matrix


'''Calcualte exponential of given value x
# input: x
# output: exponential value of x
def tanh(x):
    # print (math.exp(x))
    return math.exp(x)'''
#############################################################################################################################################
'''_____________________________________FCDenseNet Generator Function and Its helping Functions______________________________________________________________'''
''' Generator  generated probability map , which is used by the above two functions to generate stego image, using  FCDense_Net Architecture
# input : Cover image , and Target Capacity
# output: Probability map , and Stego images
'''

'''Defining the FCDenseNet Generator Architecture'''
# Input: no input
# Output: FCDenseNet model

'''Defining the FCDenseNet Generator Architecture'''


# Input: no input
# Output: FCDenseNet model

def FCDenseNet(input_shape=(256, 256, 1)
                   , n_filters_first_conv=48
                   , Filter_size_first_conv=3
                   , n_path_layer=4
                   , growth_rate=12
                   , n_layers_per_block=5
                   , dropout_p=0.2
                   , Kernal_intializer='he_uniform'
                   , Activation='relu'
                   , pooling=tf.keras.layers.MaxPooling2D((2, 2))
                   , DB_FilterSize=3
                   , TD_FilterSize=1
                   , TU_filterSize=3
                   , n_filters_last_conv=1
                   , Filter_size_last_conv=3
                   ):

        """
        def BN_RelU_Conv(inputs, n_filters,filter_size=3,dropout_p=0.2,Activation='relu',Kernal_initializer='he_uniform'):
        def TransitionUp( inputs,skip_connection,n_filters,kernal_size=3,Kernal_intializer='he_uniform'):
        def TransitionDown(inputs,n_filters,filter_size=1,dropout_p=0.2,Activation='relu',Kernal_initializer='he_uniform',pooling=tf.keras.layers.MaxPooling2D((2,2))):
            This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
            The network consist of a downsampling path, where dense blocks and transition down are applied,
            followed by an upsampling path where transition up and dense blocks are applied.
            Skip connections are used between the downsampling path and the upsampling path
            Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.
            ######################
            # Parameters #
            ######################
            :param n_filters_first_conv: number of filters for the first convolution applied
            :param n_Path_layers: number of transition down = number of transition up
            :param growth_rate: number of new feature maps created by each layer in a dense block (number of filter)
            :param n_layers_per_block: number of layers per block. Can be an int (all blocks have same number of layers) or a list of size (2 * n_path_layer + 1) (transition up layers+transition down layer+ bottleneck layers)
            :param dropout_p: dropout rate applied after each convolution (0. for not using)
        """

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_path_layer + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block for i in range(2 * n_path_layer + 1)]
        else:
            raise ValueError

        ####################
        # First Convolution #
        ####################

        # input_shape = (256, 256, 1)
        inputs = tf.keras.Input(shape=input_shape)

        output = tf.keras.layers.Conv2D(filters=n_filters_first_conv, kernel_size=n_filters_first_conv, padding='same',
                                        kernel_initializer=Kernal_intializer)(inputs)
        # n_filters = n_filters_first_conv

        #print('First Convolution Output', output.shape)
        # n_filter=n_filters_first_conv

        #print("_______________________________________________________")

        #####################
        # DownSamplingPath:
        # 1.DenseBlock
        # 2.Transition Down
        #####################

        skip_connection_list = []

        # Downsampling Path
        for i in range(n_path_layer):  # 0,1,2,...(n_path_layer-1)
            # DenseBlock#
            for j in range(n_layers_per_block[i]):
                # layer
                l = BN_RelU_Conv(inputs=output, n_filters=growth_rate, filter_size=DB_FilterSize, dropout_p=dropout_p,
                                 Activation=Activation, Kernal_initializer=Kernal_intializer)
                output = tf.keras.layers.Concatenate()([output, l])
                # no need to do
                # n_filter= n_filter+growth_rate
            #print('DenseBlock', i, output.shape)
            # at the end of each DenseBlock, add the output of the desnseblock to skip_connection_list to during up sampling Path.
            skip_connection_list.append(output)

            # TransitionDown# output.shape[-1]
            output = TransitionDown(output, n_filters=output.shape[-1], filter_size=TD_FilterSize, dropout_p=dropout_p,
                                    Activation=Activation, Kernal_initializer=Kernal_intializer, pooling=pooling)
            # TD_Denseblock shape
            #print('DenseBlock_TD', i, output.shape)

        skip_connection_list = skip_connection_list[
                               ::-1]  # this reverse the list, cause later on we can concatenate i with i when up sampling, starting from beggining

        #print("_______________________________________________________")

        ###################
        #  Bottleneck:
        #  1. DenseBlock
        ##################

        block_to_upsample = []

        # DenseBlock#
        for j in range(n_layers_per_block[n_path_layer]):  # n_path_layer
            # layer
            l = BN_RelU_Conv(output, n_filters=growth_rate, filter_size=DB_FilterSize, dropout_p=dropout_p,
                             Activation=Activation, Kernal_initializer=Kernal_intializer)
            block_to_upsample.append(l)
            output = tf.keras.layers.Concatenate()([output, l])
        # DB shape Bottleneck
        #print('Bottleneck_DenseBlock', i, output.shape)
        #print("_______________________________________________________")

        #######################
        #  UpsamplingPath :
        #  1.TransitionUp
        #  2.DenseBlock
        #######################

        for i in range(n_path_layer):
            # Transition Up# (up sampling +concatenation with skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[
                n_path_layer + i]  # or l.shape[-1] n_filters_keep # need to understand how the Computation of  n_of_filter for TU is perfromed.
            output = TransitionUp(l, skip_connection_list[i], n_filters=n_filters_keep, kernal_size=TU_filterSize,
                                  Kernal_intializer=Kernal_intializer)  # block to upsample is output of previous Dense block without concatenation with input of DenseBlock
            #print('TransitionUp', i, output.shape)

            # DesnseBlock#
            block_to_upsample = []
            for j in range(n_layers_per_block[n_path_layer + i + 1]):
                # Layer
                l = BN_RelU_Conv(output, n_filters=growth_rate, filter_size=DB_FilterSize, dropout_p=dropout_p,
                                 Activation=Activation, Kernal_initializer=Kernal_intializer)
                # block_to_upsample.append(l)
                output = tf.keras.layers.Concatenate()([output, l])  # DenseBlock
            # TU_DB shape
            #print('TU_DenseBlock', i, output.shape)

        #print("_______________________________________________________")

        #####################
        # Last Convolution #
        ####################

        output = tf.keras.layers.Conv2D(filters=n_filters_last_conv, kernel_size=Filter_size_last_conv, padding='same',
                                        kernel_initializer=Kernal_intializer)(output)
        #print('Last Convolution Output', output.shape)

        # Preprocing the output to make sure that the probability between 0 and 0.5.
        output = tf.keras.activations.sigmoid(output)
        output = tf.keras.layers.ReLU()(output * 0.5)

        model = tf.keras.Model(inputs=[inputs], outputs=[output])

        print("Model FCDenseNet Built")
        return model,output

'''Apply successivly BatchNormalization, ReLu, Convolution and Dropout (if dropout_p > 0) on the inputs'''

def BN_RelU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2, Activation='relu',
                     Kernal_initializer='he_uniform'):
        outputs = tf.keras.layers.BatchNormalization()(inputs)
        outputs = tf.keras.layers.Activation(Activation)(outputs)
        outputs = tf.keras.layers.Conv2D(n_filters, filter_size, padding='same', kernel_initializer=Kernal_initializer)(
            outputs)
        if dropout_p != 0.0:
            outputs = tf.keras.layers.Dropout(dropout_p)(outputs)
        return outputs

'''Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2'''

def TransitionDown(inputs, n_filters, filter_size=1, dropout_p=0.2, Activation='relu',
                       Kernal_initializer='he_uniform', pooling=tf.keras.layers.MaxPooling2D((2, 2))):
        outputs = BN_RelU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p, Activation=Activation,
                               Kernal_initializer=Kernal_initializer)
        outputs = pooling(outputs)
        return outputs

'''Perform upsampling on block to upsamle by factor 2, and concatenates it with the skip connection'''

def TransitionUp(inputs, skip_connection, n_filters, kernal_size=3, Kernal_intializer='he_uniform'):
        # outputs= tf.keras.layers.Concatenate()(block_to_upsample)
        outputs = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernal_size, strides=(2, 2),
                                                  padding='same', kernel_initializer=Kernal_intializer)(inputs)
        outputs = tf.keras.layers.Concatenate()([outputs, skip_connection])
        return outputs


'''Calculate How well the  FCDense generator does the job using equation 3.8 excluding 3.10'''

# input: Softmaxoutput(class probabilities) Pv,Actual Value,Probability Map (tensor float 32),Target Payload (float)
# output: Loss of UT Generator
def FCDenseNetGeneratorLossFunction(p, TargetPayload, Av, Pv):
        # Calculate Capacity using probability Map.
        lc = 0
        Capacity = 0
        A = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        B = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        C = (1 - p) * (tf.math.log(1 - p) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        Capacity = tf.reduce_sum(- A - B - C)
        # Calculating loss in Capacity
        lc = Capacity - (p.shape[0] * p.shape[1] * p.shape[2] * TargetPayload)
        # Calculating Adverserial Loss
        ld = DiscriminatorLossFunction(Av, Pv)
        # Calcualting Generator Loss with alph of 1, and beta of 10^-7
        lg = ld + math.pow(10, -7) * tf.math.pow(lc, tf.constant(2, dtype=p.dtype))

        '''
        # Comparing the results of the two approaches (Tensor, loop)
        lc=0
        for batch in range(p.shape[0]):
            Capacity=0
            for i in range(p.shape[1]):
                for j in range(p.shape[2]):
                    # Probability for changing pixel x[i,j] to x[i,j]+1 generated by Generator
                    A = (p[batch, i, j, 0] / 2) * (math.log(p[batch, i, j, 0] / 2,2))
                    # Probability for changing pixel x[i,j] to x[i,j]-1 generated by Generator
                    B = (p[batch,i, j, 0] / 2) * (math.log(p[batch,i, j, 0] / 2,2))
                    # Probability for Leaving pixel x[i,j] unchangable generated by generator
                    C = (1 - p[batch,i, j, 0]) * (math.log(1 - p[batch,i, j, 0],2))
                    Capacity = Capacity + (- A - B - C)
            #Calcualting Capacity
            lc= lc + (Capacity - p.shape[1] * p.shape[2] * TargetPayload)
        #Calculating Adverserial Loss
        ld= DiscriminatorLossFunction(Av,Pv)
        #Calcualting Generator Loss with alph of 1, and beta of 10^-7
        lg= ld + math.pow(10,-7)* tf.math.pow(lc,tf.constant(2, dtype=p.dtype))
        '''
        return lg


def Wrapper(p, TargetPayload):
    def Custom_loss(y_true, y_pred):
        # Calculating Adverserial Loss
        ld = cross_entropy(y_true, y_pred)
        # Calculating loss in Capacity
        # Calculate Capacity using probability Map.
        lc = 0
        Capacity = 0
        A = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        B = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        C = (1 - p) * (tf.math.log(1 - p) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        Capacity = tf.reduce_sum(- A - B - C)
        # Finding the loss in capacity
        #lc = Capacity - (p.shape[0] * p.shape[1] * p.shape[2] * TargetPayload)
        lc= Capacity - tf.math.multiply(float(p.shape[1]),tf.math.multiply(float(tf.math.multiply(p.shape[2], p.shape[3])), TargetPayload))
        # Calcualting Generator Loss with alph of 1, and beta of 10^-7
        lg = ld + math.pow(10, -7) * tf.math.pow(lc, tf.constant(2, dtype=p.dtype))
        loss=lg
        return loss
    return Custom_loss

def Wrapper2(p, TargetPayload):
    def Custom_loss(y_true, y_pred):
        # Calculating Adverserial Loss
        ld = cross_entropy(y_true, y_pred)
        # Calculating loss in Capacity
        # Calculate Capacity using probability Map.
        lc = 0
        Capacity = 0
        A = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        B = (p / 2) * (tf.math.log(p / 2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        C = (1 - p) * (tf.math.log(1 - p) / tf.math.log(tf.constant(2, dtype=p.dtype)))
        Capacity = tf.reduce_sum(- A - B - C)
        # Finding the loss in capacity
        lc = Capacity - tf.math.multiply(p.shape[0],tf.math.multiply(tf.math.multiply(p.shape[1], p.shape[2]),TargetPayload))
        # Calcualting Generator Loss with alph of 1, and beta of 10^-7
        lg = ld + math.pow(10, -7) * tf.math.pow(lc, tf.constant(2, dtype=p.dtype))
        loss=lg
        return loss
    return Custom_loss


##################################################################################################################################################################
''' _______________________GBRAS-Net Discriminator Function and Its helping Functions_____________________________'''

''' Discriminator predict the label for input image , cover or stego (generated using FCDense_Net)
# input : image , which may be cover or stego
# output: prediction label
'''

'''Defining  the GBRAS_Net Discriminator Architecture'''

# Input: no input
# Output: GBRAS_Net model
def GBRAS_NetDiscriminator():
        tf.keras.backend.clear_session()
        # Inputs
        inputs = tf.keras.Input(shape=(256, 256, 1), name="input_1")
        # Layer 1 (Preprocessing)
        layers = tf.keras.layers.Conv2D(30, (5, 5), weights=[np.load('30SRM.npy'), np.ones(30)], strides=(1, 1),
                                        padding='same', trainable=False, activation=Tanh3, use_bias=True)(inputs)
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                     trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                     renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 2
        layers = tf.keras.layers.DepthwiseConv2D(1)(layers1)
        layers = tf.keras.layers.SeparableConv2D(30, (3, 3), padding='same', activation="elu", depth_multiplier=3)(
            layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 3
        layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
        layers = tf.keras.layers.SeparableConv2D(30, (3, 3), padding='same', activation="elu", depth_multiplier=3)(
            layers)
        layers2 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                     trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                     renorm_momentum=0.4, adjustment=None)(layers)
        skip1 = tf.keras.layers.Add()([layers1, layers2])
        # Layer 4
        layers = tf.keras.layers.Conv2D(30, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(skip1)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 5
        layers = tf.keras.layers.Conv2D(30, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 6
        layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
        # Layer 7
        layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                     trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                     renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 8
        layers = tf.keras.layers.DepthwiseConv2D(1)(layers3)
        layers = tf.keras.layers.SeparableConv2D(60, (3, 3), padding='same', activation="elu", depth_multiplier=3)(
            layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 9
        layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
        layers = tf.keras.layers.SeparableConv2D(60, (3, 3), padding='same', activation="elu", depth_multiplier=3)(
            layers)
        layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                     trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                     renorm_momentum=0.4, adjustment=None)(layers)
        skip2 = tf.keras.layers.Add()([layers3, layers4])
        # Layer 10
        layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(skip2)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 11
        layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
        # Layer 12
        layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 13
        layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
        # Layer 14
        layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 15
        layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
        # Layer 16
        layers = tf.keras.layers.Conv2D(30, (1, 1), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 17
        layers = tf.keras.layers.Conv2D(2, (1, 1), strides=(1, 1), activation="elu", padding='same',
                                        kernel_initializer='glorot_uniform')(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False,
                                                    trainable=True, fused=None, renorm=False, renorm_clipping=None,
                                                    renorm_momentum=0.4, adjustment=None)(layers)
        # Layer 18
        layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
        # Layer 19
        predictions = tf.keras.layers.Softmax(axis=1)(layers)
        # Model generation
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        print("Model GBRAS-Net Built")
        return model

'''Helping function for computing Tanh3 Activation function '''

# Input: X matrix
# Output: Computed matrix
def Tanh3(x):
        return (tf.keras.backend.tanh(x) * 3)


'''Calculate How well the discriminator does the job using following equation 3.7'''

# input: Pair of predicted Value matrix shape(1,1,2) and Actual Value matrix shape= (1,1,2)
# output: Loss of Discriminator
def DiscriminatorLossFunction(y_true, y_pred):
        ld = cross_entropy(y_true, y_pred)
        # Will never get log zero because of Softamx function output.
        # loss of Discriminator (lD)
        # ld = 0
        # for i in range(2):
        #   ld = ld + Av[0, 0, i] * math.log(Pv[0, 0, i])
        # ld = -ld
        return ld



######################################################################################################################################################
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def GAN(input_shape=(256, 256, 1)
                   ,payload=0.4
                   , n_filters_first_conv=48
                   , Filter_size_first_conv=3
                   , n_path_layer=4
                   , growth_rate=12
                   , n_layers_per_block=5
                   , dropout_p=0.2
                   , Kernal_intializer='he_uniform'
                   , Activation='relu'
                   , pooling=tf.keras.layers.MaxPooling2D((2, 2))
                   , DB_FilterSize=3
                   , TD_FilterSize=1
                   , TU_filterSize=3
                   , n_filters_last_conv=1
                   , Filter_size_last_conv=3):


    global discriminator, generator,GAN

    # Discriminator
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,                                                   decay=0.0)
    discriminator = GBRAS_NetDiscriminator()
    discriminator.compile(loss=DiscriminatorLossFunction,optimizer=discriminator_optimizer )

    #Generator
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    generator, probabilityMap =FCDenseNet(n_filters_first_conv=48
                   , Filter_size_first_conv=3
                   , n_path_layer=4
                   , growth_rate=12
                   , n_layers_per_block=5
                   , dropout_p=0.2
                   , Kernal_intializer='he_uniform'
                   , Activation='relu'
                   , pooling=tf.keras.layers.MaxPooling2D((2, 2))
                   , DB_FilterSize=3
                   , TD_FilterSize=1
                   , TU_filterSize=3
                   , n_filters_last_conv=1
                   , Filter_size_last_conv=3)
    #generator.compile(loss=Wrapper(generator.layers[-1].output,payload),optimizer=generator_optimizer)
    generator.compile(loss=DiscriminatorLossFunction, optimizer=generator_optimizer)
    CoverImage=tf.keras.layers.Input((256,256,1))
    ProbabilityMap = generator(CoverImage)
    # plt.imshow(tf.reshape(Generated_ProbabilityMap, (256,256)))
    # plt.show()
    ModificationMap = TernaryEmbeddingSimulator(ProbabilityMap)
    # plt.imshow(tf.reshape(ModificationMap, (256, 256)))
    # plt.show()
    StegoImage = tf.math.add(CoverImage, ModificationMap)
    discriminator.trainable = False
    prediction = discriminator(StegoImage)
    GAN=  tf.keras.Model(inputs=CoverImage, outputs=prediction)
    GAN.compile(loss=DiscriminatorLossFunction, optimizer=generator_optimizer)
    #GAN.compile(loss=Wrapper(ProbabilityMap, payload), optimizer=generator_optimizer)

    print("GAN Model has been Built ")

    return GAN


def Train(epoch=1, batch_size=16, save_interval=1):
    MasterPath = os.path.dirname(os.path.abspath(__file__))
    # load Training  and Testing Data
    TrainX = np.load(os.path.join(MasterPath, r'Dataset_GAN_try\TrainingData\TrainingData256\TrainX.npy')).astype('float32')  # shape(NumberofImages,256,256,1)
    # TrainY = np.load(os.path.join(MasterPath, r'Dataset\TrainingData\TrainingData256\TrainY.npy'))#shape (NumberofImages,2)
    TestX = np.load(os.path.join(MasterPath, r'Dataset_GAN_try\TestingData\TestingData256\TestX.npy')).astype('float32')  # shape(NumberofImages,256,256,1)
    # TestY = np.load(os.path.join(MasterPath, r'Dataset\TestingData\TestingData256\TestY.npy')) # shape(NumberofImages,2)
    # Normalize Training images X between -1 and 1.
    TrainX = (TrainX - 127.5) / 127.5
    TestX = (TestX - 127.5) / 127.5

    #half_batch = int(batch_size / 2)

    for epoch in range(epoch):
        print(f"epoch:{epoch}")
        start = time.time()
        #Shuffel Training set, and divide them into min batches
        np.random.shuffle(TrainX)
        mini_Batch_TrainX = np.array_split(TrainX, math.floor(TrainX.shape[0] / batch_size))
        # Preparing the lists, which will be plotted by GAN_Evaluation
        G_Loss = []
        D_Loss = []
        iteration=1
        for CoverImage_batch in mini_Batch_TrainX:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            print("Train Discriminator on Batch of Data")
            #Generate a batch of Fake(Stego Images)
            ProbabilityMap = generator.predict(CoverImage_batch)
            ModificationMap = TernaryEmbeddingSimulator(ProbabilityMap)
            StegoImage_batch = tf.math.add(CoverImage_batch, ModificationMap)


            #Training the discriminator in Real(Cover), and Fake(Stego) Images
            CoverLossD =discriminator.train_on_batch(CoverImage_batch,tf.convert_to_tensor(np.full((CoverImage_batch.shape[0], 2), (1, 0))))
            StegoLossD=discriminator.train_on_batch(StegoImage_batch, tf.convert_to_tensor(np.full((StegoImage_batch.shape[0], 2), (0, 1))))

            DiscriminaotorLoss = 0.5 * tf.add(CoverLossD,StegoLossD)
            D_Loss.append(DiscriminaotorLoss)

            # ---------------------
            #  Train GAN(Generator while discriminator is not being trained)
            # ---------------------
            print("Train GAN on batch of Data")
            GeneratorLoss = GAN.train_on_batch(CoverImage_batch,tf.convert_to_tensor(np.full((CoverImage_batch.shape[0], 2), (1, 0))))
            G_Loss.append(GeneratorLoss)
            iteration = iteration + 1

        epochTrainingTime = time.time() - start
        # 1. Visualizing Training Loss Function of generator and Discriminator after each epoch .
        lossGraph(epoch, G_Loss, D_Loss)
        # Evaluating Generator performance After each epoch Using Testing Data.
        # 2. Qualitative Evaluation
        GAN_Evaluation.QualitativeEvaluation(epoch, generator, TestX)
        # 3. Quantitative Evaluation
        GAN_Evaluation.QuantitativeEvaluation(epoch, generator, TestX, epochTrainingTime)
        print('Time for epoch {} is {} sec'.format(epoch + 1, epochTrainingTime))
        # saving after each 10 epoch
        if (epoch % 10 == 0):
            generator.save("FCDenseNet_GAN/saved_G_models0.4/generator_model_{}.hd5".format(epoch))
            discriminator.save("FCDenseNet_GAN/Saved_D_models0.4/discriminator_model_{}.hd5".format(epoch))

def lossGraph(epoch, G_Loss, D_Loss):
    os.makedirs(os.path.join(MasterPath, r'TraininglossGraphs'), exist_ok=True)
    plt.plot(G_Loss, 'r-', label='G_Loss')
    plt.plot(D_Loss, 'k-', label='D_Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(MasterPath, f'TraininglossGraphs\epoch {epoch + 1}.png'))
    plt.close()
    return

MasterPath = os.path.dirname(os.path.abspath(__file__))
#FCDenseNet_GAN=GAN()
#FCDenseNet_GAN.summary()
#Train()

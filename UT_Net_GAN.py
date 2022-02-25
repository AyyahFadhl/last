# Importing
import matplotlib

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tensorflow as tf

#import matlab.engine
import GAN_Evaluation
import numpy as np
import math
import os
import time
import re




'''______________________________________________________ General Helping Functions___________________________________________ '''
'''Return image data from a raw PGM file as numpy array.
Format specification: http://netpbm.sourceforge.net/doc/pgm.html'''
#input: file name
#output: numpy matrix of image
def read_pgm(filename):
    image = matplotlib.pyplot.imread(filename,format='pgm')
    #Plotting read image
    #pyplot.imshow(image, pyplot.cm.gray)
    #pyplot.show()
    return image


''' Write Numpy array in PGM format to file.'''
#input: numpy matrix of image, file name
#output:no output
def write_pgm(StegoImage, filename):
    height, width = StegoImage.shape
    maxval = StegoImage.max()
    with open(filename, 'wb') as f:
        f.write(bytes('P5\n{} {}\n{}\n'.format(width, height, maxval), 'ascii'))
        # not sure if next line works universally, but seems to work on my mac
        StegoImage.tofile(f)
    return

'''
Produce a randomly generated matrix (CoverImage H,CoverImage Width) of  zeros and once (Uniform distribution) using Numpy random (PRNG)'''
# input: ImageSize
# output: Randomly generated matrix of zeros and once. (Numpy type, float32)
def BinaryMatrixGenerator(ProbabilityMapSize):
    # initializing the seed for PRNG based on system (Truly RNG)
    np.random.seed(list(os.urandom(1)))

    # generating  binary numpy array of size 512x512
    Random_matrix = np.random.randint(low=0, high=2, size=ProbabilityMapSize)

    # plotting matrix
    #cmap = ListedColormap(['k', 'w'])
    #cax = plt.matshow(Random_matrix, cmap=cmap)
    #plt.show()

    # Casting to float32
    Random_matrix = np.cast['float32'](Random_matrix)
    return Random_matrix

'''Calcualte exponential of given value x'''
#input: x
#output: exponential value of x
def tanh(x):
    #print (math.exp(x))
    return math.exp(x)


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
    #m = np.empty(p.shape, np.float32)
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



'''_______________________UT_Net Generator Function and Its helping Functions___________________________'''

'''
Generate the stego image based on the target capcity with the help of first two functions. UT_Net Architecture.
Note: it has been coded as if the input is downsampled before going to layer1
#input :Target capacity, cover image
#output : stego image, probability map
def Generator():
    return'''

'''Defining the UT_Net Generator Architecture'''
#Input: no input
#Output: UT_Net model
def UT_Net():
    # preparing the input
    input_shape = (256, 256, 1)
    input = tf.keras.layers.Input(input_shape)

    # input
    #print(input .shape)
    Processedinput = tf.keras.layers.Conv2D(1, 1, input_shape=input_shape)(input )
    Processedinput = tf.keras.layers.BatchNormalization()(Processedinput  )
    Processedinput = tf.keras.layers.LeakyReLU(alpha=0.02)(Processedinput )
    #print(Processedinput  .shape)
    Processedinput = tf.keras.layers.MaxPooling2D((2, 2))(Processedinput )

    # Concatenation PATH Encoder.
    ConVout1, Pout1 = Encoder(Processedinput , 16)
    #print("layer 1", ConVout1.shape)
    ConVout2, Pout2 = Encoder(Pout1, 32)
    #print("layer 2", ConVout2.shape)
    ConVout3, Pout3 = Encoder(Pout2, 64)
    #print("layer 3", ConVout3.shape)
    ConVout4, Pout4 = Encoder(Pout3, 128)
    #print("layer 4", ConVout4.shape)
    ConVout5, Pout5 = Encoder(Pout4, 128)
    #print("layer 5", ConVout5.shape)
    ConVout6, Pout6 = Encoder(Pout5, 128)
    #print("layer 6", ConVout6.shape)
    ConVout7, Pout7 = Encoder(Pout6, 128)
    #print("layer 7", ConVout7.shape)

    # Base (Without down-sampling or up-sampling )
    ConVout8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(Pout7)
    ConVout8 = tf.keras.layers.BatchNormalization()(ConVout8)
    ConVout8 = tf.keras.layers.LeakyReLU(alpha=0.2)(ConVout8)
    #print("layer 8", ConVout8.shape)

    # EXPANSTION PATH Decoder.
    Con9, U9 = Decoder(ConVout8, ConVout7, 128)
    #print("layer 9", Con9.shape)
    Con10, U10 = Decoder(Con9, ConVout6, 128)
    #print("layer15", Con10.shape)
    Con11, ConVTout11 = Decoder(Con10, ConVout5, 128)
    #print("layer11", Con11.shape)
    Con12, ConVTout12 = Decoder(Con11, ConVout4, 128)
    #print("layer12", Con12.shape)
    Con13, ConVTout13 = Decoder(Con12, ConVout3, 64)
    #print("layer13", Con13.shape)
    Con14, ConVTout14 = Decoder(Con13, ConVout2, 32)
    #print("layer14", Con14.shape)
    Con15, ConVTout15 = Decoder(Con14, ConVout1, 16)
    #print("layer15", Con15.shape)

    # Output preprocessing
    # layer16
    ConVTout16 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=2, padding='same')(ConVTout15)
    ConVTout16 = tf.keras.layers.BatchNormalization()(ConVTout16)
    ConVTout16 = tf.keras.layers.ReLU()(ConVTout16)
    #print("layer 16", ConVTout16.shape)
    # Layer17
    output = tf.keras.activations.sigmoid(ConVTout16)
    output = tf.keras.layers.ReLU()(output * 0.5)
    #print("layer17", output.shape)

    # Model
    model = tf.keras.Model(inputs=[input], outputs=[output])
    print("Model UT_Net built")
    #model.summary()
    return model

'''Helping function for UT_Net Generator function '''
#input: input image or feature map from previous layer, and number of filters
#output: output of convolution, output of downsampling operation
def Encoder(inp, f):
    ConVout = tf.keras.layers.Conv2D(f, (3, 3), padding='same')(inp)
    ConVout = tf.keras.layers.BatchNormalization()(ConVout)
    ConVout = tf.keras.layers.LeakyReLU(alpha=0.2)(ConVout)
    Pout = tf.keras.layers.MaxPooling2D((2, 2))(ConVout)  # for down sampling
    return ConVout, Pout

'''Helping function for UT_Net Generator function '''
#input:feature map from previous layer, convolution output of mirriored layer,and number of filters
#output: output of deconvoltion(upsampling), output of concatenation
def Decoder(inp, C, f):
    U = tf.keras.layers.Conv2DTranspose(f, (5, 5), strides=2, padding='same')(inp)
    U = tf.keras.layers.BatchNormalization()(U)
    U = tf.keras.layers.ReLU()(U)
    Concatenation = tf.keras.layers.Concatenate()([U, C])
    return Concatenation, U

'''Calculate How well the  UT Net generator does the job using equation 3.8 excluding 3.10'''
#input: Softmaxoutput(class probabilities) Pv,Actual Value,Probability Map (tensor float 32),Target Payload (float)
#output: Loss of UT Generator
def UT_NetGeneratorLossFunction(p,TargetPayload,Av,Pv):
    #Calculate Capacity using probability Map.
    lc=0
    Capacity = 0
    A=(p / 2) * (tf.math.log(p/2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
    B=(p / 2) * (tf.math.log(p/2) / tf.math.log(tf.constant(2, dtype=p.dtype)))
    C= (1 - p) * (tf.math.log(1 - p) / tf.math.log(tf.constant(2, dtype=p.dtype)))
    Capacity = tf.reduce_sum(- A - B - C)
    #Calculating loss in Capacity
    lc= Capacity - (p.shape[0] *p.shape[1] * p.shape[2] * TargetPayload)
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


''' _______________________GBRAS-Net Discriminator Function and Its helping Functions_____________________________'''

''' predict the label for input image , cover or stego (generated using UT_Net)
# input : image , which may be cover or stego
# output: prediction label
def Discriminator():
    return
'''

'''Defining  the GBRAS_Net Discriminator Architecture'''
#Input: no input
#Output: GBRAS_Net model
def GBRAS_NetDiscriminator():
    tf.keras.backend.clear_session()
    #Inputs
    inputs = tf.keras.Input(shape=(256,256,1), name="input_1")
    #Layer 1 (Preprocessing)
    layers = tf.keras.layers.Conv2D(30, (5,5), weights=[np.load('30SRM.npy'),np.ones(30)], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 2
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers1)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 3
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers)
    layers2 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip1 =   tf.keras.layers.Add()([layers1, layers2])
    #Layer 4
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip1)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 5
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 6
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 7
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 8
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers3)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 9
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip2 =   tf.keras.layers.Add()([layers3, layers4])
    #Layer 10
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip2)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 11
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 12
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 13
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 14
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 15
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 16
    layers = tf.keras.layers.Conv2D(30, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 17
    layers = tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 18
    layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
    #Layer 19
    predictions = tf.keras.layers.Softmax(axis=1)(layers)
    #Model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    print ("Model GBRAS-Net Built")
    return model



'''Helping function for computing Tanh3 Activation function '''
#Input: X matrix
#Output: Computed matrix
def Tanh3(x):
    return ( tf.keras.backend.tanh(x) * 3 )


'''Calculate How well the discriminator does the job using following equation 3.7'''
# input: Pair of predicted Value matrix shape(1,1,2) and Actual Value matrix shape= (1,1,2)
# output: Loss of Discriminator
def DiscriminatorLossFunction(Av,Pv):
    ld= cross_entropy(Av,Pv)
    # Will never get log zero because of Softamx function output.
    # loss of Discriminator (lD)
    #ld = 0
    #for i in range(2):
     #   ld = ld + Av[0, 0, i] * math.log(Pv[0, 0, i])
    #ld = -ld
    return ld


'''_______________________ GAN functions_________________________________'''


'''Function to calculate the Gradient of generator and discriminator with respect to the loss function'''
# input : Cover image, Target Capacity
# output :UTGradientG,UTGradientD.
@tf.function   #'''the purpose of tf.function  annotation is to cause the function to be compiled'''
def GradientCalculateFunction(CoverImage, TargetCapacity):
    #plt.gray()
    #plt.imshow(tf.reshape(tf.convert_to_tensor(CoverImage), (256, 256)))
    #plt.show()

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        #run the generator to generate Stego images from CoverImages.
        Generated_ProbabilityMap =generator(CoverImage,training=True)
        #plt.imshow(tf.reshape(Generated_ProbabilityMap, (256,256)))
        #plt.show()

        ModificationMap= TernaryEmbeddingSimulator(Generated_ProbabilityMap)
        #plt.imshow(tf.reshape(ModificationMap, (256, 256)))
        #plt.show()

        StegoImage= tf.math.add(CoverImage,ModificationMap)
        #plt.imshow(tf.reshape(StegoImage, (256, 256)))
        #plt.show()

        #run the discriminator to classify both Cover and Stego images
        UT_CoverPrediction= discriminator(CoverImage,training=True)
        UT_StegoPrediction= discriminator(StegoImage,training=True)

        #Evaluate How will the generator work for Generating the stego images
        GeneratorLoss= UT_NetGeneratorLossFunction(Generated_ProbabilityMap,TargetCapacity,tf.convert_to_tensor(np.full((UT_StegoPrediction.shape[0],2),(1,0))), UT_StegoPrediction)

        #Evaluting How will the discriminator work for stego and cover images
        CoverLossD= DiscriminatorLossFunction(tf.convert_to_tensor(np.full((UT_CoverPrediction.shape[0],2),(1,0))),UT_CoverPrediction)
        StegoLossD=DiscriminatorLossFunction(tf.convert_to_tensor(np.full((UT_StegoPrediction.shape[0],2),(0,1))),UT_StegoPrediction)
        DiscriminaotorLoss=CoverLossD+StegoLossD


    UT_GradientG=gen_tape.gradient(GeneratorLoss,generator.trainable_variables)
    UT_GradientD=disc_tape.gradient(DiscriminaotorLoss,discriminator.trainable_variables)

    return UT_GradientG,UT_GradientD,GeneratorLoss,DiscriminaotorLoss

def Train(TrainX,TestX,BatchSize,NumberOfepochs,TargetCapacity):
    for epoch in range(NumberOfepochs):
        start = time.time()
        #Preparing the lists, which will be plotted by GAN_Evaluation
        G_Loss = []
        D_Loss = []
        #Reseting , shuffeling the TrainingDataset, and diving it into batches.
        np.random.shuffle(TrainX)
        mini_Batch_TrainX= np.array_split(TrainX,math.floor(TrainX.shape[0] / BatchSize))
        #resting iteration counter
        iteration=0
        #start= time.time()
        for CoverImage_batch in mini_Batch_TrainX:
            #Find Gradient for the discriminator and generator
            GradientG,GradientD,G_L, D_L= GradientCalculateFunction(CoverImage_batch,TargetCapacity)
            G_Loss.append(G_L)
            D_Loss.append(D_L)
            #update the weights of generator and discriminator
            generator_optimizer.apply_gradients(zip(GradientG,generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(GradientD, discriminator.trainable_variables))
            print("updated")
            iteration= iteration+1
        epochTrainingTime = time.time() - start
        # 1. Visualizing Training Loss Function of generator and Discriminator after each epoch .
        lossGraph(epoch, G_Loss, D_Loss)
        #Evaluating Generator performance After each epoch Using Testing Data.
        # 2. Qualitative Evaluation
        GAN_Evaluation.QualitativeEvaluation (epoch,generator,TestX)
        # 3. Quantitative Evaluation
        GAN_Evaluation.QuantitativeEvaluation(epoch,generator,TestX,epochTrainingTime)
        print('Time for epoch {} is {} sec'.format(epoch + 1, epochTrainingTime))
        #saving after each 10 epoch
        if (epoch%10==0):
            generator.save("UT_Net_GAN/saved_G_models0.4/generator_model_{}.hd5".format(epoch))
            discriminator.save("UT_Net_GAN/Saved_D_models0.4/discriminator_model_{}.hd5".format(epoch))


def lossGraph(epoch, G_Loss, D_Loss):
    os.makedirs(os.path.join(MasterPath, r'TraininglossGraphs'), exist_ok=True)
    plt.plot(G_Loss, 'r-', label='G_Loss')
    plt.plot(D_Loss, 'k-', label='D_Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(MasterPath, f'TraininglossGraphs\epoch {epoch+1}.png'))
    plt.close()
    return




MasterPath = os.path.dirname(os.path.abspath(__file__))

# load Training  and Testing Data
TrainX = np.load(os.path.join(MasterPath, r'Dataset_GAN\TrainingData\TrainingData256\TrainX.npy')).astype('float32') #shape(NumberofImages,256,256,1)
#TrainY = np.load(os.path.join(MasterPath, r'Dataset\TrainingData\TrainingData256\TrainY.npy'))#shape (NumberofImages,2)
TestX = np.load(os.path.join(MasterPath, r'Dataset_GAN\TestingData\TestingData256\TestX.npy')).astype('float32') #shape(NumberofImages,256,256,1)
#TestY = np.load(os.path.join(MasterPath, r'Dataset\TestingData\TestingData256\TestY.npy')) # shape(NumberofImages,2)
#Normalize Training images X between -1 and 1.
TrainX=(TrainX-127.5)/127.5
TestX=(TestX-127.5)/127.5
'''shuffel and Batch inside the train function'''
#build generator and discriminator
generator= UT_Net()
discriminator=GBRAS_NetDiscriminator()
#Define Loss and Optimizer
cross_entropy= tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
Train(TrainX,TestX,3,5,0.4)













































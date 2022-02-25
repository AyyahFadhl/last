import pickle

import numpy as np
import tensorflow as tf


import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import matlab.engine

from numpy import ndarray

MasterPath = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN\TrainingData\TrainingData256\X'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN\TrainingData\TrainingData256\Y'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN\TestingData\TestingData256\X'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN\TestingData\TestingData256\Y'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\X'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\Y'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\X'), exist_ok=True)
os.makedirs(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\Y'), exist_ok=True)
TrainingData=[]
TestingData=[]

'''Read Data from BOSSBase , and Divide it randomly into training and testing 80% to 20% respectively '''
#input: no input
#output:No output
def BOSSBase256_setup():
    # Generate Random Permutation of BOSSBase , which has names from 1 to 10,000.
    Dataset = np.arange(1, 11)
    Dateset = np.random.permutation(Dataset)

    #------------------------------------------------#Training Dataset#------------------------------------------------#
    x=1
    y=1
    for img_num in range(8):
        try:
            # reading,resizing and saving image
            BOSSBase_path = os.path.join(MasterPath, f'Dataset_GAN\BOSSbase_1.01\{Dateset[img_num]}.pgm')
            BOSSBase_img_array = read_pgm(BOSSBase_path)
            BOSSBase_img_array = tf.image.resize(BOSSBase_img_array, (256, 256))
            write_pgm(BOSSBase_img_array, os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\X\{x}.pgm'))

            # Saving the label
            with open(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\Y\{y}.txt'), 'w') as f:
                f.write("1  0")
                f.close()

            # Filling the TrainingData List
            TrainingData.append([BOSSBase_img_array.numpy(), 1, 0])

        except Exception as e:
            print(img_num, "   ", Dateset[img_num])
            pass
        #Updating the counter
        x=x+1
        y=y+1

    #--------------------------------------------#Testing Dataset#------------------------------------------------------#

    x = 1
    y = 1
    for img_num in range(8,10):
        try:
            # reading,resizing, and saving cover image
            BOSSBase_path = os.path.join(MasterPath, f'Dataset_GAN\BOSSbase_1.01\{Dateset[img_num]}.pgm')
            BOSSBase_img_array = read_pgm(BOSSBase_path)
            BOSSBase_img_array = tf.image.resize(BOSSBase_img_array, (256, 256))
            write_pgm(BOSSBase_img_array, os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\X\{x}.pgm'))
            #Saving the label
            with open(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\Y\{y}.txt'), 'w') as f:
                f.write("1  0")
                f.close()
            # Filling the Testing Data List
            TestingData.append([BOSSBase_img_array.numpy(), 1, 0])

        except Exception as e:
            print(img_num, "   ", Dateset[img_num])
            pass
        # Updating the counter
        x=x+1
        y=y+1

    return




'''Read Data from BOWS2, and Divide it randomly into training and testing 80% to 20% respectively '''
#input: No input
#output:No output

def BOWS2256_setup():
    # Generate Random Permutation of  BOWS2, which has names from 1 to 10,000.
    Dataset = np.arange(1, 11)
    Dateset = np.random.permutation(Dataset)

    # ------------------------------------------------#Training Dataset#------------------------------------------------#
    x = 8001
    y = 8001
    for img_num in range(8):
        try:
            # reading,resizing,and saving image
            BOWS2_path = os.path.join(MasterPath, f'Dataset_GAN\BOWS2\{Dateset[img_num]}.pgm')
            BOWS2_img_array = read_pgm(BOWS2_path)
            BOWS2_img_array = tf.image.resize(BOWS2_img_array, (256, 256))
            write_pgm(BOWS2_img_array, os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\X\{x}.pgm'))

            #Saving the label
            with open(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\Y\{y}.txt'), 'w') as f:
                f.write("1  0")
                f.close()

            # Filling the TrainingData List
            TrainingData.append([BOWS2_img_array.numpy(), 1, 0])

        except Exception as e:
            print(img_num, "   ", Dateset[img_num])
            pass
        # Updating the counter
        x = x + 1
        y = y + 1
    # --------------------------------------------#Testing Dataset#------------------------------------------------------#

    x = 2001
    y = 2001
    for img_num in range(8, 10):
        try:
            # reading,resizing and saving image
            BOWS2_path = os.path.join(MasterPath, f'Dataset_GAN\BOWS2\{Dateset[img_num]}.pgm')
            BOWS2_img_array = read_pgm(BOWS2_path)
            BOWS2_img_array = tf.image.resize(BOWS2_img_array, (256, 256))
            write_pgm(BOWS2_img_array, os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\X\{x}.pgm'))
            #Saving the label
            with open(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\Y\{y}.txt'), 'w') as f:
                f.write("1  0")
                f.close()

                # Filling the TrainingData List
                TestingData.append([BOWS2_img_array.numpy(), 1, 0])

        except Exception as e:
            print(img_num, "   ", Dateset[img_num])
            pass
        # Updating the counter
        x = x + 1
        y = y + 1
    return



def Setup():

    # Saving Beselearner Created training and Validation Data as numpy file. Note(the residual is being saved)
    TrainX = []
    TrainY = []
    TestX = []
    TestY = []

    for image, CoverProbability, StegoProbability in TrainingData:
        TrainX.append(image)
        TrainY.append(CoverProbability)
        TrainY.append(StegoProbability)

    for image, CoverProbability, StegoProbability in TestingData:
        TestX.append(image)
        TestY.append(CoverProbability)
        TestY.append(StegoProbability)

    TrainX=np.array(TrainX).reshape(-1, 256, 256, 1)
    TrainY = np.array(TrainY).reshape(-1, 2)
    TestX = np.array(TestX).reshape(-1, 256, 256, 1)
    TestY = np.array(TestY).reshape(-1, 2)

    np.save(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\TrainX.npy'), TrainX)
    np.save(os.path.join(MasterPath, f'Dataset_GAN_try\TrainingData\TrainingData256\TrainY.npy'), TrainY)
    np.save(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\TestX.npy'), TestX)
    np.save(os.path.join(MasterPath, f'Dataset_GAN_try\TestingData\TestingData256\TestY.npy'), TestY)

    return






'''______________________________________________________ General Helping Functions___________________________________________ '''
'''Return image data from a raw PGM file as numpy array.
Format specification: http://netpbm.sourceforge.net/doc/pgm.html'''
#input: file name
#output: 4D tensorflow of image
def read_pgm(filename):
    image = matplotlib.pyplot.imread(filename,format='pgm')
    image= tf.reshape(tf.convert_to_tensor(image),(1,512,512,1))
    #Plotting read image
    #pyplot.imshow(image, pyplot.cm.gray)
    #pyplot.show()
    return image


''' Write Numpy array in PGM format to file.'''
#input: 4D tesnor of image, file name
#output:no output
def write_pgm(image, filename):
    #Converting Tensor to  2D numpy array
    image= tf.cast(image,dtype= tf.uint8)
    image=np.reshape(image.numpy(),(image.shape[1],image.shape[2]))
    height= image.shape[0]
    width = image.shape[1]
    maxval = image.max()
    with open(filename, 'wb') as f:
        f.write(bytes('P5\n{} {}\n{}\n'.format(width, height, maxval), 'ascii'))
        # not sure if next line works universally, but seems to work on my mac
        image.tofile(f)
    return





if __name__ == '__main__':
    '''
    eng = matlab.engine.start_matlab()
    # Generate the SRM feature map, reshape it and prepare it to be tested by trained ensemble
    SRMFeatures = eng.SRM(os.path.join(MasterPath,r"Dataset_GAN_try\TestingData\TestingData256\X\1.pgm"))
    SRMFeatures = np.array(list(SRMFeatures.items()))  # convert dict to list, then convert to np array(2,106)
    SRMFeatures = SRMFeatures.transpose()  # transpose it to make it  (2, 106), instead of (106,2)
    SRMFeatures = np.delete(SRMFeatures, (0), axis=0)

    # 1. load the trained SRM+EC model as list
    with open(os.path.join(MasterPath, "Dataset_Steganalyzer/SRMTrainedEnsemble.pkl"), 'rb') as f:
        TrainedEnsemble = pickle.load(f)
        f.close()

    # 2. call ensemble Testing to get the prediction, and save all the prediction in list to compute detection error later on
    Results = eng.ensemble_testing(matlab.double(SRMFeatures.tolist()), TrainedEnsemble)
    print(Results["predictions"])# this return -1 cover or +1 stego

    #BOSSBase256_setup()
    #print("BOSSBase256_setup is done")
    #BOWS2256_setup()
    #print("BOWS2256_setup is done")
    #Setup()'''





# See PyCharm help at https://www.jetbrains.com/help/pycharm/

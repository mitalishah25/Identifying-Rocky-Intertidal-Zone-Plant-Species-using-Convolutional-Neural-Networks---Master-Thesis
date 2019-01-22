# Mitali Shah
# Code to test the trained unet model on unseen data
# Perform image preprocessing by applying histogram equalization to obtain accurate results.
# Save the amount of Silvetia present in each image in an excel file.
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from train_unet import *
import cv2
import numpy as np
import glob
import time
import os
import xlsxwriter
from PIL import Image

class test_unet(object):
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
        self.test_image = input("Enter the image directory to be processed: ")
        self.folder_path = ''# main folder path
        self.test_path = '' # path for test images
        self.img_type = '' # image filetype same as train unet
        self.npy_path = '' # path to store .npy data
        self.test_results = '' #path to store test results
        if not os.path.exists(self.test_results):
            os.mkdir(self.test_results)
        self.test_image_mask = self.test_results+'test/'
        if not os.path.exists(self.test_image_mask ):
            os.mkdir(self.test_image_mask)
        self.processed = self.test_results+'processed/'
        if not os.path.exists(self.processed):
            os.mkdir(self.processed)

    def create_test_data(self):
        # image to array
        i = 0
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.img_rows, self.img_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[512, 512])
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = self.folder_path+'results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        test_data_npy = self.npy_path + '/imgs_test.npy'
        np.save(test_data_npy, imgdatas)


    def load_test_data(self):
        print('-' * 30)
        print('Loading the test images')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + '/imgs_test.npy')
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

    def make_predictions(self):       
        #json_file = open (train_unet.model_filename,'r')
        json_file = open (self.folder_path+'saved_models/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model =model_from_json(loaded_model_json)
        #loaded_model.load_weights(train_unet.saved_model)
        loaded_model.load_weights(self.folder_path+'unet.hdf5')
        imgs_test = modified_unet.load_test_data()
        print('-' * 30)
        print('Predicting the segmented mask')
        print('-' * 30)
        imgs_mask_test = loaded_model.predict(imgs_test, batch_size=1, verbose=1)
        imgs_mask_test_name = self.folder_path+'results/imgs_mask_test.npy'
        np.save(imgs_mask_test_name, imgs_mask_test)
	
    def save_img(self):
	    imgs = np.load(self.folder_path+'results/imgs_mask_test.npy')
	    piclist = []
	    for line in open(self.folder_path+'results/pic.txt'):
	        line = line.strip()
	        picname = line.split('/')[-1]
	        piclist.append(picname)
	    for i in range(imgs.shape[0]):
	        path = self.test_image_mask + piclist[i]
	        img = imgs[i]
	        img = array_to_img(img)
	        img.save(path)
	        cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	        cv_pic = cv2.resize(cv_pic,(512,512),interpolation=cv2.INTER_CUBIC)
	        binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	        cv2.imwrite(path, cv_save)

    def test(self):
        print('-'*30)
        print('Loading the model and weights from the disk')
        print('-'*30)
        modified_unet.make_predictions()
        print('-'*30)
        print('Saving the predicted mask')
        print('-'*30)
        modified_unet.save_img()
        #print('-'*30)
        #print('Merging the predicted images')
        #print('-'*30)



if __name__ == '__main__':
    modified_unet = test_unet()
    modified_unet.create_test_data()
   
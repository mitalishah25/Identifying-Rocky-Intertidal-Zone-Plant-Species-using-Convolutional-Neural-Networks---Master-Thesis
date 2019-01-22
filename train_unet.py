#Mitali Shah
# Modified unet model for segmenting images.
# Used a small inter-tidal zone dataset for training.

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
import cv2
from data import *
import numpy as np
import glob
import time
import os
import matplotlib.pyplot as plt

class modifiedUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.main_folder = '' # path of main folder
        self.data_path = self.main_folder + '' # path of folder containing training and validation images
        self.label_path = self.main_folder + '' # path of folder containing training and validation image mask
        self.img_type = self.main_folder + '' # using filetype of image
        self.test_path = self.main_folder + '' # folder containing test image 
        self.npy_path = self.main_folder + '' # folder to store npydata

    def create_train_arr(self):
        i = 0
        imgs = glob.glob(self.data_path+"/*."+self.img_type) # images with same filetype 
        imgdata = np.ndarray((len(imgs), self.img_rows, self.img_cols, 3), dtype=np.uint8) 
        imglabel = np.ndarray((len(imgs), self.img_rows, self.img_cols, 1), dtype=np.uint8)

        for x in range(len(imgs)):
            imgpath = imgs[x]
            pic_name = imgpath.split('/')[-1]
            labelpath = self.label_path + '/' + pic_name
            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdata[i] = img
            imglabel[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        np.save(self.npy_path + '/imgs_train.npy', imgdata)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabel)
       


    def load_train_data(self):
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1  
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train


    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
       
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        #model.summary()
        plot_model(model, to_file=self.main_folder + 'model.png')

        return model

    def plot_model_history(self, model_history):
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
        axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
        axs[0].legend(['Training', 'Validation'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'],'r--')
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'], 'b-')
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['Training', 'Validation'], loc='best')
        fig.savefig(self.main_folder + 'accuracy_vs_loss%s.png'%time.strftime("%Y%m%d-%H%M%S"))
        plt.close(fig)

    def save_model_to_json(self):
        classifier_model_json = model.to_json()
        model_filename = self.main_folder + 'model.json'
        with open(model_filename, "w") as json_file:
            json_file.write(classifier_model_json)

    def train(self):
        # callbacks for model
        model_checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True) # check change in validation loss
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto') # stop fitting the model if there no change
        print('-'*30)
        print("Loading the training data")
        print('-'*30)
        imgs_train, imgs_mask_train = self.load_train_data()
        print('-'*30)
        print("Gettting the U-Net model")
        print('-'*30)
        model = self.get_unet()
        print('-'*30)
        print("Saving the model")
        print('-'*30)
        saved_model = self.main_folder + '/unet.hdf5'
        print('-'*30)
        print('Fitting the model')
        print('-'*30)
        model_info = model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=60, verbose=1,
                  validation_split=0.15, shuffle=True, callbacks=[model_checkpoint])
        print('-'*30)
        print('Saving the model to a json file')
        print('-'*30)
        modified_unet.save_model_to_json()
        print('-'*30)
        print('Saving the model accuracy and loss graph')
        print('-'*30)
        modified_unet.plot_model_history(model_info)


if __name__ == '__main__':
    modified_unet = modifiedUnet()
    model = modified_unet.get_unet()
    modified_unet.create_train_arr()
    modified_unet.train()
    
    
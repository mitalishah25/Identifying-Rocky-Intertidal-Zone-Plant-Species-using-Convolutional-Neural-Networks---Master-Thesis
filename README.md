## Identifying Rocky Intertidal Zone Plant Species using Convolutional Neural Networks - Master Thesis 
[Read Thesis](http://repository.library.csuci.edu/handle/10211.3/206009)

The goal of the thesis was to perform image segmentation and calculate and record the percentage of Silvetia Compressa present in the image for analysts to perform ecological monitoring.

**Dataset used:** Marine Dataset collected by Biology Department, CSUCI. The dataset comprised of 100 training and validation images that were split randomly (85% training images and 15% validation images) during the model training and 50 test images.
<p align="center"><img src="https://github.com/jtisaacs/BioImages/blob/master/sil0.png" width="300" height="300" title="Sample image of Silvetia Compressa"/></p>

**Application used for annotating images:** [Image Segmenter App in MATLAB](https://www.mathworks.com/help/images/ref/imagesegmenter-app.html). This application was used to create segmented images for training and validation. The 50 test images were also segmented and were used as ground truth images to test the model. 

**Machine Learning Technique used:** Transfer Learning using [U-Net model architecture](https://arxiv.org/abs/1505.04597). The model architecture was fine-tuned for the current dataset. The fine-tuning invoved using dropout layer to avoid overfitting, using adam optimizer, etc. The model was built using Keras machine learning library and the segmented image was created using OpenCV libraries.

**Accuracy vs Error:** After training the model, training accuracy of 97.56 % and validation accuracy of 95.24 % and training loss of 0.098 and validation loss of 0.152 was achieved. The training vs error graph can be seen below.  
<p align="center"><img src="https://github.com/mitalishah25/image_segmentation_unet/blob/master/unet_graph.jpg" width="400" height="200" /></p>

**Result:** Below is the result of model on test image. The left image is the test image, the center image is the ground truth mask used to 
check the model accuracy on test images and the right image is the predicted mask.
<p align="center"><img src="https://github.com/mitalishah25/image_segmentation_unet/blob/master/unpreprocessed_test_results.png" width="400" height="200" /></p>

**Image preprocessing technique used to obtain accurate test results:** When the model was tested on test images few images did not give desired results so these images were preprocessed. Two image preprocessing techniques - histogram equalization and adaptive histogram equalization were used. On comparing the segmented result on the preprocessed images histogram equalization was selected as the image preprocessing technique.  
*Results before applying histogram equalization.* The left image is the test image, the center image is the ground truth mask and the right image is the predicted mask.
<p align="center"><img src="https://github.com/mitalishah25/image_segmentation_unet/blob/master/unpreprocessed_test_results1.png" width="400" height="166" /></p>

*Results after applying histogram equalization.* The left image is the preprocessed test image, the center image is the ground truth mask and the right image is the predicted mask.
<p align="center"><img src="https://github.com/mitalishah25/image_segmentation_unet/blob/master/preprocessed_image_result.png" width="400" height="166" /></p>



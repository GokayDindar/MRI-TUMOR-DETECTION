# MRI-TUMOR-DETECTION
Gazi University Computer Engineering Graduation Project 

NOTE: VIRTUAL ENVIROMENT SHOULD BE ACTIVED! 

Dataset

LGG Segmentation Dataset
The Cancer Imaging Archive (TCIA)

Total samples 2828 validated images.

EPOCHS = 51
BATCH_SIZE = 32
learning_rate = 1e-4

![image](https://github.com/GokayDindar/MRI-TUMOR-DETECTION/assets/50152111/a1176f49-7b20-44ea-bf6d-00f3bfdfcc4e)

loss: -0.8823 
binary_accuracy: 0.9979
iou: 0.7934
dice_coef: 0.8828
val_loss: -0.8882
val_binary_accuracy: 0.9976
val_iou: 0.8026 
val_dice_coef: 0.8897



Model: "UNET"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 256, 256, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv2d (Conv2D)                (None, 256, 256, 64  1792        ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 activation (Activation)        (None, 256, 256, 64  0           ['conv2d[0][0]']                 
                                )                                                                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 256, 256, 64  36928       ['activation[0][0]']             
                                )                                                                 
                                                                                                  
 batch_normalization (BatchNorm  (None, 256, 256, 64  256        ['conv2d_1[0][0]']               
 alization)                     )                                                                 
                                                                                                  
 activation_1 (Activation)      (None, 256, 256, 64  0           ['batch_normalization[0][0]']    
                                )                                                                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 128, 128, 64  0           ['activation_1[0][0]']           
                                )                                                                 
                                                                                                  
 conv2d_2 (Conv2D)              (None, 128, 128, 12  73856       ['max_pooling2d[0][0]']          
                                8)                                                                
                                                                                                  
 activation_2 (Activation)      (None, 128, 128, 12  0           ['conv2d_2[0][0]']               
                                8)                                                                
                                                                                                  
 conv2d_3 (Conv2D)              (None, 128, 128, 12  147584      ['activation_2[0][0]']           
                                8)                                                                
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 128, 128, 12  512        ['conv2d_3[0][0]']               
 rmalization)                   8)                                                                
                                                                                                  
 activation_3 (Activation)      (None, 128, 128, 12  0           ['batch_normalization_1[0][0]']  
                                8)                                                                
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 128)  0          ['activation_3[0][0]']           
                                                                                                  
 conv2d_4 (Conv2D)              (None, 64, 64, 256)  295168      ['max_pooling2d_1[0][0]']        
                                                                                                  
 activation_4 (Activation)      (None, 64, 64, 256)  0           ['conv2d_4[0][0]']               
                                                                                                  
 conv2d_5 (Conv2D)              (None, 64, 64, 256)  590080      ['activation_4[0][0]']           
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 64, 64, 256)  1024       ['conv2d_5[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_5 (Activation)      (None, 64, 64, 256)  0           ['batch_normalization_2[0][0]']  
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 256)  0          ['activation_5[0][0]']           
                                                                                                  
 conv2d_6 (Conv2D)              (None, 32, 32, 512)  1180160     ['max_pooling2d_2[0][0]']        
                                                                                                  
 activation_6 (Activation)      (None, 32, 32, 512)  0           ['conv2d_6[0][0]']               
                                                                                                  
 conv2d_7 (Conv2D)              (None, 32, 32, 512)  2359808     ['activation_6[0][0]']           
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 32, 32, 512)  2048       ['conv2d_7[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_7 (Activation)      (None, 32, 32, 512)  0           ['batch_normalization_3[0][0]']  
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 512)  0          ['activation_7[0][0]']           
                                                                                                  
 conv2d_8 (Conv2D)              (None, 16, 16, 1024  4719616     ['max_pooling2d_3[0][0]']        
                                )                                                                 
                                                                                                  
 activation_8 (Activation)      (None, 16, 16, 1024  0           ['conv2d_8[0][0]']               
                                )                                                                 
                                                                                                  
 conv2d_9 (Conv2D)              (None, 16, 16, 1024  9438208     ['activation_8[0][0]']           
                                )                                                                 
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 16, 16, 1024  4096       ['conv2d_9[0][0]']               
 rmalization)                   )                                                                 
                                                                                                  
 activation_9 (Activation)      (None, 16, 16, 1024  0           ['batch_normalization_4[0][0]']  
                                )                                                                 
                                                                                                  
 conv2d_transpose (Conv2DTransp  (None, 32, 32, 512)  2097664    ['activation_9[0][0]']           
 ose)                                                                                             
                                                                                                  
 concatenate (Concatenate)      (None, 32, 32, 1024  0           ['conv2d_transpose[0][0]',       
                                )                                 'conv2d_7[0][0]']               
                                                                                                  
 conv2d_10 (Conv2D)             (None, 32, 32, 512)  4719104     ['concatenate[0][0]']            
                                                                                                  
 activation_10 (Activation)     (None, 32, 32, 512)  0           ['conv2d_10[0][0]']              
                                                                                                  
 conv2d_11 (Conv2D)             (None, 32, 32, 512)  2359808     ['activation_10[0][0]']          
                                                                                                  
 batch_normalization_5 (BatchNo  (None, 32, 32, 512)  2048       ['conv2d_11[0][0]']              
 rmalization)                                                                                     
                                                                                                  
 activation_11 (Activation)     (None, 32, 32, 512)  0           ['batch_normalization_5[0][0]']  
                                                                                                  
 conv2d_transpose_1 (Conv2DTran  (None, 64, 64, 256)  524544     ['activation_11[0][0]']          
 spose)                                                                                           
                                                                                                  
 concatenate_1 (Concatenate)    (None, 64, 64, 512)  0           ['conv2d_transpose_1[0][0]',     
                                                                  'conv2d_5[0][0]']               
                                                                                                  
 conv2d_12 (Conv2D)             (None, 64, 64, 256)  1179904     ['concatenate_1[0][0]']          
                                                                                                  
 activation_12 (Activation)     (None, 64, 64, 256)  0           ['conv2d_12[0][0]']              
                                                                                                  
 conv2d_13 (Conv2D)             (None, 64, 64, 256)  590080      ['activation_12[0][0]']          
                                                                                                  
 batch_normalization_6 (BatchNo  (None, 64, 64, 256)  1024       ['conv2d_13[0][0]']              
 rmalization)                                                                                     
                                                                                                  
 activation_13 (Activation)     (None, 64, 64, 256)  0           ['batch_normalization_6[0][0]']  
                                                                                                  
 conv2d_transpose_2 (Conv2DTran  (None, 128, 128, 12  131200     ['activation_13[0][0]']          
 spose)                         8)                                                                
                                                                                                  
 concatenate_2 (Concatenate)    (None, 128, 128, 25  0           ['conv2d_transpose_2[0][0]',     
                                6)                                'conv2d_3[0][0]']               
                                                                                                  
 conv2d_14 (Conv2D)             (None, 128, 128, 12  295040      ['concatenate_2[0][0]']          
                                8)                                                                
                                                                                                  
 activation_14 (Activation)     (None, 128, 128, 12  0           ['conv2d_14[0][0]']              
                                8)                                                                
                                                                                                  
 conv2d_15 (Conv2D)             (None, 128, 128, 12  147584      ['activation_14[0][0]']          
                                8)                                                                
                                                                                                  
 batch_normalization_7 (BatchNo  (None, 128, 128, 12  512        ['conv2d_15[0][0]']              
 rmalization)                   8)                                                                
                                                                                                  
 activation_15 (Activation)     (None, 128, 128, 12  0           ['batch_normalization_7[0][0]']  
                                8)                                                                
                                                                                                  
 conv2d_transpose_3 (Conv2DTran  (None, 256, 256, 64  32832      ['activation_15[0][0]']          
 spose)                         )                                                                 
                                                                                                  
 concatenate_3 (Concatenate)    (None, 256, 256, 12  0           ['conv2d_transpose_3[0][0]',     
                                8)                                'conv2d_1[0][0]']               
                                                                                                  
 conv2d_16 (Conv2D)             (None, 256, 256, 64  73792       ['concatenate_3[0][0]']          
                                )                                                                 
                                                                                                  
 activation_16 (Activation)     (None, 256, 256, 64  0           ['conv2d_16[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_17 (Conv2D)             (None, 256, 256, 64  36928       ['activation_16[0][0]']          
                                )                                                                 
                                                                                                  
 batch_normalization_8 (BatchNo  (None, 256, 256, 64  256        ['conv2d_17[0][0]']              
 rmalization)                   )                                                                 
                                                                                                  
 activation_17 (Activation)     (None, 256, 256, 64  0           ['batch_normalization_8[0][0]']  
                                )                                                                 
                                                                                                  
 conv2d_18 (Conv2D)             (None, 256, 256, 1)  65          ['activation_17[0][0]']          
                                                                                                  
==================================================================================================
Total params: 31,043,521
Trainable params: 31,037,633
Non-trainable params: 5,888
_______________________________


<img width="1203" alt="Screenshot 2023-06-09 at 18 36 52" src="https://github.com/GokayDindar/MRI-TUMOR-DETECTION/assets/50152111/435649b4-f93e-486b-8663-6f5ae1d68400">
<img width="1203" alt="Screenshot 2023-06-09 at 18 37 16" src="https://github.com/GokayDindar/MRI-TUMOR-DETECTION/assets/50152111/af3c8d18-7566-4b6b-9fa7-d5b671fd4a79">



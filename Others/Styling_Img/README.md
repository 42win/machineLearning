# Styling Image

## Library

- tensorflow
- tensorflow_hub [link](https://tfhub.dev/)
- numpy
- PIL.Image
- time
- functools

## Description

- **Type** : Others - styling 
  
- **Background** : to prove that ML can also make art like human do
  
- **Business Understanding**:
  1. Problem Statement
     - How to make art using ML
   
  2. Goals:
     - to make art using ML

- **Data understanding**
  1. Overview
     - we use cat image and painting.
     - images format are jpg  
  
  2. Data Preparation
     - resize image 
     - change image to tensor
  
  3. Modeling
     - model : Neural Style Transfer

## Work Stages

1. make function to change tensor to image to input image to model
2. make function to chage image to tensor to show result of model
3. insert images 
4. change images to tensor
5. do style transfer 

## Code Snipet

- function tensor to image
  ```py
   def tensor_to_image(tensor):
      tensor = tensor*255
      tensor = np.array(tensor, dtype=np.uint8)

      if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
      return PIL.Image.fromarray(tensor)
  ```

- function image to tensor
  ```py
   import tensorflow as tf

   def image_to_tensor(path_to_img):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
        
        return img
  ```

- resize image
  ```py
   # Resize the image to specific dimensions
   img = tf.image.resize(img, [512, 512])
   img = img[tf.newaxis, :]

   return img
  ```

- use function image_to_tensor
  ```py
   kucing_tensor = image_to_tensor('/content/kucing.jpg')
  ```

- neural style transfer
  ```py
   import tensorflow_hub as hub
   import numpy as np
   import PIL.Image
   import time
   import functools

   hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
   stylized_image = hub_module(tf.constant(kucing_tensor), tf.constant(style_tensor))[0]
    
   tensor_to_image(stylized_image)
  ```

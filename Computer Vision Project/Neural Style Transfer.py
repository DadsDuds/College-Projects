# Neural Style Transfer program that merges two styles from two different images.
# Actually super proud of this one.

# A BUNCH OF MODULES
import os                      # this module provides a way of using operating system functionality
import time                    # helps represent time in code
import numpy as np             # used for mainly array computing
import keras                   # deep learning module with various predefined classes and functions
import keras.preprocessing     # utilizes the preprocessing tools from keras to work with our images
import scipy.optimize          # this package provides optimization algorithms (though it's only used for one line - crucial nonetheless)
import tensorflow as tf        # open-source library in the sense that it's a math library specialized for machine learning
tf.compat.v1.disable_eager_execution()  

# Evaluator class makes it possible to compute loss and gradients in one pass
class Evaluator(object):
    
    # initializes the class
    def __init__(self, rows: int, cols: int, outputs: []):
        
        self.loss_value = None
        self.grads_value = None
        self.rows = rows
        self.cols = cols
        self.outputs = outputs
    
    # calculates loss
    def loss(self, x):
        
        loss_value, grad_values = eval_loss_and_grads(x, self.rows, self.cols, self.outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    # calculates gradients
    def grads(self, x):
        
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
def gram_matrix(x): # gram matrix of an image tensor
    
    # Gram matrices capture styles of feature maps from both the style image
    # and from the resulting image
    
    if keras.backend.image_data_format() == 'channels_first':
        features = keras.backend.batch_flatten(x)
    else:
        features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1)))
    
    return keras.backend.dot(features, keras.backend.transpose(features))

def preprocess_image(path: str, rows: int, cols: int):
    # this function allows us to open, resize, and format our images into tensors
    
    x = keras.preprocessing.image.load_img(path, target_size = (rows, cols))
    
    x = keras.preprocessing.image.img_to_array(x)   # converts into an array
    x = np.expand_dims(x, axis = 0)
    
    x = keras.applications.vgg19.preprocess_input(x)
    
    return x

def deprocess_image(x, rows: int, cols: int):
    # this function converts tensors into valid images
    
    if keras.backend.image_data_format() == 'channels_first':
        x = x.reshape((3, rows, cols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((rows, cols, 3))
    
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    
    # BGR TO RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

def style_loss(style, combination, rows: int, cols: int):
    # keeps the generated image close to the local textures of the style reference image
    
    # calculate input values
    S = gram_matrix(style)
    C = gram_matrix(combination) 
    channels = 3
    size = rows * cols
    
    return keras.backend.sum(keras.backend.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    # keeps the high-level representation of the generated image close to the base image
    
    return keras.backend.sum(keras.backend.square(combination - base))

def total_variation_loss(x, rows: int, cols: int):
    # keeps the generated image from looking like a mess
    
    # element-wize squaring
    if keras.backend.image_data_format() == 'channels_first':
        a = keras.backend.square(x[:,:,:rows - 1, :cols - 1] - x[:,:, 1:,:cols - 1])
        b = keras.backend.square(x[:,:,:rows - 1, :cols - 1] - x[:,:,:rows - 1, 1:])
    else:
        a = keras.backend.square(x[:,:rows - 1, :cols - 1, :] - x[:, 1:, :cols - 1, :])
        b = keras.backend.square(x[:,:rows - 1, :cols - 1, :] - x[:,:rows - 1, 1:, :])
    
    return keras.backend.sum(keras.backend.pow(a + b, 1.25))

def eval_loss_and_grads(x, rows: int, cols: int, outputs: []):  # name is fairly straight-forward
    
    # reshape image
    if keras.backend.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, rows, cols))
    else:
        x = x.reshape((1, rows, cols, 3))
    
    # get loss value
    outs = outputs([x])
    loss_value = outs[0]
    
    # gets gradient values
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    
    return loss_value, grad_values

def main():
    
    # --- USER SETTINGS ---
    
    # THIS IS WHERE YOU'D ADD THE PATH TO YOUR IMAGES (YOURS WILL VARY)
    # ACCEPTS BOTH JPG AND PNG
    base_image_path = 'C:/Users/username/Documents/--.jpg/.png'
    style_image_path = 'C:/Users/username/Documents/--.jpg/.png'
    output_image_path = 'C:/Users/username/Documents/--.jpg/.png'
    
    # CHANGE THE WEIGHTS OF DIFFERENT LOSS COMPONENTS [0-1]
    # IN OTHER WORDS, HOW STRONG SHOULD THE VARIATION AND THE STYLE IMAGE BE
    total_variation_weight = 0.75
    style_weight = 0.8
    content_weight = 0.2
    
    iterations = 60 # HOW MANY TIMES SHOULD THE PROGRAM KEEP MERGING BOTH IMAGES' STYLES
    
    # ---
    
    # Dimensions of the generated image
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    
    rows = 256  # I'm pretty sure you can change this
    cols = int(width * rows / height)
    
    # preprocess images
    base_image = keras.backend.variable(preprocess_image(base_image_path, rows, cols))
    style_image = keras.backend.variable(preprocess_image(style_image_path, rows, cols))
    output_image = None
    
    # the output_image will be our resulting image
    if keras.backend.image_data_format() == 'channels_first':
        output_image = keras.backend.placeholder((1, 3, rows, cols))
    else:
        output_image = keras.backend.placeholder((1, rows, cols, 3))
    
    # combine 3 images into a single keras tensor
    input_tensor = keras.backend.concatenate([base_image, style_image, output_image], axis = 0)
    
    # uses the VGG19 network
    model = keras.applications.vgg19.VGG19(input_tensor = input_tensor,
                                           weights = 'imagenet', include_top = False)
    
    # gets the outputs of each layer
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    
    # Adds content loss
    loss = keras.backend.variable(0.0)
    layer_features = outputs_dict['block5_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)
    
    # list of layers the program uses for the style loss
    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    # Adds style loss
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features, rows, cols)
        loss = loss + (style_weight / len(feature_layers)) * sl
    
    # this line adds total variation loss
    loss = loss + total_variation_weight * total_variation_loss(output_image, rows, cols)
    
    grads = keras.backend.gradients(loss, output_image)
    outputs = [loss]
    
    # gets outputs
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    
    # creates an evaluator
    evaluator = Evaluator(rows, cols, keras.backend.function([output_image], outputs))
    
    # gets input image
    if(os.path.isfile(output_image_path)) == True:
        x = preprocess_image(output_image_path, rows, cols)
    else:
        x = preprocess_image(base_image_path, rows, cols)
    
    # repeatedly runs gradient descents and saves the resulting image every iteration
    for i in range(iterations):
        
        print('Start of iteration', i + 1)
        start_time = time.time()
        
        # run scipy-based optimization (uses the L-BFGS-B algorithm)
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                                        fprime = evaluator.grads, maxfun = 20)
        print('Current loss value: ', min_val)
        
        # deprocess and save generated image
        img = deprocess_image(x.copy(), rows, cols)
        keras.preprocessing.image.save_img(output_image_path, img)
        
        print('Iteration {0} completed in {1} seconds'.format(i + 1, round(time.time() - start_time, 2)))

# tells python to run the main function
if __name__ == '__main__': main()    
    
    
    
    


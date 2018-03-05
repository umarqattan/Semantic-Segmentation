
import tensorflow as tf
import os.path
import warnings
from distutils.version import LooseVersion
import glob
import helper
import project_tests as tests

# DIRECTORY PATHS

DATA_DIRECTORY = './data'
RUNS_DIRECTORY = './runs'
TRAINING_DATA_DIRECTORY ='./data/data_road/training'
NUMBER_OF_IMAGES = len(glob.glob('./data/data_road/training/calib/*.*'))
VGG_PATH = './data/vgg'
training_losses_list = [] # Used for plotting to visualize if our training is going well given parameters

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))


# Check for a GPU
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

correct_label = tf.placeholder(tf.float32, [None, helper.IMAGE_SHAPE[0], helper.IMAGE_SHAPE[1], helper.NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

def load_vgg(sess, vgg_path):
  
  # 1. Load a an already trained (Pre-trained) VGG Model into TensorFlow
  # 2. Create and load a TensorFlow session
  # 3. Get relative paths to the variables and saved_model.pb directory and file, respectively
  # 4. Return a tuple of tensors including the image_input, keep_prob, and layers 3, 4, and 7.
  
  model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes = helper.NUMBER_OF_CLASSES):
  
  # 1. Apply 1x1 convolutions to the gg_layer(3,4,7)_output encoded layers
  # 2. Add decoded layers to the fully convolutional network by adding skip layers such as layers 2 and 4
  #    and upsampled layers such as 1, 3, and the ouput
  # 3. Return a tensor for the decoded output layer, decoderlayer_output.

  layer3_1by1 = helper.conv1by1(layer = vgg_layer3_out, layer_name = "layer3conv1by1")
  layer4_1by1 = helper.conv1by1(layer = vgg_layer4_out, layer_name = "layer4conv1by1")
  layer7_1by1 = helper.conv1by1(layer = vgg_layer7_out, layer_name = "layer7conv1by1")
 
  decoderlayer1 = helper.upsample(layer = layer7_1by1, k = 4, s = 2,          layer_name = "decoderlayer1")
  decoderlayer2 = tf.add(decoderlayer1, layer4_1by1,                                name = "decoderlayer2")
  decoderlayer3 = helper.upsample(layer = decoderlayer2, k = 4, s = 2,        layer_name = "decoderlayer3")
  decoderlayer4 = tf.add(decoderlayer3, layer3_1by1,                                name = "decoderlayer4")
  decoderlayer_output = helper.upsample(layer = decoderlayer4, k = 16, s = 8, layer_name = "decoderlayer_output")

  return decoderlayer_output

def optimize(nn_last_layer, correct_label, learning_rate, num_classes = helper.NUMBER_OF_CLASSES):
  
  # 1. Turn the 4-dimensional tensors to 2-dimensional where each row corresponds to the image's pixel value and
  #    each column corresponds to the image's class
  # 2. Compute the cross_entropy and the loss by applying TensorFlow's neural network's softmax_cross_entropy_with_logits
  #    function. This will enable us to find the labels with which to accurately classify the images.
  # 3. Apply the AdamOptimizer function with a learning rate of 0.005 and minimize it by passing it the cross_entropy_loss.
  # 4. Return a tuple consisting of logits, train_op, and cross_entropy_loss

  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  class_labels = tf.reshape(correct_label, (-1, num_classes))

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
  cross_entropy_loss = tf.reduce_mean(cross_entropy)

  train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

  return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
  
  # 1. For each epoch in the given number of epochs to train the neural network, get the batch size of the current batch to pass to the
  #    get_batches_fn helper function and iterate over the images and labels.
  # 2. For each image and label, create a dictionary of the images and their corresponding labels along with the
  #    keep_prob = DROPOUT and learning rate = 0.005
  # 3. Using the train_op parameter, pass it to the current TensorFlow session along with the cross_entropy_loss and dictionary
  #    to calculate the loss
  # 4. For all the losses for the images and labels within the current batch, find the training loss by dividing the sum of the 
  #    losses by the total losses, append it to a global losses list, and continue until all the batches have been iterated over.

  for epoch in range(helper.EPOCHS):
    
    losses, i = [], 0
    
    for images, labels in get_batches_fn(helper.BATCH_SIZE):
        
      i += 1
    
      feed = { input_image: images,
               correct_label: labels,
               keep_prob: helper.DROPOUT,
               learning_rate: helper.LEARNING_RATE }
        
      _, partial_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
      
      print("Iteration #: ", i, " partial loss:", partial_loss)
      losses.append(partial_loss)
          
    training_loss = sum(losses) / len(losses)
    training_losses_list.append(training_loss)
    
    print("--------------------------------")
    print("epoch #: ", epoch + 1, "/", helper.EPOCHS, "with a training loss of: ", training_loss)
    print("--------------------------------")
    
def execute_test_suite():
  tests.test_layers(layers)
  tests.test_optimize(optimize)
  tests.test_for_kitti_dataset(DATA_DIRECTORY)
  tests.test_train_nn(train_nn)

def run():

  helper.maybe_download_pretrained_vgg(DATA_DIRECTORY)
  get_batches_fn = helper.gen_batch_function(TRAINING_DATA_DIRECTORY, helper.IMAGE_SHAPE)
  
  with tf.Session() as session:
        
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, VGG_PATH)
    model_output = layers(layer3, layer4, layer7, helper.NUMBER_OF_CLASSES)
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, helper.NUMBER_OF_CLASSES)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    train_nn(session, helper.EPOCHS, helper.BATCH_SIZE, get_batches_fn, 
             train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)
    helper.save_inference_samples(RUNS_DIRECTORY, DATA_DIRECTORY, session, IMAGE_SHAPE, logits, keep_prob, image_input)

if __name__ == "__main__":
  execute_test_suite()
  run() 
  print(training_losses_list)

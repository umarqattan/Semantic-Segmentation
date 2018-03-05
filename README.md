# Semantic Segmentation
### Implementation
  LOAD VGG
  1. Load a an already trained (Pre-trained) VGG Model into TensorFlow
  2. Create and load a TensorFlow session
  3. Get relative paths to the variables and saved_model.pb directory and file, respectively
  4. Return a tuple of tensors including the image_input, keep_prob, and layers 3, 4, and 7.
  LAYERS
  5. Apply 1x1 convolutions to the gg_layer(3,4,7)_output encoded layers
  6. Add decoded layers to the fully convolutional network by adding skip layers such as layers 2 and 4
     and upsampled layers such as 1, 3, and the ouput
  7. Return a tensor for the decoded output layer, decoderlayer_output.
  OPTIMIZE
  8. Turn the 4-dimensional tensors to 2-dimensional where each row corresponds to the image's pixel value and
     each column corresponds to the image's class
  9. Compute the cross_entropy and the loss by applying TensorFlow's neural network's softmax_cross_entropy_with_logits
     function. This will enable us to find the labels with which to accurately classify the images.
  10. Apply the AdamOptimizer function with a learning rate of 0.005 and minimize it by passing it the cross_entropy_loss.
  11. Return a tuple consisting of logits, train_op, and cross_entropy_loss
  MAIN
  12. For each epoch in the given number of epochs to train the neural network, get the batch size of the current batch to pass to the
      get_batches_fn helper function and iterate over the images and labels.
  13. For each image and label, create a dictionary of the images and their corresponding labels along with the
      keep_prob = DROPOUT and learning rate = 0.005
  14. Using the train_op parameter, pass it to the current TensorFlow session along with the cross_entropy_loss and dictionary
      to calculate the loss
  15. For all the losses for the images and labels within the current batch, find the training loss by dividing the sum of the 
      losses by the total losses, append it to a global losses list, and continue until all the batches have been iterated over.

# Results
Having performed 3 trials with the following dropout rates, `dropout = 0.70` and `dropout = 0.75`, and `dropout = 0.80`  with the following parameters

```
EPOCHS = 20
BATCH_SIZE = 1

LEARNING_RATE = 0.005
DROPOUT = 0.70
```
the following results came out to
epoch #:  1/1 with a training loss:  2.2188401046921227
--------------------------------------------------------------------
Training Finished. Saving test images to: ./runs/1520234339.6033611, where all the tests passed.
 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.

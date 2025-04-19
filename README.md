### Start Up

First look into the `dataset_explore.ipynb` and get familiar with the data.

### Codes need your implementation

1. `op.py` 
   Implement the forward and backward function of `class Linear`
   Implement the `MultiCrossEntropyLoss`. Note that the `Softmax` layer could be included in the `MultiCrossEntropyLoss`.
   Try to implement `conv2D`, do not worry about the efficiency.
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py` You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py` You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.


### Train the model.

Open test_train.py, modify parameters and run it.

If you want to train the model on your own dataset, just change the values of variable *train_images_path* and *train_labels_path*

### Test the model.

Open test_model.py, specify the saved model's path and the test dataset's path, then run the script, the script will output the accuracy on the test dataset.


### Note

1. `op.py` contains the operations of linear layer, softmax, cross entropy loss, and convolutional layer and other useful layers.
2. `models.py` contains the model structure, MLP and different CNN models.
3. `mynn/lr_scheduler.py` contains the learning rate scheduler.
4. `optimizer.py` contains the optimizer, including MomentGD and SGD.
5. `runner.py` contains the training and testing process.
6. `test_train.py` is the script for training the model, it only left one example of training on MNIST dataset.
7. `affine_train.py`is the script for making radomly affine transformation before training the model.
8. `nosie_train.py` is the script for adding noise before training the model.
9. `strength_train.py` is the script for training the model while adding some wrong-classifications samples.
10. `test_model.py` is the script for testing the accuracy of the saved model on the test dataset.
11. `draw_test.py` is the script which you can draw the number on your own and test the model's performance.
12. `weight_visualization.py` is the script for visualizing the weights of the model. It only left the weight on the fisrt layer of the model `CNN_v2_1`



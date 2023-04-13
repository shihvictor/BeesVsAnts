# Bees vs Ants
Classifying bees/ants.

## TABLE OF CONTENTS
* [Installation Instructions](#installation-instructions)
* [Dataset](#dataset)
* [Project Description](#project-description)
* [Model](#model)
* [Training (Testing Tensorboard)](#training-testing-tensorboard)
* [Results](#results)
* [Problems](#problems)
* [Conclusion](#conclusion)
* [References](#references)

## Installation Instructions
1. In VSCode, create a new workspace.  
2. Activate a conda env.
3. Install git in the conda environment. `conda install git`
4. Clone the project. `git clone [URL]`
5. Install the necessary packages.

# Experimental Setup
## Dataset
https://download.pytorch.org/tutorial/hymenoptera_data.zip  
Training set: 245 total images - 124 ant images, 121 bee images  
Validation set: 153 total images - 70 ant images, 83 bee images  
Dimensions: The dimensions of each image in the data vary.  
Labels: Bee, Ant

## Project Description
This project serves as an extension of the LearnPytorchFashionMNIST project to test and practice the use of transfer learning, tensorboard, and the use of basic rapid hyperparameter tuning.

## Data Preprocessing
Resnet18 was used to apply transfer learning and the data preprocessing used for resnet18 was also applied to the training set which includes:
    
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    

For the validation set, a simple resize and center crop was used for consistency.
    
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=[224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
## Model
In order to apply resnet18 to this classification problem, the number of output features for the fully connected layer was changed to 2. One element for each classification label.

## Training (Testing Tensorboard)
Instead of our own custom plotting and saving functions for training neural networks, Tensorboard was chosen to streamline the training process due to its wide range of functionality. This includes features such as plots for analyzing the bias and variance, histograms for the model weights, and a graph that shows the relationship between hyperparameters and the acheived accuracy and loss.

As such, the training loop code was modified to accomodate this more efficient procedure to optimize hyperparameters by testing all combinations of hyperparameters which need to be predefined. The hyperparameters that were tested include the learning rate, batch size, and shuffle status.

# Results
|Run # |lr      |bsize|shuffle|val_accuracy|val_loss    |
|---|--------|-----|-------|------------|------------|
|1  |0.001   |128  |1      |95.42483521 |0.1614577919|
|2  |0.001   |32   |1      |95.42483521 |0.1670802832|
|3  |0.001   |64   |1      |95.42483521 |0.1934683025|
|4  |0.0001  |128  |1      |86.92810822 |0.3700472713|
|5  |0.0001  |32   |1      |86.92810822 |0.412620157 |
|6  |0.0001  |64   |1      |86.92810822 |0.3496196568|
|7  |1.00E-05|128  |1      |66.01306915 |0.6316670179|
|8  |1.00E-05|64   |1      |66.01306915 |0.6570946574|
|9  |1.00E-05|32   |1      |64.70587921 |0.6619263291|


Table 1. Entries are ordered by validation accuracy in descending order.

Each run was run for a total of 20 epochs, used stochastic gradient descent as the optimizer, used cross entropy loss as the cost function, and used stepLR as the learning rate scheduler. Table 1 shows that the runs with the best performance used a learning rate of 1E-3.

# Problems
One problem faced during the implementation of Tensorboard was the large difference between the sample images shown in Tensorboard and sample images manually plotted using matplotlib. This difference was caused by the difference in range for the pixel values. Specifically, Tensorboard images are assumed to lie in the range [0, 1) for float32 values, while the locally loaded image values lie in the range [-1,  1] after data preprocessing. Therefore, in order to ensure that Tensorboard was accurately displaying sample images, the same images plotted using matplotlib were reverted to the state before the data preprocessing step by using torch matrix functions.

# Conclusion
Overall, the objective of applying transfer learning and using tensorboard to streamline the testing process was achieved.

## References
[1] https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning%20ant%20bees

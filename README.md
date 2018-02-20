# MatConv2TensorFlow
Port a MatConvNet trained model to TensorFlow (python) model. 

## IMPORTANT: 
These 2 files are not intended to be universally runable out of the box, they are only intended to serve as a starting point. I only added the important parts I was interested in - Relu, softmax, learning rates, etc. If you look at the datastructure created by matconvnet, you can extend the code for yourself. 

## Coming Soon:
More details on how to modify and/or run for yourself. 

## Files:
 * MatLab2Python.py: Run this first to make a model.p pickle file. You just need to make sure the "net.mat" string matches the name/path to your matconvnet mat file
 
 * Conv2TF.py: this is the specific Tensorflow Model. It REALLY should only be used for reference. 

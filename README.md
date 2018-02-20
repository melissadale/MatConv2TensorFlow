# MatConv2TensorFlow
Port a MatConvNet trained model to TensorFlow (python) model. 

## IMPORTANT: 
These 2 files are not intended to be universally runable out of the box, they are only intended to serve as a starting point. I only added the important parts I was interested in - layer weights/biases, softmax, learning rates, etc. If you look at the datastructure created by matconvnet, you can extend the code for yourself. 


## Files:
This is performed in 2 stages: scrape the matconvnet to make a 

 * **MatLab2Python.py**: Run this first to make a model.p pickle file. You just need to make sure the "net.mat" string matches the name/path to your matconvnet mat file at the bottom of the file. This pulls a python dictionary from the matconvnet model, and then creates a human readable model representation - which is a list of dictionaries for the layers. 
 
 * **Conv2TF.py**: this is the specific Tensorflow Model. It REALLY should only be used for reference in creating your own models. It loads the pickle file generated in the previous file, and creates a basic tensorflow CNN model. To Use: you must update the input size at the top - and adjust the number of layers accordingly. 


## DISCLAIMERS
 * I make no gaurentees about preserving accuracies. I currently am working with unlabelled data. Visually the results appear consistent between MatConvNet and Tensorflow, however I have not compared accuracies. 
 * This was a quick and dirty implementation, I am aware there are more efficient ways to perform this function. My use case, I simply needed the ability to obtain predictions, and the network is very small, allowing a lot of manual coding. 
 
 **PLEASE** If you use this for your own code, consider sharing your edits if they make it more robust.  


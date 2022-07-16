# Answers to Quick Questions
-	How to recognize the images of the same ID?

The datasets, e.g., Market-1501, DukeMTMC-reID, VeRi-776 and CUHK03-NP, involve the ID name in the file names. 
So, as shown in the `prepare.py`, we read the ID from the file name and rearrange images according to their ID. 
Here we use the naming rule of the DukeMTMC-reID dataset as an example. 
In "0005_c2_f0046985.jpg", "0005" is the identity. "c2" means the image from Camera 2. "f0046985" is the 46985th frame in the video of Camera 2.

- What is the difference between the AvgPool2d and AdaptiveAvgPool2d?

`AdaptiveAvgPool2d` requires output dimension as the parameter, while `AvgPool2d` requires stride, padding as the parameter like conv layer. 
If you would like to know more technical details, please refer to the official document [here](https://pytorch.org/docs/stable/nn.html?highlight=adaptiveavgpool2d#torch.nn.AdaptiveAvgPool2d).

- Why do we use AdaptiveAvgPool2d? 

For pedestrian images, of which the height is larger than the width, we need to specify the pooling kernel for `AvgPool2d`.
Therefore, `AdaptiveAvgPool2d` is more easy to implement. Of course, you still could use `AvgPool2d`.

-	Does the model have parameters now? How to initialize the parameter in the new layer?

Yes. However, we should note that the model is randomly initialized if you did not specific how to initial it.
In our case, our model is composed of model_ft and classifier. We deploy two initialization methods.
For `model_ft`, we set the parameter pretrained = True, which is set to the parameter of the model pretrained on ImageNet. 
For the add-on `classifier`, we use the functions  `weights_init_kaiming` in `model.py`. 

-	Why we need optimizer.zero_grad()? What happens if we remove it?

`optimizer.zero_grad()` is to set the gradient of Variable to zero. 
In every iteration, we use the gradient to update the network, so we need to clear the gradient after the update.
Otherwise, the gradient will be accumulated.

-	The dimension of the outputs is batchsize * 751. Why?

The output is the probability of the samples over all classes. The class number of training samples in Market-1501 is 751. 

-	Why we flip the test image horizontally when testing? How to fliplr in pytorch?

Please refer to this [issue](https://github.com/layumi/Person_reID_baseline_pytorch/issues/99).

-	Why we L2-norm the feature?

We L2-norm the feature to apply the cosine distance in the next step. 
You also may use the Euclidean distance, but cosine distance is more easy to implement (using metric multiplication). 

-	Could we directly apply the model trained on Market-1501 to DukeMTMC-reID? Why?

Yes. But the result is not good enough. As shown in many papers, different datasets usually have domain gap, which could be caused by various factors, such as different illumination and different occlusions. 
For further reference, you may check the leaderboard of transfer learning on Market-1501 to DukeMTMC-reid (https://github.com/layumi/DukeMTMC-reID_evaluation/tree/master/State-of-the-art#transfer-learning). 
 

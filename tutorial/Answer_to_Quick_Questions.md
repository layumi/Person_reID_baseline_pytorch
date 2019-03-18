# Answer to Quick Questions
-	How to recognize the images of the same ID?

The datasets, e.g., Market-1501, DukeMTMC-reID, VeRi-776 and CUHK03-NP, involve the ID name in the file names. 
So, as shown in the `prepare.py`, we read the ID from the file name and rearrange images according to their ID. 
Here we use the naming rule of the DukeMTMC-reID dataset as example. 
In "0005_c2_f0046985.jpg", "0005" is the identity. "c2" means the image from Camera 2. "f0046985" is the 46985th frame in the video of Camera 2.

- What is the difference between the AvgPool2d and AdaptiveAvgPool2d?

AdaptiveAvgPool2d requires output dimension as parameter while AvgPool2d requires stride, padding as parameter just like conv layer. 
If you would like to know more technical details, please refer the official document here.

- Why do we use AdaptiveAvgPool2d? 

For pedestrian images, of which the height is larger than the width, we need to specify the pooling kernel for `AvgPool2d`.
Therefore, AdaptiveAvgPool2d is more easy to implement. Of course, you still could use `AvgPool2d`.

-	Does the model have parameters now? How to initialize the parameter in the new layer?

Yes. However, we should note that the model is randomly initialized if you did not specific how to initial it.
In our case, our model is composed of model_ft and classifier. We deploy two initialization methods.
For model_ft, we set the parameter pretrained = True, which means it is the model pretrained on ImageNet. 
When we define the classifier, we use two functions in `model.py` to initialize parameter, that is weights_init_kaiming and weights_init_kaiming. 

## Still in revision

-	Why we need optimizer.zero_grad()? What happens if we remove it?

`optimizer.zero_grad()` is to set the gradient of Variable to zero. 
In every iteration, we just use the batch image to update the network so we need to zero the gradient before the update every batch.
If we remove it, we not only use the batch to update the network, but use previous batches.

-	The dimension of the outputs is batchsize * 751. Why?

We use classification net to train re-id network, the class number is the count of IDs which is 751, so 751 is one dimension of the output. We use mini-batch to train the network so batchsize is the other dimension of the output.

-	Why we flip the test image horizontally when testing? How to fliplr in pytorch?
We flip the image to increase the precision of re-id because of different person orientation (left or right). We flip the image to keep the orientation of the person is same.

In our code, there is a function called fliplr in test.py to flip the image. And index_select function is core function, we use index_select to get the image of which third dimension (dimension of width) is in reverse order. If you want details, please refer the test.py in our code and official document here.

-	Why we L2-norm the feature?

We L2-norm the feature so that we ensure the upper limit of every cos distance is the same. In this case, we can use the cos distance to judge which pairs image is the same or not.
We take 1*2-dimension feature as an example. A = [1,2], B = [1,1.9], C = [4,4]. We use cos-distance to judge which is more similar to A (from normal sense, C is more similar to A). First without L2-norm, A*B=4.8 while A*C = 12 which means C is more similar to A (the more cos-distance is, the more similar), then with L2-norm, A*B/|A|/|B| = 1.00 while A*C/|A|/|B|=0.948 which means B is more similar to A.

-	Could we directly apply the model trained on Market-1501 to DukeMTMC-reID? Why?

No. Two datasets have domain gap, such as the different illumination, different city, different weather. Actually, some paper apply the model trained on Market-1501 to DukeMTMC-reID as a baseline. such as spgan. And in this paper, it calls direct transfer of which precision is very low.
 

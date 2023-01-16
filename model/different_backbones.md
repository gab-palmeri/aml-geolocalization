# Use of different pretrained ResNet-18
ResNet-18 is the default backbone of the project, by default it's pretrained on ImageNet, that is a general purpose dataset with 1000 classes, so we trained the model with different pretrained ResNet-18:
- Places365, a dataset with different scenes
- Google Landmarks Dataset V2, a dataset with different human made and natural landmarks

Because of the only difference between the backbones are the pretrained weights, there is no need to change the code in order to use them, just the link from where to download the weights.

## Conclusions
ImageNet is the best pretrained model for this task, other pretrained models have lower recalls and just in R@10 and R@20 of Tokyo datasets they have a little improvement with respect to ImageNet.
We can conclude that ImageNet is the best pretrained ResNet-18 for this task.

# Use of different backbones
In order to analyze the effect of the backbone on the results, we have trained the model with different backbones pretrained on ImageNet:
- ResNet-18
- EfficientNet B0, B1, B2
- EfficentNet V2 S
- MobileNet V3 Small, Large
- ConvNext Tiny
- Swin Tiny

In order to speed up the training process, we set AMP to true for all tests, the only exception is ResNet-18 for which we trained it also with AMP set to false.

For all of the test we used the Cosface loss function.

## Why these backbones?
This backbones have been chosen for different reasons:
- ResNet-18 is the default backbone of the project, so we have to consider it
- EffiecientNets and MobileNets are networks that have a good tradeoff between accuracy and computational resources usage, that is crucial for our computational resources
- ConvNext is a 2022 network, so we decided to consider it in order to see if a new network can be better than the others
- Swin Tiny is a Transformer network and we decided to consider it because is a different type of network from the others

## Backbone freeze and truncation
Each backbone has a different number of layers and of output channels, so we have to consider how to use them in our model.

Starting from the original code, we understand which part of the backbone we have to freeze and which part we have to truncate. Generally we keep the feature extraction part of the backbone and we truncate the other parts, then we didn't freeze last layers of the backbone in a different way for each backbone.

## Results analysis
In this analysis we take into account:
- Training time
- R@1, R@5, R@10, R@20
- Test datasets: SF-XS, Tokyo-XS, Tokyo-Night

Technically we should take into acount also other factors like the complexity of the network, hyperparameter tuning, the resources usage, but due to the limited computational resources we have, we can't take into account these factors.

Our reference is ResNet-18 with AMP set to false, that is the default backbone of the project.

### Training time
*We calculated training time considering the effective training time and the validation time inside the training loop.*

MobileNet V3 Small is the fastest one, but it's the one with worst recalls, then there are MobileNet V3 Large and ResNet-18 (with AMP ON) that have the same times, with little difference in the recalls.

All other networks are slower, but we have to consider also that networks like EfficientNet V2 S, ConvNext Tiny and Swin Tiny have a complexity that is comparable to ResNet-50 that we didn't test because of the computational resources.

Swin Tiny is the slowest one, but we have to consider that it should be compared with ResNet-50 and not with ResNet-18.

### R@1, R@5, R@10, R@20
We can see that EfficientNet B0, B1, B2 MobileNet V3 Small have lower recalls, so we could say that they are not good for this task.

#### EfficientNet V2 S
The best results are obtained with EfficientNet V2 S, with an important difference with the other networks and with the default backbone.  This backbone provides a general boost in all the recalls expecially for R@1 and R@5.

It's interesting to see how much the performances improve in Tokyo-Night that is the most difficult one because query images are taken in the night and the database images are taken in the day.

Anyway the number Tokyo Night queries is very low so we should consider this result with caution and focus instead on Tokyo XS where there are all of the queries and the database images of Tokyo Night but also other queries.

Talking about SF-XS where the dataset is more populated, we can see the same trend of the other networks, keeping also a good improvement with respect to the default backbone.

## Conclusions
We can conclude that EfficientNet V2 S is the best backbone for this task, but we have to consider that it's more complex than the others and it's not the fastest one. We should try to train it for more epoches in order to see how losses and recalls evolve, then we should also do a proper hyperparameter tuning and also verify different configurations of the backbone.

Talking about the task itself, we have to consider that these dataset are smaller version of the original ones, so we can't be sure that the results are the same for the original ones. We should also try to train the model with the original datasets and compare the results.

# Notes
It's important to consider that our computational resources are limited, so we have trained models only for 3 epoches and 10'000 iterations for each epoch.
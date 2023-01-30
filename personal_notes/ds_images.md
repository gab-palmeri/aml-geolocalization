# Image size differences
As we know the images from the database for each domain are from google images and they have the same size.
*The following are in the format (width x height)*
- Tokyo Database: 640x480
- SF-XS (test) Database: 512x512

It's not the same for images from queries that are from different sources and they have different sizes.
## Tecniques
First we need to resize the images from the queries to the same size as the images from the database.
We will call this method: "single query", because like before the batch size is 1. And this is a good way to manage the images from the queries at inference time, but it's not scalable, as explained by **paper deepvisual**.

So we will apply different data augmentation techniques in order to manage more than one image at inference time.
Some of them requires both pre-processing and post-processing.

- Single Query
  - Parallelization: No
  - Pre processing: Resize the images from the queries to the same size as the images from the database
  - Post processing: No
- Central Crop
  - Parallelization: Yes
  - Pre processing: Central Crop of the image to the same size as the images from the database
  - Post processing: No
-  Five Crops
   -  Parallelization: Yes
   -  Pre Processing: Take 5 crops of the image (4 corners and the center) and resize them to a square crop of the smallest dimension of the image. Then do the the mean of the descriptors obtained
   -  Post Processing: No
- Nearest Crop
  - Parallelization: Yes
  - Pre processing: Take 5 crops of the image (4 corners and the center) and resize them to a square crop of the smallest dimension of the image.
  - Post processing: Take the crop that is the nearest to the right image
- Majority Voting
  - Parallelization: Yes
  - Pre processing: Take 5 crops of the image (4 corners and the center) and resize them to a square crop of the smallest dimension of the image.
  - Post processing: Voting mechanism taking into account the distances from each crop's first 20 predictions
- Five Custom
  - Parallelization: Yes
  - Pre processing:
    - Central Crop of original image
    - 4 * Crop obtained through a compose of random horizontal flip, random perspective and a central crop of the image
  - Post processing: Take the crop that is the nearest to the right image

## Results

In our case, we decided to test this methods to predictions provided by GeoLocalizationNet with backbone ResNet18 and EfficientNet V2 Small
### Resnet
We can see that methods with post processing generally have better results, with different percentage of improvement depending on the the dataset and the method.

It's interesting to see that Central Crop that is an easy method to implement that allows us to manage more than one image at inference time reducing the inference time, has similar results to the single query method and the original results, so we understand that we can use parallelization in the inference time with a low cost in performance.

### EfficientNet V2 Small
Like Resnet, the behaviours are similar, but results are tipically worse compared to original resulst and single query method.

## Conclusions
Parallelization is a good way to manage more than one image at inference time, reducing the inference time, but not all techniques perform better than the single query method and with lower time of execution.

Thinking about a real-scenario with a lot of images to process, we should adopt a technique like central crop that let us to inference time with a low cost in performance.

The following are the ratio of Central Crop's time, R@1 and R@5 compared to the original time, R@1 and R@5.

| Backbone          |   Dataset   | Time / Original Time | R@1 / Original R@1 | R@5 / Original R@1 |
| :---------------- | :---------: | :------------------: | :----------------: | :----------------: |
| ResNet18          |    SF-XS    |         0.57         |        0.90        |        0.93        |
| ResNet18          |  Tokyo-XS   |         0.83         |        1.00        |        0.99        |
| ResNet18          | Tokyo-Night |         1.00         |        1.01        |        0.99        |
| EfficientNet V2 S |    SF-XS    |         0.39         |        0.90        |        0.93        |
| EfficientNet V2 S |  Tokyo-XS   |         0.50         |        0.96        |        0.96        |
| EfficientNet V2 S | Tokyo-Night |         1.00         |        0.92        |        0.97        |
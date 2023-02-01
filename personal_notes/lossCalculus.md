# Loss functions analysis
CosPlace uses CosFace loss function, our **goal** is to replace it with SphereFace and ArcFace and see how the results changes.

# OpenSphere
Looking online we found an interesting library called [OpenSphere](https://opensphere.world/) which implements [CosFace](https://github.com/ydwen/opensphere/blob/main/model/head/cosface.py), [SphereFace](https://github.com/ydwen/opensphere/blob/main/model/head/sphereface.py) and [ArcFace](https://github.com/ydwen/opensphere/blob/main/model/head/arcface.py) loss functions for PyTorch. One of the most interesting things about this library is that one of the github contributors is one of the authors of the original CosFace paper: Wei Liu.

Thanks to this library we should be able to easily replace CosFace with SphereFace and ArcFace.

Analyzing CosFace functions we noticed some difference between the implementations.
### CosPlace's CosFace
The original CosPlace [code](https://github.com/gmberton/CosPlace) uses an implementation of CosFace loss function based on MuggleWang's [code](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py).

### Files
For reference, here are the files we are talking about:
- [CosPlace cosface_loss.py](cosface_loss.py)
- [OpenSphere cosface.py](cosface.py)
- [OpenSphere sphereface.py](sphereface.py)
- [OpenSphere arcface.py](arcface.py)

## Differences between CosPlace's CosFace and OpenSphere's CosFace

### Parameters
The first difference is in the default values of the parameters.
| Parameter | CosPlace | OpenSphere |
| --- | --- | --- |
| s | 30.0 | 64.0 |
| m | 0.40 | 0.35 |

### CosPlace implementation
```python
def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)
# ...
def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    cosine = cosine_sim(inputs, self.weight)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, label.view(-1, 1), 1.0)
    output = self.s * (cosine - one_hot * self.m)
    return output
```
### OpenSphere implementation
```python
# Previous code omitted
    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss
```
### Cosine similarity
The way to calculate the cosine similarity is different, but the output / logits are calculated in the same way.
#### Why?
Let's consider a simple example:
```python
>>> m = 0.40
>>> one_hot = torch.zeros(3,5).scatter_(1, torch.tensor([[0,1,2,0]]), 1.0)
>>> one_hot
tensor([[1., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.]])
>>> - one_hot * m
tensor([[-0.4000,  0.0000,  0.0000, -0.4000,  0.0000],
        [ 0.0000, -0.4000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.4000,  0.0000,  0.0000]])

>>> d_theta = torch.zeros(3,5).scatter_(1, torch.tensor([[0,1,2,0]]), -m, reduce='add')
>>> d_theta
tensor([[-0.4000,  0.0000,  0.0000, -0.4000,  0.0000],
        [ 0.0000, -0.4000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.4000,  0.0000,  0.0000]])
```
It follows that output/logits are calculated in the same way:
$s\cdot(cosine-one_{hot} * m) = s\cdot(cosine+d_{\theta})$  
### Loss function

CosPlace return the output/logits, while OpenSphere returns the loss obtained with the CrossEntropy loss function.

Notice that obviously also the loss is calculated in the CosPlace code, but it's done in the training loop.
As you can see here:
```python
# Previous code omitted
if not args.use_amp16:
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            # Calculate loss
            loss = criterion(output, targets)
            loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                # Calculate loss
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
# Next code omitted
```
# Conclusion
In CosPlace we calculate the loss in the training loop, while in OpenSphere the loss is calculated in the forward function. Watching training code `classifier.forward()` is called always in this way:
```python
output = classifiers[current_group_num](descriptors, targets)
loss = criterion(output, targets) 
```
where `criterion` is the cross entropy loss function adopted by OpenSphere inside the forward function:
```python
criterion = torch.nn.CrossEntropyLoss()
```

Considering that there are not critical differences, because we can easily change from one implementation to the other just considering a different point of loss function calculation.

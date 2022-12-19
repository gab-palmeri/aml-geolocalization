# Difference between two cross entropy adopted

## CosPlace
In CosPlace loss is calculated in the training loop with the following code:
```python
import torch
# Previous code omitted
criterion = torch.nn.CrossEntropyLoss()
# ...
output = classifiers[current_group_num](descriptors, targets)
loss = criterion(output, targets)
# Next code omitted
```
## OpenSphere
In OpenSphere loss is calculated inside the respective `{loss_name}.py` file with the following code:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# Previous code omitted
logits = self.s * (cos_theta + d_theta)
loss = F.cross_entropy(logits, y)
# Next code omitted
```
## Why?
The main difference between two methods is that the first one has a state and the second one doesn't. Because of `criterion` is initialized withouth weights, it's not a weighted cross entropy, so there is **no difference between the two methods**.

### Source
[PyTorch Forum discussion](https://discuss.pytorch.org/t/f-cross-entropy-vs-torch-nn-cross-entropy-loss/25505)
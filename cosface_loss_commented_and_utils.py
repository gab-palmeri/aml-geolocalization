
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter

""" 
    Here is the explanation for the code above:
    1. torch.mm(x1, x2.t()) -> computes the inner product between x1 and x2
    2. torch.norm(x1, 2, dim) -> computes the norm of x1
    3. torch.ger(w1, w2).clamp(min=eps) -> computes the outer product between w1 and w2
    4. ip / torch.ger(w1, w2).clamp(min=eps) -> computes the cosine similarity between x1 and x2 
"""
def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


"""
CosPlace uses the Large Cosine similarity loss, also known as CosFace, which is a cosine based loss that adds a margin to the cosine function to better separate the decision boundaries between classes. 
"""

class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    """ Here is the explanation for the code above:
        1. cosine_sim: It calculates the cosine similarity between inputs (x) and the weights (W).
        2. one_hot: It is a tensor of zeros, with the same size as cosine. 
        3. one_hot.scatter_(1, label.view(-1, 1), 1.0) will make the label-th column of the one_hot to be 1.0, which is the same as the one-hot encoding of the label. scheme: "https://miro.medium.com/max/1400/0*T5jaa2othYfXZX9W."
        4. output = self.s * (cosine - one_hot * self.m) is the final output of the model, which is the cosine similarity between input feature and each class center, minus the margin of the label-th class center. 
    """
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

        cosine = cosine_sim(inputs, self.weight)

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(1, label.view(-1, 1), 1.0)


        output = self.s * (cosine - one_hot * self.m)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'




"""
This implementation of the ArcFace loss function calculates the angle between the embeddings and the weight vector, applies the margin penalty, and then calculates the softmax and the loss. The loss can then be minimized using gradient descent or another optimization method.
"""
def arcface_loss(embeddings, labels, s, m):
  # Calculate the cosine similarity between the embeddings and the weight vector
  cos_theta = torch.matmul(embeddings, weight_vector.t())
  
  # Calculate the angle between the embeddings and the weight vector
  theta = torch.acos(cos_theta)
  
  # Calculate the margin penalty term
  margin_penalty = torch.cos(theta + m)
  
  # Calculate the logits for each class
  logits = s * cos_theta
  
  # Calculate the logits for the true class
  logits_yi = s * margin_penalty[torch.arange(0, labels.size(0)), labels]
  
  # Calculate the logits for the other classes
  logits_ji = s * cos_theta[torch.arange(0, labels.size(0)), labels]
  
  # Calculate the logits for the true class and all other classes
  logits_yi_ji = torch.cat((logits_yi.unsqueeze(1), logits_ji), dim=1)
  
  # Calculate the logits for all classes
  logits_j = torch.cat((logits[:, :labels], logits[:, labels+1:]), dim=1)
  
  # Calculate the log of the softmax
  log_softmax = torch.log(torch.softmax(logits_yi_ji, dim=1))
  
  # Calculate the loss
  loss = -torch.sum(log_softmax) / labels.size(0)
  
  return loss



"""
This implementation of the SphereFace loss function calculates the angle between the embeddings and the weight vector, applies the margin penalty, and then calculates the softmax and the loss. The loss can then be minimized using gradient descent or another optimization method.
"""
def sphereface_loss(embeddings, labels, m):
  # Calculate the cosine similarity between the embeddings and the weight vector
  cos_theta = torch.matmul(embeddings, weight_vector.t())
  
  # Calculate the angle between the embeddings and the weight vector
  theta = torch.acos(cos_theta)
  
  # Calculate the margin penalty term
  margin_penalty = torch.cos(m * theta)
  
  # Calculate the logits for each class
  logits = cos_theta
  
  # Calculate the logits for the true class
  logits_yi = margin_penalty[torch.arange(0, labels.size(0)), labels]
  
  # Calculate the logits for the other classes
  logits_ji = cos_theta[torch.arange(0, labels.size(0)), labels]
  
  # Calculate the logits for the true class and all other classes
  logits_yi_ji = torch.cat((logits_yi.unsqueeze(1), logits_ji), dim=1)
  
  # Calculate the logits for all classes
  logits_j = torch.cat((logits[:, :labels], logits[:, labels+1:]), dim=1)
  
  # Calculate the log of the softmax
  log_softmax = torch.log(torch.softmax(logits_yi_ji, dim=1))
  
  # Calculate the loss
  loss = -torch.sum(log_softmax) / labels.size(0)
  
  return loss


#method that calculates the general angular margin penalty-based loss
def angular_margin_loss(embeddings, labels, s, m):
    # Calculate the cosine similarity between the embeddings and the weight vector
    cos_theta = torch.matmul(embeddings, weight_vector.t())
    
    # Calculate the angle between the embeddings and the weight vector
    theta = torch.acos(cos_theta)
    
    # Calculate the margin penalty term
    margin_penalty = torch.cos(theta + m)
    
    # Calculate the logits for each class
    logits = s * cos_theta
    
    # Calculate the logits for the true class
    logits_yi = s * margin_penalty[torch.arange(0, labels.size(0)), labels]
    
    # Calculate the logits for the other classes
    logits_ji = s * cos_theta[torch.arange(0, labels.size(0)), labels]
    
    # Calculate the logits for the true class and all other classes
    logits_yi_ji = torch.cat((logits_yi.unsqueeze(1), logits_ji), dim=1)
    
    # Calculate the logits for all classes
    logits_j = torch.cat((logits[:, :labels], logits[:, labels+1:]), dim=1)
    
    # Calculate the log of the softmax
    log_softmax = torch.log(torch.softmax(logits_yi_ji, dim=1))
    
    # Calculate the loss
    loss = -torch.sum(log_softmax) / labels.size(0)
    
    return loss
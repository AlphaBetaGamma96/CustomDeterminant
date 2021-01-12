import os
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = "1"

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.functional import hessian
from torch import Tensor
from torchviz import make_dot

#torch.manual_seed(0) #fix RNG seed
#torch.autograd.set_detect_anomaly(True)

def gamma(S):
  """
  Creates a matrix (denoted gamma) which is equivalent to det(Sigma)Sigma^-1
  if Sigma is a diagonal matrix diag(a, b, c, d) then gamma is,

        [a 0 0 0]                        [bcd 0 0 0]
        [0 b 0 0]          ------>       [0 acd 0 0]
        [0 0 c 0]                        [0 0 abd 0]
        [0 0 0 d]                        [0 0 0 abc]
  if the elements of Sigma are sigma_i then the elements of Gamma are defined as
  gamma_j = prod_{j != i} sigma_i

  input, S (type Tensor) is vector of length N representing the diagonal of the
  Sigma matrix from a Single-Value Decomposition
  """
  if(len(S.shape)==2):
    B, N = S.shape                                          #grab batch, and size
  elif(len(S.shape)==1):
    B = 1
    N = S.shape[-1]
  else:
    raise ValueError("")
  Stile = S.repeat(1,1,N).reshape(B,N,N)                  #tile the input matrix
  mask = (1. - torch.eye(N)).repeat(B,1,1).reshape(B,N,N)  #create mask for all batch
  gamma = torch.prod(Stile*mask + (1-mask), dim=-1)
  return gamma

def rho(S):
  """
  Creates a matrix (denoted rho) whose elements are defined from the diagonal
  of the Sigma matrix. Where element rho_ij = prod_{k != i, k != j} sigma_k which
  visually looks like,

        [a 0 0 0]                        [bcd cd bd bc]
        [0 b 0 0]          ------>       [cd acd ad ac]
        [0 0 c 0]                        [bd ad abd ab]
        [0 0 0 d]                        [bc ac ab abc]

  input, S (type Tensor) is vector of length N representing the diagonal of the
  Sigma matrix from a Single-Value Decomposition
  """
  if(len(S.shape)==2):
    B, N = S.shape                                   #get batch, and size
  elif(len(S.shape)==1):
    B = 1
    N = S.shape[-1]
  else:
    raise ValueError("")
  Stile = S.repeat(1,1,N*N).reshape(B,N,N,N)       #tile the matrix via repeat/reshape

  mask1 = (1-torch.eye(N)).repeat(B,N,1,1).reshape(B,N,N,N).transpose(-2,-1)
  mask2 = (1-torch.eye(N)).repeat(B,N,1,1).reshape(B,N,N,N).transpose(-3,-2)
  mask = mask1 * mask2
  rho_matrix = torch.prod(Stile*mask + (1-mask), dim=-1)
  return rho_matrix

class CustomDeterminant(Function):

  """
  Custom Determinant function f: R^NxN -> R with custom backward and
  backward-of-backward function calls to ensure stability during training in case
  of a singular matrix and zero values within the Sigma matrix (which emerges
  from a Single-Value Decomposition of the input matrix)
  """

  @staticmethod
  def forward(ctx, A):
    ctx.save_for_backward(A)
    U, S, V = torch.svd(A) #U=[1, N, N] S=[1, N] V=[1, N, N]
    det = torch.det(U)*torch.det(torch.diag_embed(S))*torch.det(torch.transpose(V, -2, -1))
    # det=[1, N, N]
    return det

  @staticmethod
  def backward(ctx, Detbar):
    A, = ctx.saved_tensors #remember comma to unpack tuple
    return CustomDeterminantBackward.apply(A, Detbar)

class CustomDeterminantBackward(Function):

  @staticmethod
  def forward(ctx, A, Detbar):
    """
    Calculates dL/dA for a determinant node via Single-Value Decomposition
    """
    ctx.save_for_backward(A, Detbar)
    U, S, V = torch.svd(A) #U=[1, N, N] S=[1, N] V=[1, N, N]
    Abar = Detbar * torch.det(U) * torch.det(torch.transpose(V, -2, -1)) * U @ torch.diag_embed(gamma(S)) @ torch.transpose(V, -2, -1)
    #Abar=[1, N, N]
    return Abar

  @staticmethod
  def backward(ctx, Cbar):
    """
    Calculates d^2 L/dA^2 for a determinant node via Single-Value Decomposition
    """
    A, Detbar = ctx.saved_tensors #Shapes, A=[1, N, N] Detbar=[1]
    U, S, V = torch.svd(A)                 #U=[1, N, N] S=[1, N] V=[1, N, N]
    M = torch.transpose(V, -2, -1) @ torch.transpose(Cbar, -2, -1) @ U #M=[1,N,N], Cbar=[1,N,N]
    diag_M = torch.diagonal(M, offset=0, dim1=-2, dim2=-1)             #diag_M = [1,N]
    rho_matrix = rho(S)                           #rho_matrix = [1, N, N]
    mask_off_diagonal = (1-torch.eye(M.shape[-1])).reshape(-1,M.shape[-1],M.shape[-1]) #mask_off_diagonal [1, N, N]
    diag_Xi = torch.sum((diag_M * rho_matrix)*mask_off_diagonal, dim=-1) #diag_Xi [1, N]
    masked_presum_Xi = diag_M*rho_matrix*mask_off_diagonal #masked_presum_Xi = [1, N, N]
    non_diag_Xi = (-M*rho_matrix)*(1 - torch.eye(M.shape[-1])) #non_diag_Xi = [1, N ,N]
    Xi = torch.diag_embed(diag_Xi) + non_diag_Xi               #Xi = [1, N ,N]
    Abar = Detbar * torch.det(U) * torch.det(torch.transpose(V, -2, -1)) *  U @ Xi @ torch.transpose(V, -2, -1)
    DetBarBar = torch.det(U) * torch.det(torch.transpose(V, -2, -1)) * Cbar @ U @ torch.diag_embed(gamma(S)) @ torch.transpose(V, -2, -1)
    #shapes of Abar=[1, N, N] DetBarBar=[1, N, N]
    return Abar, DetBarBar

class MyDeterminant(nn.Module):

  def __init__(self):
    """
    An nn.Module for my own Custom determiant function
    """
    super(MyDeterminant, self).__init__()

  def forward(self, A):
    return CustomDeterminant.apply(A)

class PyTorchDeterminant(nn.Module):

  def __init__(self):
    """
    An nn.Module for PyTorch's determiant function
    """
    super(PyTorchDeterminant, self).__init__()

  def forward(self, A):
    return torch.det(A)

class myNetwork(nn.Module):

  def __init__(self, ninputs, nhidden):
    """
    A simple network that takes a vector of size, N, creates a matrix of size [N, N] via a
    hidden layer and takes that determinant to return a scalar value
    """
    super(myNetwork, self).__init__()

    self.fc = nn.Linear(2, nhidden)             #fully connected layer
    self.af = nn.Softplus()                     #activation function
    self.weight_matrix = nn.Parameter(torch.randn(nhidden, ninputs))   #weight and bias
    self.bias_vector = nn.Parameter(torch.randn(ninputs))              #to make our input matrix

    #self.slater_det = PyTorchDeterminant()             #PyTorch's implementation
    self.slater_det = MyDeterminant()        #Custom implementation

  def forward(self, x):
    g = x.mean(keepdim=True, dim=-1) #mean over dim=1, g size [B,1]

    g = g.repeat(1, x.shape[1])     #repeat to [B,N]
    f = torch.stack((x,g),dim=2)    #stack so f has shape [B, N, 2]

    layer_output = self.af(self.fc(f))    #[B, N, 2] @ [2, H] = [B, N, H]
    matrix = layer_output@self.weight_matrix + self.bias_vector #[B, N, H] @ [H, N] + [N] = [B, N, N]

    y = self.slater_det(matrix)  #det([B, N, N]) = [B,]
    return y

def calc_loss(Net, x: Tensor):
  y = Net(x)
  laplacian = torch.zeros(x.shape[0])
  for i, xi in enumerate(x):
    yi = Net(xi.unsqueeze(0))
    hess = hessian(Net, xi.unsqueeze(0), create_graph=True)
    laplacian[i] = hess.view(x.shape[1], x.shape[1]).diagonal(offset=0).sum()
  loss1 = (laplacian/y) #calculate loss1 (including laplacian term)

  loss2 = torch.sum(x**2, dim=-1) #calculate loss2 (including another loss)
  return torch.mean(loss1 + loss2) #add and reduce mean

B, N, H = 25, 4, 16 #batch, size of input vector, size of hidden nodes

myNet = myNetwork(ninputs=N, nhidden=H) #make the network
myX = torch.randn(B,N)                  #input tensor of shape [B, N]

loss = calc_loss(myNet, myX)    #calculate our loss

loss.backward()                 #backprop

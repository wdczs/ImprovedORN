import re
import torch
from torch.autograd import Function
from .utils import FunctionBackend
from .._ext import liborn
import torch.nn.functional as F
import numpy as np

class ORAlign1d(Function):

    def __init__(self, nOrientation, return_direction=False):
        super(ORAlign1d, self).__init__()
        self.backend = FunctionBackend(liborn)
        self.nOrientation = nOrientation
        self.return_direction = return_direction

    def forward(self, input):
        mainDirection, output = input.new().byte(), input.new()
        self.backend.set_type(input.type())
        self.backend.RIE_AlignFeature(
            input, 
            mainDirection, 
            output, 
            self.nOrientation)

        if self.return_direction:
            self.save_for_backward(input, mainDirection)
            self.mark_non_differentiable(mainDirection)
            return output, mainDirection
        else:
            self.save_for_backward(input)
            self.mainDirection = mainDirection
            return output

    def backward(self, grad_output):
        if self.return_direction:
            input, mainDirection = self.saved_tensors
        else:
            input, = self.saved_tensors
            mainDirection = self.mainDirection

        grad_input = input.new()
        self.backend.RIE_UnAlignFeature(
            grad_input, 
            mainDirection, 
            grad_output,  
            self.nOrientation)
        return grad_input

class ORAlign2d(Function):

    def __init__(self, nOrientation, return_direction=False):
        super(ORAlign2d, self).__init__()
        self.backend = FunctionBackend(liborn)
        self.nOrientation = nOrientation
        self.return_direction = return_direction

    def forward(self, input):
        mainDirection, output = input.new().byte(), input.new()
        self.backend.set_type(input.type())
        self.backend.RIE_AlignFeature2d(
            input, 
            mainDirection, 
            output, 
            self.nOrientation)

        if self.return_direction:
            self.save_for_backward(input, mainDirection)
            self.mark_non_differentiable(mainDirection)
            return output, mainDirection
        else:
            self.save_for_backward(input)
            self.mainDirection = mainDirection
            return output

    def backward(self, grad_output):
        if self.return_direction:
            input, mainDirection = self.saved_tensors
        else:
            input, = self.saved_tensors
            mainDirection = self.mainDirection

        grad_input = input.new()
        self.backend.RIE_UnAlignFeature2d(
            grad_input, 
            mainDirection, 
            grad_output,  
            self.nOrientation)
        return grad_input



# class ORAlign2d(Function):

#     def __init__(self, nOrientation, return_direction=False):
#         super(ORAlign2d, self).__init__()
#         self.backend = FunctionBackend(liborn)
#         self.nOrientation = nOrientation
#         self.return_direction = return_direction

#     def forward(self, input):
#         feature_h = input.size(2)
#         feature_w = input.size(3)
#         nOrientation = self.nOrientation

#         input_squeeze = F.avg_pool2d(input, feature_h)
#         input_squeeze_tensor = input_squeeze.data
#         mainDirection, output_squeeze_tensor = input_squeeze_tensor.new().byte(), input_squeeze_tensor.new()

#         self.backend.set_type(input_squeeze_tensor.type())
#         self.backend.RIE_AlignFeature(
#             input_squeeze_tensor, 
#             mainDirection, 
#             output_squeeze_tensor, 
#             nOrientation)
        
#         nBatch = mainDirection.size(0)
#         nFeature = mainDirection.size(1)
#         # output = torch.Tensor(input.size())
#         output_np = np.zeros([nBatch, nFeature*nOrientation,feature_h,feature_w]) 

#         for i in range(nBatch): # 128
#             for j in range(nFeature): # nChannel / nOrientation = 640 / 8 = 80
#                 direction = mainDirection[i][j]
#                 for l in range(nOrientation):
#                     alignedIndex = (l - direction + nOrientation) % nOrientation
#                     for p in range(feature_h):
#                         for q in range(feature_w):
#                             output_np[ i ][ j * nOrientation + alignedIndex][p][q] = input[ i ][ j * nOrientation + l][p][q]                            

#         output = torch.Tensor(output_np).cuda()
#         if self.return_direction:
#             self.save_for_backward(input, mainDirection)
#             self.mark_non_differentiable(mainDirection)
#             return output, mainDirection
#         else:
#             self.save_for_backward(input)
#             self.mainDirection = mainDirection
#             return output

#     def backward(self, grad_output):
#         if self.return_direction:
#             input, mainDirection = self.saved_tensors
#         else:
#             input, = self.saved_tensors
#             mainDirection = self.mainDirection  
#         feature_h = input.size(2)
#         feature_w = input.size(3)        
#         nBatch = list(mainDirection.size())[0];
#         nFeature = list(mainDirection.size())[1];
#         nOrientation = self.nOrientation
#         grad_input_np = np.zeros([input.size(0), input.size(1), input.size(2), input.size(3)])

#         for i in range(nBatch): # 128
#             for j in range(nFeature): # nChannel / nOrientation = 640 / 8 = 80
#                 direction = mainDirection[i][j]
#                 for l in range(nOrientation):
#                     alignedIndex = (l + direction) % nOrientation
#                     for p in range(feature_h):
#                         for q in range(feature_w):
#                             grad_input_np[ i ][ j * nOrientation + alignedIndex][p][q] = grad_output[ i ][ j * nOrientation + l][p][q]

#         grad_input = torch.Tensor(grad_input_np).cuda()
#         return grad_input
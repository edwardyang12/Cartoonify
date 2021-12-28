import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import random

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0.0)

# https://github.com/sanghviyashiitb/GANS-VanillaAndMinibatchDiscrimination/blob/master/minibatch_discrimination.py
# reduce mode collapse by appending similarity value to discriminated outputs
class MiniBatchDiscrimination(nn.Module):
	def __init__(self, A, B, C, batch_size):
		super(MiniBatchDiscrimination, self).__init__()
		self.feat_num = A
		self.out_size = B
		self.row_size = C
		self.N = batch_size
		self.T = Parameter(torch.Tensor(A,B,C)).to("cuda")
		self.reset_parameters()

	def forward(self, x):
		# Output matrices after matrix multiplication
		M = x.mm(self.T.view(self.feat_num,self.out_size*self.row_size)).view(-1,self.out_size,self.row_size)
		out = Variable(torch.zeros(self.N,self.out_size), requires_grad=True).to("cuda")
		for k in range(self.N): # Not happy about this 'for' loop, but this is the best we could do using PyTorch IMO
			c = torch.exp(-torch.sum(torch.abs(M[k,:]-M),2)) # exp(-L1 Norm of Rows difference)
			if k != 0 and k != self.N -1:
				out[k,:] = torch.sum(c[0:k,:],0) + torch.sum(c[k:-1,:],0)
			else:
				if k == 0:
					out[k,:] = torch.sum(c[1:,:],0)
				else:
					out[k,:] = torch.sum(c[0:self.N-1],0)
		return out

	def reset_parameters(self):
		stddev = 1/self.feat_num
		self.T.data.uniform_(stddev)

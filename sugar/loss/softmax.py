# -*- encoding: utf-8 -*-

import torch.nn as nn

class Softmax(nn.Module):
	def __init__(self, nOut, nClasses):
	    super(Softmax, self).__init__()

	    self.test_normalize = True
		
	    self.fc = nn.Linear(nOut, nClasses)

	    print('Initialized Softmax Loss without CrossEntropyLoss')

	def forward(self, x, label=None):
		x = self.fc(x)
		return x
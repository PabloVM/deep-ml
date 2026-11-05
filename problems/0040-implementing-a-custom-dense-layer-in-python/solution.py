
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

	def initialize(self, optimizer):
		# Initialize weights W, biases w0, and optimizers
		limit = 1 / np.sqrt(self.input_shape[0])
		self.W = np.random.uniform(-limit,limit,(self.input_shape[0],self.n_units))
		self.w0 = np.zeros((1,self.n_units))
		self.w_optimizer = copy.copy(optimizer)
		self.w0_optimizer = copy.copy(optimizer)

	def parameters(self):
		# Return total number of parameters
		return len(self.W) + len(self.w0)

	def forward_pass(self, X, training=True):
		
		return X @ self.W + self.w0

	def backward_pass(self, accum_grad):
		# Compute gradients, update weights if trainable, return grad w.r.t. input
		grad = accum_grad @ self.W.T
		if self.trainable:
			self.W = self.w_optimizer.update(self.W,X.T @ accum_grad)
			self.w0 = self.w0_optimizer.update(self.w0,np.sum(accum_grad,axis=0))
		return grad

	def output_shape(self):
		# Return output shape tuple
		return (self.n_units,)

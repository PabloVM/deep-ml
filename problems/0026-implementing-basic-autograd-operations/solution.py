class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		other = other if isinstance(other,Value) else Value(other)
		out = Value(self.data + other.data, (self,other),"+")
		def _backward():
			self.grad += out.grad
			other.grad += out.grad
		out._backward = _backward
		return out

	def __mul__(self, other):
		other = other if isinstance(other,Value) else Value(other)
		out = Value(self.data * other.data,(self,other),"*")
		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = _backward
		return out

	def relu(self):
		out =  Value(self.data if self.data > 0.0 else 0.0,(self,),"ReLU")
		def _backward():
			self.grad += (out.data > 0.0) * out.grad
		out._backward = _backward
		return out

	def backward(self):
		order = []
		visited = set()
		def build_order(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_order(child)
				order.append(v)
		build_order(self)

		self.grad = 1.0

		for node in reversed(order):
			node._backward()
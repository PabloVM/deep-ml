# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        self.cache = {}
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        sigma = np.sqrt(2/input_size)
        self.W1 =sigma * np.random.randn(input_size,hidden_size)
        self.b1 = np.zeros((1,hidden_size))
        sigma = np.sqrt(1/hidden_size)
        self.W2 = sigma * np.random.randn(hidden_size,output_size)
        self.b2 =np.zeros((1,output_size))
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0,z1)
        z2 = h1 @ self.W2 + self.b2
        z2_max = np.max(z2,axis=1,keepdims=True)
        h2 = np.exp(z2 - z2_max) / np.sum(np.exp(z2-z2_max),axis=1,keepdims=True)
        
        self.cache = {'X': X, 'z1': z1, 'h1': h1, 'z2': z2}
        
        return h2
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        N = X.shape[0]
        # 1. Compute loss
        loss = -np.mean(np.log(probs[np.arange(X.shape[0]), y] + 1e-12))
        # 2. Compute gradient of loss w.r.t. softmax output
        dlz2 = probs.copy()
        dlz2[np.arange(N),y] -= 1
        dlz2 /=N
        # 3. Backprop through linear layer 2
        dlW2 = self.cache["h1"].T @ dlz2
        dlb2 = np.sum(dlz2,axis=0,keepdims=True)
        dlh1 = dlz2 @ self.W2.T
        # 4. Backprop through ReLU
        z1 = self.cache["z1"]
        dlz1 = dlh1 * (z1>0)
        # 5. Backprop through linear layer 1        
        dlW1 = X.T @ dlz1
        dlb1 = np.sum(dlz1,axis=0,keepdims=True)
        # 6. Update all parameters using gradients
        self.W1 -= self.lr * dlW1
        self.b1 -= self.lr * dlb1
        self.W2 -= self.lr * dlW2
        self.b2 -= self.lr * dlb2
        
        return loss
        
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
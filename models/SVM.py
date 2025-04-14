import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.hessian = np.zeros((n_features,n_features))
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        

        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                g = (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                for i in range(g.shape[0]):
                    self.hessian[i] = g - g[i]
                self.w -= self.lr*self.hessian*g
                self.b -= self.lr * y_[idx]
                # if condition:
                #     self.w -= self.lr * (2 * self.lambda_param * self.w)
                # else:
                #     self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                #     self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
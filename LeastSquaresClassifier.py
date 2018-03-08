import numpy as np
import pandas as pd

class LeastSquaresClassifier(object):
    """Linear least squares classifier with l2 regularization
    
    Parameters
    ----------
    X : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data
    
    y : array-like, shape = [n_samples]
        Training targets
        
    lambda_par : float
        Regularization strength
    
    """
    def __init__(self, X, y, lambda_par=0):
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices = False)
        self.y = y
        self.lambda_par = lambda_par

    def train(self):
        """Train the Least Squares model
        
        Find the weights for each feature
        """
        s_inv = np.diag(list(map((lambda x: x / (self.lambda_par + x**2)), self.S)))
        self.w_params = np.dot(np.dot(np.dot(self.V.T, s_inv), self.U.T), self.y)
        
    def predict(self, x_test):
        """Predict the class for every point in the test set

        Parameters
        ----------
        x_test : array-like, shape = [n_samples, n_features]
            Test data
            
        Returns
        ----------
        pred : array-like, shape = [n_samples]
            Predictions made by the classifier   
        """
        pred = np.sign(np.dot(x_test, self.w_params))
        return pred
    
    def getParams(self):
        """Return the weights

        Returns
        ----------
        w_params : array-like, shape = [n_features]
            Weights for each feature   
        """
        return self.w_params
    
    def reverse_w(self):
        """Reverse the weights
        
        For boosting when the error term > 0.5
        """
        self.w_params *= -1   
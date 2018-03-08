import numpy as np
import pandas as pd

class AdaBoost(object):
    """AdaBoost Ensemble Classifier
    
    Parameters
    ----------
    x_train : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data
    
    y_train : array-like, shape = [n_samples]
        Training targets
        
    classifier : callable
        Classifier to boost
    
    """
    def __init__(self, x_train, y_train, classifier):
        self.X = x_train
        self.y = y_train
        self.classifier = classifier
        self.probabilities = pd.DataFrame(np.ones(len(x_train))/len(x_train))
        
        self.errors = []
        self.alphas = []
        self.training_errors = []
        self.testing_errors = []
        self.classifiers = []
        self.upper_bounds = []
        
        self.all_samples = []
        
    def update_probabilities(self, alpha, predictions):
        """Update the probabilities given the new alpha and predictions

        Parameters
        ----------
        alpha : float
            alpha parameter
            
        predictions : array-like, shape = [n_samples]
            predicted classes

        """
        self.probabilities *= np.exp(-alpha*self.y*predictions)
        self.probabilities /= self.probabilities.sum()
            
    def compute_error(self, predictions):
        """Compute the error value

        Parameters
        ----------
        predictions : array-like, shape = [n_samples]
            predicted classes
            
        Returns
        ----------
        error : float
            adaboost error term  
        """
        misclassified = predictions != self.y
        error = float((misclassified * self.probabilities).sum())
        return error
    
    def compute_alpha(self, error):
        """Compute the alpha value given the error

        Parameters
        ----------
        error : float
            adaboost error term 
            
        Returns
        ----------
        alpha : float
            adaboost alpha term  
        """
        alpha = 0.5*np.log((1-error)/error)
        return alpha
    
    def append_error(self, error_type, sum_val, y_test = None):
        """Predict and store the errors for later analysis

        Parameters
        ----------
        error_type : string
            Specifiy whether training or testing error
            
        sum_val : float
            Accumulated sum over train or testing predictions scaled by alpha
            
        y_test : array-like, shape = [n_samples]
            Testing targets
        """
        boosted_prediction = np.sign(sum_val)
        if error_type == 'training':
            training_error = (boosted_prediction != self.y).sum()/len(self.y)
            self.training_errors.append(training_error)
        elif error_type == 'testing':
            testing_error = (boosted_prediction != y_test).sum()/len(y_test)
            self.testing_errors.append(testing_error)
        
    def ada_boost(self, x_test, y_test, iterations):
        """Perform all parts of the Ada Boost algorithm

        Parameters
        ----------
        x_test : array-like, shape = [n_samples, n_features]
            Test data
            
        y_test : array-like, shape = [n_samples]
            Testing targets
            
        iterations : integer
            Number of iterations to run the boosted classifier
            
        """
        
        training_sum = 0
        boosted_sum = 0
        bound_sum = 0

        for t in range(iterations):
            bootstrap_sample_x = self.X.sample(n = len(self.X), replace = True, weights = self.probabilities[0])
            bootstrap_sample_y = self.y.iloc[bootstrap_sample_x.index]
            self.all_samples.extend(bootstrap_sample_x.index)
            
            model = self.classifier(bootstrap_sample_x, bootstrap_sample_y)
            model.train()
            
            predictions = model.predict(self.X)
            error = self.compute_error(predictions)
            
            if error > 0.5:
                model.reverse_w()
                predictions = model.predict(self.X)
                error = self.compute_error(predictions)
            
            alpha = self.compute_alpha(error)
            
            bound_sum += ((0.5-error)**2)
            self.upper_bounds.append(np.exp(-2*bound_sum))
            
            weighted_prediction_train = pd.DataFrame(predictions) * alpha
            training_sum += weighted_prediction_train
            self.append_error('training', training_sum)
            
            weighted_prediction_test = pd.DataFrame(model.predict(x_test)) * alpha
            boosted_sum += weighted_prediction_test
            self.append_error('testing', boosted_sum, y_test)
            
            self.update_probabilities(alpha, predictions)
            
            self.alphas.append(alpha)
            self.errors.append(error)
            self.classifiers.append(model)
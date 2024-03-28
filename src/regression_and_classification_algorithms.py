from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

class Regression_model:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
                
    def train(self,estimator, grid):
        model = GridSearchCV(estimator, grid)
        model.fit(self.train_X, self.train_y)
        print("Best parameters:",model.best_params_)
        return model
        
    def results(self, model):
        prob = model.predict(self.test_X)
        class_lbl = np.array(prob)
        class_lbl[prob>=0.5] = 1
        class_lbl[prob<0.5] = 0
        print('Mean Squared Error:',mean_squared_error(self.test_y, prob))
        print('Accuracy:', accuracy_score(self.test_y, class_lbl))
        return prob, class_lbl 
    
class Predictive_model(Regression_model):
    def __init__(self, train_X, train_y, test_X, test_y):
        super().__init__(train_X, train_y, test_X, test_y)
        
    def predictive_results(self, model):
        lbl = model.predict(self.test_X)
        print('Accuracy:', accuracy_score(self.test_y, lbl))
        return lbl 
import numpy as np
from river import optim
from river import linear_model
from river.tree import HoeffdingTreeRegressor

class BinaryTransformation():
    def __init__(self, n_labels,
                model=HoeffdingTreeRegressor(leaf_model=linear_model.LinearRegression(optimizer=optim.Adam(lr=1e-5), intercept_lr=0.5, loss=optim.losses.Squared()))
                ):
        self.n_labels = n_labels
        self.model = model
    
    def learn_one(self, X, y):
        # Transform the label vector into a real-valued integer and learn
        y_ = self.transform((list(y.values())))
        self.model.learn_one(X, y_)

    def predict_one(self, X):
        # Predict and transform the predicted float back to a label vector
        y_pred = np.round(self.model.predict_one(X)).astype(np.int32)
        y_pred = self.binarize(y_pred)
        return y_pred

    def transform(self, Y):
        Y_ = 0
        for bit in Y:
            Y_ = (Y_ << 1) | bit
        return Y_

    def binarize(self, Y):
        if Y > 2**self.n_labels - 1:
            y_ = [1] * self.n_labels
        y_ = np.array(list(map(int, list(np.binary_repr(Y, self.n_labels))))).astype(np.int32)
        if len(y_) > self.n_labels:
            y_ = np.ones(self.n_labels).astype(np.int32)
        return np.array(y_).astype(np.int32)
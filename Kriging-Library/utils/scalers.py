import numpy as np


class Scaler:
    def __init__(self, method=None, scale_value=None):
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        # elif method == 'minmax':
        #     self.scaler = MinMaxScaler()
        elif method == 'fixed':
            if scale_value is None:
                raise ValueError("scale_value must be provided for 'fixed' method")
            self.scaler = None
            self.scaler_value = scale_value
        else:
            raise ValueError("Unsupported method: {}".format(method))

    def fit(self, data):
        if self.scaler:
            self.scaler.fit(data)

    def transform(self, data):
        if self.method == 'fixed':
            return data / self.scaler_value
        elif self.scaler:
            return self.scaler.transform(data)

    def inverse_transform(self, data):
        if self.method == 'fixed':
            return data * self.scaler_value
        elif self.scaler:
            return self.scaler.inverse_transform(data)

    def fit_transform(self, data):
        if self.method == 'fixed':
            return self.transform(data)
        elif self.scaler:
            return self.scaler.fit_transform(data)
        return data


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X)
        self.scale_ = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """结合fit和transform方法。"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """将归一化的数据转换回原始尺度。"""
        return X_scaled * self.scale_ + self.mean_
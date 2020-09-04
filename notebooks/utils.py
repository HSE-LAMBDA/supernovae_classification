import numpy as np

def binarization(x, y, n=10, x_ranges=[0, 100]):
    
    new_y = []
    bins = np.linspace(x_ranges[0], x_ranges[1], n+1)
    for i_bin in range(n):
        left = bins[i_bin]
        right = bins[i_bin+1]
        y_sel = y[(x >= left) * (x < right)]
        ay = 0
        if len(y_sel) != 0:
            ay = np.mean(y_sel)
        new_y.append(ay)
    
    return new_y



def interpolation(x, y):
    
    y_iter = np.zeros(len(y))
    
    for i1 in range(len(x)-1):
        if y[i1] != 0:
            break
            
    for i2 in range(i1+1, len(x)):

        if y[i2] == 0:
            continue

        k = (y[i2] - y[i1]) / ((x[i2] - x[i1]))
        b = y[i2] - k * x[i2]

        for i in range(i1, i2+1):
            y_iter[i] = k * x[i] + b

        i1 = i2
        
    return y_iter

def rolling_diff(x):
    
    x_roll = np.zeros(len(x))
    for i in range(1, len(x)):
        x_roll[i] = x[i] - x[i-1]
        
    return x_roll


def rolling_ratio(x):
    
    x_roll = np.zeros(len(x))
    for i in range(1, len(x)):
        if x[i-1] != 0:
            x_roll[i] = x[i] / x[i-1]
        
    return x_roll

def log10_(x):
    
    x_roll = np.zeros(len(x))
    for i in range(0, len(x)):
        if x[i] != 0:
            x_roll[i] = np.log10(x[i])
        
    return x_roll

def get_bin_curves(row):
    aname = row.sn_name
    atype = row.type
    asize = row.size_r
    acurve = row.curve_r
    aline = []


    y_cor = acurve.y / acurve.y.max()
    x_cor = (acurve.x - acurve.x.min())
    x_cor = (acurve.x - acurve.x[y_cor.argmax()])

    new_y = binarization(x_cor, y_cor, n=16, x_ranges=[-50, 100])
    new_y = interpolation(np.arange(len(new_y)), new_y)
    #new_y = rolling_diff(new_y)
    new_y2 = rolling_ratio(new_y)
    # new_y2 = log10_(new_y2)
        
    aline += list(new_y) + list(new_y2) + [is_ia, row.name]
    return aline

class KFoldsClassifier(object):
    
    def __init__(self, classifier, n_splits=2, random_state=None, shuffle=False):
        import copy
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        self.classifier = classifier
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        
        self.kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        
        self.fitted_classifiers = []
        self.classifier_test_index = []
        
    def fit(self, X, y, W):
        import copy
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import train_test_split
        
        for train_index, test_index in self.kfold.split(X, y, W):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            W_train, W_test = W[train_index], W[test_index]
            
            X_tr, X_cal, y_tr, y_cal, W_tr, W_cal = train_test_split(X_train, 
                                                                     y_train, 
                                                                     W_train, 
                                                                     test_size=0.33,
                                                                     random_state=42)
            reg = copy.deepcopy(self.classifier)
            reg.fit(X_tr, y_tr, sample_weight=W_tr)
            calibrator = CalibratedClassifierCV(reg, cv='prefit', method='sigmoid')
            calibrator.fit(X_cal, y_cal, sample_weight=W_cal)
            self.fitted_classifiers.append(calibrator)
            self.classifier_test_index.append(test_index)
            
    def fit_for_tune(self, X, y, W):
        import copy
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import train_test_split
        
        for train_index, test_index in self.kfold.split(X, y, W):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            W_train, W_test = W[train_index], W[test_index]
            
            X_tr, X_cal, y_tr, y_cal, W_tr, W_cal = train_test_split(X_train, 
                                                                     y_train, 
                                                                     W_train, 
                                                                     test_size=0.33,
                                                                     random_state=42)
            reg = copy.deepcopy(self.classifier)
            reg.fit(X_tr, y_tr, sample_weight=W_tr)
            calibrator = CalibratedClassifierCV(reg, cv='prefit', method='sigmoid')
            calibrator.fit(X_cal, y_cal, sample_weight=W_cal)
            return calibrator
            
    def predict_test_sample(self, X):
        y_pred = np.zeros(len(X))
        predictions = []
        for reg, test_index in zip(self.fitted_classifiers, self.classifier_test_index):
            y_pred[test_index] = reg.predict_proba(X[test_index])[:,1]
        return y_pred
    
    def predict(self, X):
        predictions = []
        for reg in self.fitted_classifiers:
            predictions.append(reg.predict(X))
        print(np.mean(predictions, axis=0))
        return np.mean(predictions, axis=0)
import joblib
from tensorflow.keras.models import load_model


class model:
    def __init__(self):
        self.cover_DL = 'cover_DL.h5'
        self.cover_GB = 'cover_GB.pkl'
        self.cover_LinearSVC = 'cover_LinearSVC.pkl'
        self.cover_LR = 'cover_LR.pkl'
        
        self.motor_DL = 'motor_DL.h5'
        self.motor_GB = 'motor_GB.pkl'
        self.motor_LinearSVC = 'motor_LinearSVC.pkl'
        self.motor_LR = 'motor_LR.pkl'
        
        
    def load_cover_DL(self):
        return load_model(self.cover_DL)
    
    def load_cover_GB(self):
        return joblib.load(self.cover_GB)
    
    def load_cover_LinearSVC(self):
        return joblib.load(self.cover_LinearSVC)
    
    def load_cover_LR(self):
        return joblib.load(self.cover_LR)
    
    
    def load_motor_DL(self):
        return load_model(self.motor_DL)
    
    def load_motor_GB(self):
        return joblib.load(self.motor_GB)
    
    def load_motor_LinearSVC(self):
        return joblib.load(self.motor_LinearSVC)
    
    def load_motor_LR(self):
        return joblib.load(self.motor_LR)
    
    def predict(self, model, data):
        return model.predict(data)
    
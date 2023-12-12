
class feature_improtance_extractor():
    '''
    feature_importance_extractor
    '''  
    def __init__(self):
        self.feature_importance = []
        self.iter = 0
        pass
    def update(self, new_feature_importance):
        if self.iter == 0:
            self.feature_importance = new_feature_importance
        else:
            self.feature_importance += new_feature_importance
        self.iter += 1
        return
    
    def get(self):
        return (self.feature_importance / self.iter)
    
    def reset(self):
        self.feature_importance = []
        self.iter = 0
        return
    
extractor = feature_improtance_extractor()
def get_feature_importance_extractor() -> feature_improtance_extractor:
    return extractor



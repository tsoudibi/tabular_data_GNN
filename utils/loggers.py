
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
def get_feature_importance_extractor():
    return extractor


class wandb_logger():
    '''
    wandb_logger
    '''
    def __init__(self, project_name, run_name, config):
        import wandb
        wandb.init(project=project_name, name=run_name, config=config)
        self.wandb = wandb
        self.iter = 0
        return
    
    def log(self, package):
        self.wandb.log(package, step=self.iter)
        self.iter += 1
        return
    
    def finish(self):
        self.wandb.finish()
        return
    
    def get(self):
        return self.wandb
    
    def reset(self):
        self.iter = 0
        return
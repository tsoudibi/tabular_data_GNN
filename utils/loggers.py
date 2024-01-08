from utils.utils import *
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
    def __init__(self, wandb_config: dict):
        import wandb
        import os
        os.environ["WANDB_SILENT"] = "true"
        self.run = wandb.init( project = wandb_config['project'], 
                    name = wandb_config['name'],
                    notes = wandb_config['notes'],
                    entity = wandb_config['entity'],
                    group = wandb_config['group'],
                    # track hyperparameters and run metadata
                    config = dict(get_run_config(), **wandb_config))
        self.iter = 0
        return
    
    def log(self, package: dict):
        self.run.log(package, step=self.iter)
        self.iter += 1
        return
    
    def finish(self):
        self.run.finish()
        self.reset()
        return
    
    def get(self):
        return self.run
    
    def reset(self):
        self.iter = 0
        return
    
logger = None
def get_logger():
    return logger

def set_logger(new_logger) -> (None):
    global logger
    logger = new_logger
    return
def del_logger() -> (None):
    global logger
    logger = None
    return
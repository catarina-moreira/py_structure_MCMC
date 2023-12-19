from abc import ABC, abstractmethod

import pandas as pd
import networkx as nx

from scores.ScoreAbstract import Score

class MCMC(ABC):
    
    def __init__(self, data, score_function):
         self.data  = data
         self.score_function = score_function

    @abstractmethod
    def run(self):
        pass
    
    # GETTERS AND SETTERS
    ####################################################
    def setData(self, data : pd.DataFrame):
        self.data = data
        
    def getData(self):
        return self.data
    
    def setScore_function(self, score_function : Score):
        self.score_function = score_function
        
    def getScore_function(self):
        return self.score_function




####################################
#         OrderMCMC                #
####################################

class OrderMCMC(MCMC):
    
    def run(self):
        pass


####################################
#         PartitionMCMC            #
####################################
class PartitionMCMC(MCMC):
    
    def run(self):
        pass


'''
Created on Apr 10, 2016

@author: Fatemeh
'''

class CustomKey(object):
    ''' 
        Successful experiment with customized object
    '''
    
     
    def __init__(self, individual, cell):
        '''
        Constructor
        '''
        self.individual = individual
        self.cell = cell

    def __eq__(self, other):
        if isinstance(other, CustomKey):
            return (self.individual == other.individual) & (self.cell == other.cell)
        return NotImplemented
     
    def __hash__(self):
        return 101*hash(frozenset(self.individual))+ hash(self.cell)

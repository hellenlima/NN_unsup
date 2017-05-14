class TrnParams(object):
    
    def __init__(self, learning_rate=0.1,verbose= False):
        self.learning_rate = learning_rate
        self.verbose = verbose
    
    def Print(self):
        print 'Class KohonenSup TrnParams'
        print 'Learning Rate: %1.5f'%(self.learning_rate)
        if self.verbose:
            print 'Verbose: True'
        else:
            print 'Verbose: False'
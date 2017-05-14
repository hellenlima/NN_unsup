import numpy as np
import random

class KohonenSup(object):

    """
    KohonenSup class
        This class implement the Supervised Kohononen Map
    """
    def __init__(self, n_clusters=2, similarity_radius=0.1, dist="euclidean", randomize=True, dev=True):
        """
        KohonenSup constructor
            n_clusters: Number of cluster to be used (default: 2)
            similarity_radius: Similarity Radius (default: 0.1)
            dist: distance method used (default: euclidean)
            randomize: do or not random access to data            
            dev: Development flag
        """
        self.n_clusters = n_clusters
        self.clusters = None
        self.cluster_last_used = None
        self.similarity_radius = similarity_radius
        self.dist = dist
        self.randomize = randomize
        self.dev = dev
        
    def calc_dist(self, pt1, pt2):
        if self.dist == "euclidean":
            return np.linalg.norm((pt1-pt2),ord=2)
        
    def update_cluster(self, cluster_id, event, trn_params):
        self.clusters[cluster_id,:] = (self.clusters[cluster_id,:] + 
                                       trn_params.learning_rate*
                                       (event-self.clusters[cluster_id,:]))
    
    def create_cluster(self,new_cluster):
        if self.clusters is None:
            self.clusters = new_cluster[:,np.newaxis].T
        else:
            if len(self.clusters.shape) == 1:
                self.clusters = np.append(self.clusters[:,np.newaxis].T,new_cluster[:,np.newaxis].T,axis=0)
            else:
                self.clusters = np.append(self.clusters,new_cluster[:,np.newaxis].T,axis=0)
    
    def init_clusters(self, data, trgt):
        #y_data CANNOT be in categorical form
        #trgt_classes = np.argmax(trgt, axis=1);
        classes = np.unique(trgt);
        if self.clusters is None:
            for icluster in range(len(classes)):
                new_cluster = data[trgt==classes[icluster]][0]
                self.create_cluster(new_cluster)
                
    def fit(self, data, trgt, trn_params=None):
        #trgt is not categorical
        if trn_params is None:
            trn_params = TrnParams()
            
        if self.dev:
            trn_params.Print()
            
        if self.randomize:
            if data.shape[0] < data.shape[1]:
                ids = np.random.permutation(data.shape[1])
                trn_data = data[:,ids].T
                trn_trgt = trgt[ids]
            else:
                ids = np.random.permutation(data.shape[0])
                trn_data = data[ids,:]
                trn_trgt = trgt[ids]
        else:
            if data.shape[0] < data.shape[1]:
                trn_data = data.T
            else:             
                trn_data = data
            trn_trgt = trgt
            
        print "Number of events:",trn_data.shape[0]
        
        self.init_clusters(data, trgt)
        
        for ievent in range(trn_data.shape[0]):
            #print 'ievent: ',ievent
            self.update_cluster(trn_trgt[ievent],data[ievent,:],trn_params=trn_params)
                
        
'''    def fit(self, x_data, y_data, n_clusters, trn_params=None): #usa o parametro da classe?
        if trn_params is None:
            trn_params = TrnParams()
            
        if self.dev:
            trn_params.Print()
            
        if self.randomize:
            if x_data.shape[0] < x_data.shape[1]:
                trn_data = x_data[:,np.random.permutation(x_data.shape[1])].T
            else:
                trn_data = x_data[np.random.permutation(x_data.shape[0]),:]
        else:
            if x_data.shape[0] < x_data.shape[1]:
                trn_data = x_data.T
            else:
                trn_data = x_data
        print "Number of events:",trn_data.shape[0]
        
        self.init_clusters(x_data, y_data);
        
        for ievent in range(trn_data.shape[0]):
            #print 'ievent: ',ievent
            mat_dist = np.zeros([self.clusters.shape[0]])
            for icluster in range(self.clusters.shape[0]):
                mat_dist[icluster] = self.calc_dist(trn_data[ievent],self.clusters[icluster,:])
            #if np.min(mat_dist) > self.similarity_radius:
            #    continue #??
                #self.create_cluster(trn_data[ievent,:]) #data does not belong to any cluster...
            #else:
                update_cluster_id = np.argmin(mat_dist)
                self.update_cluster(update_cluster_id,data[ievent,:],trn_params=trn_params) '''

            
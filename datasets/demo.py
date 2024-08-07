import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from pyDOE import lhs         #Latin Hypercube Sampling

class PINN_dataset(Dataset):
    def __init__(self, filepath, test=False, N_u=100, N_f = 10000):
        self.test = test
        data = scipy.io.loadmat(filepath)  	# Load data from file
        self.x = data['x']                                   # 256 points between -1 and 1 [256x1]
        self.t = data['t']                                   # 100 time points between 0 and 1 [100x1] 
        self.usol = data['usol']                             # solution of 256x100 grid points
        

        self.X, self.T = np.meshgrid(self.x,self.t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
        ''' X_u_test = [X[i],T[i]] [25600,2] for interpolation'''
        self.X_u_test = np.hstack((self.X.flatten()[:,None], self.T.flatten()[:,None]))

        # Domain bounds
        self.lb = self.X_u_test[0]  # [-1. 0.]
        self.ub = self.X_u_test[-1] # [1.  0.99]

        '''
        Fortran Style ('F') flatten,stacked column wise!
        u = [c1 
                c2
                .
                .
                cn]

        u =  [25600x1] 
        '''
        self.u_true = self.usol.flatten('F')[:,None] 
        # 'Generate Training data'
        self.N_u = N_u #Total number of data points for 'u'
        self.N_f = N_f #Total number of collocation points 
        X_f_train_np_array, X_u_train_np_array, u_train_np_array = self.training_data()

        # 'Convert to tensor'
        self.X_f_train = torch.from_numpy(X_f_train_np_array).float()
        self.X_u_train = torch.from_numpy(X_u_train_np_array).float()
        self.u_train = torch.from_numpy(u_train_np_array).float()
        self.X_u_test_tensor = torch.from_numpy(self.X_u_test).float()
        self.u = torch.from_numpy(self.u_true).float()
        self.f_hat = torch.zeros(self.X_f_train.shape[0],1)
        
    
    def training_data(self):
        '''Boundary Conditions'''

        #Initial Condition -1 =< x =<1 and t = 0  
        leftedge_x = np.hstack((self.X[0,:][:,None], self.T[0,:][:,None])) #L1
        leftedge_u = self.usol[:,0][:,None]

        #Boundary Condition x = -1 and 0 =< t =<1
        bottomedge_x = np.hstack((self.X[:,0][:,None], self.T[:,0][:,None])) #L2
        bottomedge_u = self.usol[-1,:][:,None]

        #Boundary Condition x = 1 and 0 =< t =<1
        topedge_x = np.hstack((self.X[:,-1][:,None], self.T[:,0][:,None])) #L3
        topedge_u = self.usol[0,:][:,None]

        all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x]) # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
        all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])   #corresponding u [456x1]

        #choose random N_u points for training
        idx = np.random.choice(all_X_u_train.shape[0], self.N_u, replace=False) 

        X_u_train = all_X_u_train[idx, :] #choose indices from  set 'idx' (x,t)
        u_train = all_u_train[idx,:]      #choose corresponding u

        '''Collocation Points'''

        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        X_f_train = self.lb + (self.ub-self.lb)*lhs(2,self.N_f) 
        X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points 

        return X_f_train, X_u_train, u_train 


    def __getitem__(self, index):
        if self.test:
            return self.X_u_test_tensor, self.u
        return self.X_u_train, self.u_train, self.X_f_train, self.f_hat

    def __len__(self):
        return 250
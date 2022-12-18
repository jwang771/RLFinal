import numpy as np
import nn
import torch

def normalize(X, cuts):
    concat = np.concatenate((X, cuts))
    min_X = np.min(concat)
    range_X = np.ptp(concat)
    return (X - min_X)/range_X, (cuts - min_X)/range_X

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


# define neural net \pi_\phi(s) as a class

class Policy(object):
    
    def __init__(self, obssize, hidden, linear, lr):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # TODO DEFINE THE MODEL
        self.model = nn.encdecoder_actor(obssize, hidden, linear)
        
        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    
    def compute_prob(self, Ab, cuts):
        """
        compute prob distribution over all actions given state: pi(s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        Ab = torch.FloatTensor(Ab)
        cuts = torch.FloatTensor(cuts)
        prob = torch.nn.functional.softmax(self.model.forward(Ab, cuts), dim=-1)
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)
    
    def train(self, Ab, cuts, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        actions = torch.LongTensor(actions)
        Qs = torch.FloatTensor(Qs)
        tot_loss = 0
        # COMPUTE probability vector pi(s) for all s in states
        for i in range(len(cuts)):

            logits = self.model.forward(Ab[i], cuts[i])
            prob = torch.nn.functional.softmax(logits, dim=-1)

            # Compute probaility pi(s,a) for all s,a
            action_onehot = self._to_one_hot(actions[i], cuts[i].shape[0])
            prob_selected = torch.sum(prob * action_onehot, axis=-1)
        
            # FOR ROBUSTNESS
            prob_selected += 1e-8

            # TODO define loss function as described in the text above
            loss = - torch.mean(torch.log(prob_selected) * Qs[i])

            # BACKWARD PASS
            self.optimizer.zero_grad()
            loss.backward()

            # UPDATE
            self.optimizer.step()
            tot_loss += loss.detach().cpu().data.numpy()
            
        return tot_loss

class ValueFunction(object):
    
    def __init__(self, obssize, hidden, linear, lr):
        """
        obssize: size of states
        """
        # TODO DEFINE THE MODEL
        self.model = nn.encdecoder_critic(obssize, hidden, linear)
        
        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # RECORD HYPER-PARAMS
        self.obssize = obssize
    
    def compute_values(self, Ab, cuts):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        Ab = torch.FloatTensor(Ab)
        cuts = torch.FloatTensor(cuts)
        return self.model.forward(Ab, cuts).cpu().data.numpy()
    
    def train(self, Ab, cuts, targets):
        """
        states: numpy array
        targets: numpy array
        """
        targets = torch.FloatTensor(targets)
        tot_loss = 0
        v_preds = []
        for i in range(len(Ab)):
            # COMPUTE Value PREDICTIONS for states 
            v_pred = self.model.forward(Ab[i], cuts[i]) 
            v_preds.append(v_pred.detach().cpu().data.numpy())      
            # LOSS
            # TODO: set LOSS as square error of predicted values compared to targets
            loss = torch.mean((v_pred  - targets[i])**2)
            # BACKWARD PASS
            self.optimizer.zero_grad()
            loss.backward()

            # UPDATE
            self.optimizer.step()
            tot_loss += loss.detach().cpu().data.numpy()
        
        return v_preds, tot_loss
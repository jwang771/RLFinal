from gymenv_v2 import make_multiple_env
import numpy as np
import torch
import torch.nn as nn
import pg

import wandb
wandb.login()
#run=wandb.init()
run=wandb.init(project="finalproject", entity="orcs4529", tags=["training-easy"])
#run=wandb.init(project="finalproject", entity="orcs4529", tags=["training-hard"])

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}


test_config = {
    "load_dir" : 'instances/test_100_n60_m60',
    "idx_list" : list(range(99)),
    "timelimit" : 50,
    "reward_type" : 'obj'
}


if __name__ == "__main__":

    PATH = "easypolicy/A_64_30_2.pt"

    # create env
    lr = 3e-3
    epsilon = 0.99
    min_eps = 0.05
    best = 0
    count = 0
    #Initiate policy 
    actor = pg.Policy(61, 30, 64, lr)
    baseline = pg.ValueFunction(61, 16, 64, lr)
    repeats = 2
    gamma = 0.99 
    rrecord = []
    c = 0
    for i in range(repeats):
        env = make_multiple_env(**easy_config)
        for e in range(50):
            c+=1
            # gym loop
            # To keep a record of states actions and reward for each episode
            
            OBS_Ab = []  
            OBS_cuts = []
            ACTS = []
            ADS = []
            VAL = []
            rews = []

            obs = env.reset()   
            done = False
            repisode = 0
            counter = 0
            while not done:
                counter+=1
                A, b, c0, cuts_a, cuts_b = obs

                # normalization
                A, cuts_a = pg.normalize(A, cuts_a)
                b, cuts_b = pg.normalize(b, cuts_b)
                Ab = np.concatenate((A, np.expand_dims(b, 1)), axis=1)
                cuts = np.concatenate((cuts_a, np.expand_dims(cuts_b, 1)), axis=1)
                
                act_prob = actor.compute_prob(Ab, cuts)
                act_prob /= np.sum(act_prob)
                act_prob = act_prob.flatten()

                if np.random.uniform(0,1) <= max(min_eps, epsilon**(3*c)):
                    a = np.random.randint(0, obs[-1].size, 1)
                else:
                    a = [np.random.choice(np.arange(obs[-1].size), size = 1, p=act_prob).item()]
                prev=obs
                obs, r, done, _ = env.step(list(a))

                OBS_Ab.append(Ab)
                OBS_cuts.append(cuts)
                ACTS.append(np.asarray(a))
                rews.append(r)
                repisode += r
            
            rrecord.append(np.sum(rews))
            returns = pg.discounted_rewards(rews, gamma)
            
            # NO CRITIC
            advantage = returns
            # WITH CRITIC
            v_preds, loss = baseline.train(OBS_Ab, OBS_cuts, returns)
            v_preds = np.vstack(v_preds).flatten()
            advantage = returns - v_preds
            print("episode: ", c)
            print("sum reward: ", repisode)
            ACTS = np.vstack(ACTS)
            loss = actor.train(OBS_Ab, OBS_cuts, ACTS, advantage)
            print("Loss: ", loss)
            #save model
            if repisode >= best:
                torch.save(actor, PATH)
                best = repisode
            
            #wandb logging
            wandb.log({"Discounted Reward": np.sum(returns)})
            fixedWindow = 10
            movingAverage = 0
            if len(rrecord) >= fixedWindow:
                movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow:len(rrecord) - 1])
            wandb.log({"Training reward": repisode, "training reward moving average": movingAverage})
            


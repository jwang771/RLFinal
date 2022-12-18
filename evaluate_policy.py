from gymenv_v2 import make_multiple_env
import numpy as np
import torch
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

if __name__ == "__main__":
    # create env
    env = make_multiple_env(**easy_config) 
    gamma = 0.99
    PATH = 'easypolicy/A_64_30_2.pt'
    actor = torch.load(PATH)
    rrecord = []
    for e in range(50):
        # gym loop
        obs = env.reset()   # samples a random instance every time env.reset() is called
        done = False
        t = 0
        repisode = 0
        rews = []
        while not done:
            A, b, c0, cuts_a, cuts_b = obs

            # normalization
            A, cuts_a = pg.normalize(A, cuts_a)
            b, cuts_b = pg.normalize(b, cuts_b)
            Ab = np.concatenate((A, np.expand_dims(b, 1)), axis=1)
            cuts = np.concatenate((cuts_a, np.expand_dims(cuts_b, 1)), axis=1)

            act_prob = actor.compute_prob(Ab, cuts)
            act_prob /= np.sum(act_prob)
            act_prob = act_prob.flatten()
            a = [np.random.choice(np.arange(obs[-1].size), size = 1, p=act_prob).item()]          # s[-1].size shows the number of actions, i.e., cuts available at state s
            prev = obs
            obs, r, done, _ = env.step(list(a))
        
            
            t += 1
            repisode += r
            rews.append(r)
        rrecord.append(np.sum(rews))
        returns = pg.discounted_rewards(rews, gamma)
    	    #wandb logging
        print('episode:', e, 'reward:', repisode)
        wandb.log({"Discounted Reward": np.sum(returns)})
        fixedWindow = 10
        movingAverage = 0
        if len(rrecord) >= fixedWindow:
            movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow:len(rrecord) - 1])
        wandb.log({"evaluation reward": repisode, "evaluation moving average": movingAverage})
	    #make sure to use the correct tag in wandb.init in the initialization on top

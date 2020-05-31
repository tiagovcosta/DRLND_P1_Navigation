# Report - Project 1: Navigation

Full project (including source code and trained weights) is available on [the GitHub repo](https://github.com/tiagovcosta/DRLND_P1_Navigation).

## Learning Algorithm

The agent uses the [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorithm to solve this RL problem, taking advantage of both '*experience replay*' and '*fixed Q targets*' to improve training.

The implementation also supports 2 extensions to the algorithm:
-  [Double DQN](https://arxiv.org/abs/1509.06461)
-  [Dueling DQN](https://arxiv.org/abs/1511.06581)

### Model Architecture

```python
#DQN architecture
QNetwork(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc_final): Linear(in_features=64, out_features=4, bias=True)
)

#Dueling DQN architecture
QNetwork(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc_v): Linear(in_features=64, out_features=1, bias=True)
  (fc_a): Linear(in_features=64, out_features=4, bias=True)
)
```

### Code implementation

- `model.py`

    Implementation of the **QNetwork** PyTorch module. The network consists of 4 fully connected layers (with bias), the first 3 of which are followed by ReLU activations.

    The module also supports **Dueling QDN** architecture by adding an extra fully connected layer and combining state and action values, as described in the [Dueling DQN paper](https://arxiv.org/abs/1511.06581).

- `dqn_agent.py`

    - Agent

        Implementation of the **Agent** class which manages the DQN networks and provides functions to select actions according to the agents policy, as well as, functions to train the agent.

        The Agent class supports both the Dueling DQN and Double DQN extensions.

    - Replay Buffer

        The **ReplayBuffer** class consisits of a fixed-size buffer to store experience tuples (state, action, reward, next_state, done), with support for uniformly sampling batches of tuples for training.

### Hyperparameters

```python
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate 
UPDATE_PARAM_EVERY = 4      # how often to update the network weights
UPDATE_TARGET_EVERY = 8     # how often to update the target network

# Exploration EPS:
start = 1.0
end = 0.02
decay = 0.95
```

## Plot of Rewards

The **DQN Agent** was able to solve this problem in **353 episodes**.

On the other hand, the Agent using **Double and Dueling DQN** took **383 episodes** to solve the problem.

![Reward plot][reward_plot]

## Ideas for Future Work

* Implement [other extensions to DQN](https://arxiv.org/pdf/1710.02298.pdf) such as:
    * Prioritized Experience Replay
    * Multi-step learning
    * Noisy Nets
    * Distributional RL
* Experiment with soft vs hard target network update
* Different weight initialization methods
* Hyperparameter optimization

[reward_plot]: reward_plot.png "Reward plot"
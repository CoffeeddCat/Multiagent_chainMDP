# Multiagent_chainMDP

- It's a simple&naive platform for multiagent chainMDP problem.

- The algorithm now used: DDQN+reward updated.

- Environment needed: 

  ```
  python3.5, Tensorflow1.4.0, Numpy1.13.3, matplotlib2.1.0, Tensorboard0.4.0rc(unnecessary)
  ```

- Both GPU or CPU for tensorflow are surpported.

- References:

  - https://arxiv.org/abs/1507.00814
  - http://papers.nips.cc/paper/6500-deep-exploration-via-bootstrapped-dqn
  - ...

- Need to be done:

  - Multi thread to speed up
  - More algorithm should be applied
  - A more efficient exploration method to solve it.
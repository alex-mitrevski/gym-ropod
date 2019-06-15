# gym-ropod

Defines a Gazebo-based OpenAI gym environment for a ROPOD robot.

## Defined environments

### `ropod-nav-discrete-v0`

A navigation environment for a ropod with a discrete action space.

The action space contains the following actions:
* `0`: straight motion with `0.1m/s`
* `1`: left motion with `0.1m/s`
* `2`: right motion with `0.1m/s`
* `3`: left turn with `0.1m/s` linear speed and `0.1m/s` rotational speed
* `4`: right turn with `0.1m/s` linear speed and `0.1m/s` rotational speed

## Usage

A simple usage example for the environment is given below:

```
import gym
launch_file = '/path/to/my_simulation_launch_file.launch'

# create, render, and reset the environment
env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file)
env.render(mode='human')
env.reset()

# sample an action
action = env.action_space.sample()

# apply the sampled action
env.step(action)
```

Test scripts that illustrate the environment use and should run out of the box can be found under [test](test).

## Requirements

* Python 3.5+
* `gym`
* `numpy`
* `transforms3d`
* The [`ropod simulation`](https://github.com/ropod-project/ropod_sim_model)
* `rospkg`
* `termcolor`

## Acknowledgments

The implementation is heavily based on [this Toyota HSR gym environment](https://github.com/ascane/gym-gazebo-hsr)

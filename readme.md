# Deep Reinforcement learning in AirSim

## Requirements
Install AirSim and Unreal Engine 4 using the instructions from the [official website](https://microsoft.github.io/AirSim/). Pytorch can be installed from the instructions given [here](https://pytorch.org/).\
Other dependencies can be installed using the following line of command
```console
pip install --user -r requirements.txt
```
## How to Run
Train using DQN from the following command
```console
python main.py --batch_size 32 --num_frames 10000000
```
Train using dueling DQN from the following command
```console
python main.py --batch_size 32 --num_frames 10000000 --dueling
```
Load previously trained model to restart training
```console
python main.py --batch_size 32 --num_frames 10000000 --load_model --model_path \path\to\model --optimizer_path \path\to\optimizer
```

## How to modify

### Change environment
Add a different environment by adding environment class in lines 57-60 of *main.py*

### Modifying network
The network could be modified in *network.py*

## Acknowledgements

The environment class is inspired by the works of [Subin Yang](https://github.com/ysbsb/airsim_quadrotor_pytorch)

## Author

[Varun Pawar](mailto:varunpwr897@gmail.com) 
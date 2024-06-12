# A Comparative Analysis of Reinforcement Learning Methods for Stratospheric Balloon Station Keeping
Stanford CS234 student project on the topics of reinforcement learning for stratospheric balloons.

## Environment
This project uses Google's [Balloon Learning Environment](https://github.com/google/balloon-learning-environment) described in the associated paper ["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).

```
pip install balloon_learning_environment
```

Environment setup can prove finicky and may not work on all machines. See `requirements.txt` for a list of dependencies we found usable on Windows. We recommend Python 3.9.

## Baselines

- **Random Walk** ([Google BLE](https://github.com/google/balloon-learning-environment)) - Randomly samples atmospheric pressures and chooses actions that approach it.
  ```
  python eval.py --agent random-walk
  ```
  
- **Station-Seeker** ([Google BLE](https://github.com/google/balloon-learning-environment)) - A deterministic algorithm using heuristics.
  ```
  python eval.py --agent station-seeker
  ```

- **Perciatelli** ([Google BLE](https://github.com/google/balloon-learning-environment)) - A pretrained QR-DQN model from the Google paper suggested to be state-of-the-art.
  ```
  python eval.py --agent perciatelli
  ```

## Reported Models

- **PPO** (Stanford CS234 HW2) - A PPO implementation adapted from a CS234 homework assignment for use in our framework.

  Train:
  ```
  python train.py --agent ppo --config configs/ppo_lr=1e-5.yml --out-dir results/ppo/
  ```

  Eval (pretrained ckpt):
  ```
  python eval.py --agent ppo --config configs/ppo_lr=1e-5.yml --ckpt ckpts/ppo/ckpt-1000.pt
  ```

- **DQN** ([PyTorch](https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py)) - A vanilla DQN implementation modified to fit the BLE environment.

  Train:
  ```
  python pt_dqn.py --out-dir results/dqn/
  ```

  Eval (pretrained ckpt):
  ```
  python eval.py --agent pt_dqn --config configs/pt_dqn_ble.yml --ckpt ckpts/dqn/ckpt-1999.pt
  ```

- **QR-DQN** ([Google BLE](https://github.com/google/balloon-learning-environment)) - The QR-DQN implementation from the Google paper, wrapped for use in our framework.

  Train:
  ```
  python train.py --agent qrdqn --config configs/ble_qrdqn.yml --gin-config configs/ble_qrdqn.gin --out-dir results/qrdqn/
  ```

  Eval (pretrained ckpt):
  ```
  python eval.py --agent qrdqn --config configs/ble_qrdqn.yml --gin-config configs/ble_qrdqn.gin --ckpt ckpts/qrdqn/checkpoint_00200.pkl
  ```

- **Discrete SAC** ([Revisiting-Discrete-SAC](https://github.com/coldsummerday/Revisiting-Discrete-SAC)) - A discrete Soft Actor-Critic implementation modified to fit the BLE environment.

  Train:
  ```
  cd Revisiting_Discrete_SAC/src/
  python balloon_sac.py --logdir ../../results/dsac/
  ```

  Eval (pretrained ckpt):
  ```
  python eval.py --agent dsac --config configs/revisiting_dsac.yml --ckpt ckpts/dsac/policy-200.pth
  ```

## Unreported Models
Underperforming variants of some of the models reported in our project. It is possible that these methods could work well if the hyperparameters were properly tuned.

- **DQN** ([PyTorch](https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py)) - The same DQN implementation as reported, but wrapped for use in our framework.
  ```
  python train.py --agent vdqn --config configs/old/vdqn_lr=1e-5.yml --out-dir results/vdqn/
  python eval.py --agent vdqn --config configs/old/vdqn_lr=1e-5.yml --ckpt results/vdqn/ckpt-200.pt
  ```

- **DQN** ([Google BLE](https://github.com/google/balloon-learning-environment?tab=readme-ov-file)) - The vanilla DQN implementation from the Google paper, wrapped for use in our framework.
  ```
  python train.py --agent ble_vdqn --config configs/old/ble_vdqn.yml --gin-config configs/old/ble_vdqn.gin --out-dir results/ble_vdqn/
  python eval.py --agent ble_vdqn --config configs/old/ble_vdqn.yml --gin-config configs/old/ble_vdqn.gin --ckpt results/ble_vdqn/checkpoint_00200.pkl
  ```

- **PPO** ([XinJingHao](https://github.com/XinJingHao/PPO-Discrete-Pytorch)) - A PPO implementation with more features and hyperparameters than the one reported.
  ```
  python train.py --agent knob_ppo --config configs/old/knobbified_ppo_lambd=1.0.yml --out-dir results/knob_ppo/
  python eval.py --agent knob_ppo --config configs/old/knobbified_ppo_lambd=1.0.yml --ckpt results/knob_ppo/ckpt-200.pt
  ```

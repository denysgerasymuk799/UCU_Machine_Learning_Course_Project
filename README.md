# UCU_Machine_Learning_Course_Project

## Project Aim

Forecast the amount of Solar Radiation 10-15 hours ahead.

## How to run the project

```shell
conda create -n ML_project_env_v1 python=3.7

conda activate ML_project_env_v1

# Use below command or install requirements via DataSpell UI, if possible (recommended)
conda install --file requirements.txt

# Run TensorBoard to see NN training progress
tensorboard --logdir logs/fit
```
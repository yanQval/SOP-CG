# SOP-CG: Self-Organized Polynomial-Time Coordination Graphs

This codebase is based on [PyMARL](https://github.com/oxwhirl/pymarl) and contains the implementation
of the SOP-CG algorithm.

## Run an experiment 

Tasks can be found in `src/envs`. To run experiments on MACO benchmark:

```shell
python src/main.py --config=sopcg --env-config=pursuit with construction='tree' use_action_repr=False
```

To run experiment on Tag:
```shell
python src/main.py --config=sopcg_vs_vdn --env-config=tag with construction='tree' use_action_repr=False
```

To run experiments on SMAC benchmark:
```shell
python src/main.py --config=sopcg --env-config=sc2 with env_args.map_name='10m_vs_11m' construction='tree' use_action_repr=True
```

The hyperparameter `construction` is used to control the graph class, where `tree` represents $\mathcal{G}_T$, and `matching` represents $\mathcal{G}_P$. By default, it is set to `tree`. The hyperparameter `use_action_repr` is set to `False` by default, and setting it to `True` would use the action representation learning technique.

The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder named *models*. The directory corresponding to each run will contain models saved throughout the training process, each of which is named by the number of timesteps passed since the learning process starts.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

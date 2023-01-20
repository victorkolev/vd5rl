# d4rl2

## WARNING!

This repository uses submodules. Please clone and pull taking this into account.

```bash
git clone --recurse-submodules https://github.com/ikostrikov/d4rl2
```

## A1 Maze

### Visualization 

Run to visualize the data collection policy
```bash
python -m d4rl2.envs.a1.collect.maze.visualize
```

### Data collection 

Run to collect data for a1 maze
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= python -m d4rl2.envs.a1.collect.maze.collect --maze_name=umaze
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= python -m d4rl2.envs.a1.collect.maze.collect --maze_name=medium_maze
```

Run to collect data for a1 walk
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= python -m d4rl2.envs.a1.collect.walk.train_online
```

Run all
```bash
tmuxp load run_all.yaml
```

### Download datasets

First install [gsutil](https://cloud.google.com/storage/docs/gsutil_install#deb).

```bash

mkdir -p ~/.d4rl2/datasets/a1/

gsutil cp \
  "gs://d4rl2/a1/a1-medium_maze.hdf5" \
  "gs://d4rl2/a1/a1-umaze.hdf5" \
  "gs://d4rl2/a1/a1-walk.hdf5" \
  ~/.d4rl2/datasets/a1/
```

### Baselines 

[Click](baselines)

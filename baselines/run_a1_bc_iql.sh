
# Should take roughly 10 * 10 * 3 / 4 = 75 minutes to finish on a GPU:
parallel --delay 20 --linebuffer -j 4 python train_bc_iql.py --config=configs/bc_iql_default.py:bc --seed={1} --env_name={2} ::: 0 1 2 3 4 5 6 7 8 9 ::: a1-umaze-diverse-v0 a1-medium_maze-diverse-v0 a1-walk_stable-v0

# IQL training for default configs.
parallel --delay 20 --linebuffer -j 4 python train_bc_iql.py --config=configs/bc_iql_default.py:iql --seed={1} --env_name={2} ::: 0 1 2 3 4 5 6 7 8 9 ::: a1-umaze-diverse-v0 a1-medium_maze-diverse-v0 a1-walk_stable-v0
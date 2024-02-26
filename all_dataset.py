from trainer import *

# update_run_config('num_learner', 8)

# credit
update_run_config('dataset', 'credit')
update_run_config('max_epoch', 250)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
update_run_config('random_state', 95408)
train_K_fold(get_run_config())

# compas
update_run_config('dataset', 'compas')
update_run_config('max_epoch', 500)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
update_run_config('random_state', 95408)
train_K_fold(get_run_config())

# adult
update_run_config('dataset', 'adult')
update_run_config('max_epoch', 50)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
update_run_config('random_state', 95408)
train_K_fold(get_run_config())

# comapss_old
update_run_config('dataset', 'compass_old')
update_run_config('max_epoch', 500)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
# update_run_config('random_state', 95408)
# train_K_fold(get_run_config())

# electricity_cat
update_run_config('dataset', 'electricity_cat')
update_run_config('max_epoch', 250)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
update_run_config('random_state', 95408)
train_K_fold(get_run_config())
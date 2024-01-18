from trainer import *
# test_run()
# test_run()
# train_K_fold(get_run_config())

update_run_config('dataset', 'compass_old')
update_run_config('max_epoch', 50)
update_run_config('random_state', 42)
train_K_fold(get_run_config())
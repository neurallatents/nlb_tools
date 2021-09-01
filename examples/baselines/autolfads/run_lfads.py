import ray, yaml, shutil
from ray import tune
from os import path

from tune_tf2.models import create_trainable_class
from tune_tf2.pbt.hps import HyperParam
from tune_tf2.pbt.schedulers import MultiStrategyPBT
from tune_tf2.pbt.trial_executor import SoftPauseExecutor
from lfads_tf2.utils import flatten
from lfads_tf2.tuples import LoadableData
import h5py
import numpy as np

import sys
import time
from datetime import datetime

dataset_name = 'mc_maze'
bin_size_ms = 5
binsuf = "" if bin_size_ms == 5 else "_20"

# ---------- PBT I/O CONFIGURATION ----------
# the default configuration file for the LFADS model
CFG_PATH = f"./config/{dataset_name}{binsuf}.yaml"
# the directory to save PBT runs (usually '~/ray_results')
PBT_HOME = "~/autolfads/runs/"
# the name of this PBT run (run will be stored at {PBT_HOME}/{PBT_NAME})
RUN_NAME = f'{dataset_name}' # the name of the PBT run
# the dataset to train the PBT model on
DATA_DIR = '~/data/lfads_input/'
DATA_PREFIX = f'{dataset_name}{binsuf}_train_'

# ---------- PBT RUN CONFIGURATION ----------
# whether to use single machine or cluster
SINGLE_MACHINE = True
# the number of workers to use - make sure machine can handle all
NUM_WORKERS = 20
# the resources to allocate per model
RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
# the hyperparameter space to search
HYPERPARAM_SPACE = {
    'TRAIN.LR.INIT': HyperParam(1e-5, 5e-3, explore_wt=0.3, 
        enforce_limits=True, init=4e-3),
    'MODEL.DROPOUT_RATE': HyperParam(0.0, 0.6, explore_wt=0.3,
        enforce_limits=True, sample_fn='uniform'),
    'MODEL.CD_RATE': HyperParam(0.01, 0.7, explore_wt=0.3,
        enforce_limits=True, init=0.5, sample_fn='uniform'),
    'TRAIN.L2.GEN_SCALE': HyperParam(1e-4, 1e-0, explore_wt=0.8),
    'TRAIN.L2.CON_SCALE': HyperParam(1e-4, 1e-0, explore_wt=0.8),
    'TRAIN.KL.CO_WEIGHT': HyperParam(1e-6, 1e-4, explore_wt=0.8),
    'TRAIN.KL.IC_WEIGHT': HyperParam(1e-6, 1e-3, explore_wt=0.8),
}
# override if necessary
if dataset_name == 'area2_bump':
    HYPERPARAM_SPACE['TRAIN.KL.IC_WEIGHT'] = HyperParam(1e-6, 1e-4, explore_wt=0.8)
elif dataset_name == 'dmfc_rsg':
    HYPERPARAM_SPACE['TRAIN.LR.INIT'] = HyperParam(1e-5, 7e-3, explore_wt=0.3, 
        enforce_limits=True, init=5e-3)
    HYPERPARAM_SPACE['MODEL.CD_RATE'] HyperParam(0.01, 0.99, explore_wt=0.3,
        enforce_limits=True, init=0.5, sample_fn='uniform')
    HYPERPARAM_SPACE['TRAIN.L2.GEN_SCALE'] HyperParam(1e-6, 1e-1, explore_wt=0.8)
    HYPERPARAM_SPACE['TRAIN.L2.CON_SCALE'] HyperParam(1e-6, 1e-1, explore_wt=0.8)
    HYPERPARAM_SPACE['TRAIN.KL.CO_WEIGHT'] HyperParam(1e-7, 1e-4, explore_wt=0.8)
    HYPERPARAM_SPACE['TRAIN.KL.IC_WEIGHT'] HyperParam(1e-7, 1e-3, explore_wt=0.8)
PBT_METRIC='smth_val_nll_heldin'
EPOCHS_PER_GENERATION = 25
# ---------------------------------------------

# setup the data hyperparameters
dataset_info = {
    'TRAIN.DATA.DIR': DATA_DIR,
    'TRAIN.DATA.PREFIX': DATA_PREFIX}
# setup initialization of search hyperparameters
init_space = {name: tune.sample_from(hp.init) 
    for name, hp in HYPERPARAM_SPACE.items()}
# load the configuration as a dictionary and update for this run
flat_cfg_dict = flatten(yaml.full_load(open(CFG_PATH)))
flat_cfg_dict.update(dataset_info)
flat_cfg_dict.update(init_space)
# Set the number of epochs per generation
tuneLFADS = create_trainable_class(EPOCHS_PER_GENERATION)
# connect to Ray cluster or start on single machine
address = None if SINGLE_MACHINE else 'localhost:10000'
ray.init(address=address)
# create the PBT scheduler
scheduler = MultiStrategyPBT(
    HYPERPARAM_SPACE,
    metric=PBT_METRIC)
# Create the trial executor
executor = SoftPauseExecutor(reuse_actors=True)
# Create the command-line display table
reporter = tune.CLIReporter(metric_columns=['epoch', PBT_METRIC])
try:
    # run the tune job, excepting errors
    tune.run(
        tuneLFADS,
        name=RUN_NAME,
        local_dir=PBT_HOME,
        config=flat_cfg_dict,
        resources_per_trial=RESOURCES_PER_TRIAL,
        num_samples=NUM_WORKERS,
        sync_to_driver='# {source} {target}', # prevents rsync
        scheduler=scheduler,
        progress_reporter=reporter,
        trial_executor=executor,
        verbose=1,
        reuse_actors=True,
    )
except tune.error.TuneError:
    print("tune error!??!?")
    pass

# load the results dataframe for this run
pbt_dir = path.join(PBT_HOME, RUN_NAME)
df = tune.Analysis(pbt_dir).dataframe()
df = df[df.logdir.apply(lambda path: not 'best_model' in path)]
# find the best model
best_model_logdir = df.loc[df[PBT_METRIC].idxmin()].logdir
best_model_src = path.join(best_model_logdir, 'model_dir')
# copy the best model somewhere easy to find
best_model_dest = path.join(pbt_dir, 'best_model')
shutil.copytree(best_model_src, best_model_dest)
# perform posterior sampling
from lfads_tf2.models import LFADS
model = LFADS(model_dir=best_model_dest)
model.sample_and_average()

loadpath = f'~/data/lfads_input/{dataset_name}{binsuf}_test_lfads.h5'
h5file = h5py.File(loadpath, 'r')
test_data = LoadableData(
    train_data=h5file['train_data'][()].astype(np.float32),
    valid_data=h5file['valid_data'][()].astype(np.float32),
    train_ext_input=None,
    valid_ext_input=None,
    train_inds=h5file['train_inds'][()].astype(np.float32),
    valid_inds=h5file['valid_inds'][()].astype(np.float32),
)
h5file.close()

model.sample_and_average(loadable_data=test_data, ps_filename='posterior_samples_test.h5', merge_tv=True)
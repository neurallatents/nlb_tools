import os, sys
from pathlib import Path
import pytest
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from datetime import datetime, timezone
from dandi.download import download

from nlb_tools.nwb_interface import NWBDataset

SEED = 0

DATA_DIR = Path(os.path.dirname(__file__), "temp_data")
DUMMY_RAW_FILE_NAME = "dummy.npz"
DUMMY_NWB_TRAIN_FILE_NAME = "dummy_train.nwb"
DUMMY_NWB_TEST_FILE_NAME = "dummy_test.nwb"
DANDISET_URL = "https://dandiarchive.org/dandiset/000138"
NLB_FILE_NAME = "000138/sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"

DUMMY_BIN_SIZE = 0.001
DUMMY_N_NEUR = 50
DUMMY_TRIAL_LEN = 700 # to match MC_Maze
DUMMY_FP_LEN = 200 # to match MC_Maze
DUMMY_ALIGN_OFFSET = 250 # to match MC_Maze
DUMMY_N_TRIALS = 10
DUMMY_ITI = (150, 400)
DUMMY_TRIAL_INFO = ['start_time', 'end_time', 'onset', 'split']
DUMMY_BEHAVIOR_LAG = 120 # to match MC_Maze
DUMMY_SPIKE_THRESH = 0.98

np.random.seed(SEED)

# TODO: generate some random data to more thoroughly test all operations
# @pytest.fixture(scope="session")
# def dummy_true_filepath():
#     # Make sure dir exists
#     if not os.path.exists(DATA_DIR):
#         os.mkdir(DATA_DIR)
#     # Choose trial info
#     t = 0
#     trial_data = np.empty((DUMMY_N_TRIALS, len(DUMMY_TRIAL_INFO)))
#     val_idx = np.random.choice(np.arange(DUMMY_N_TRIALS * 0.8))
#     for i in range(DUMMY_N_TRIALS):
#         trial_data[i, 0] = t
#         trial_data[i, 1] = t + DUMMY_TRIAL_LEN + DUMMY_FP_LEN
#         trial_data[i, 2] = t + DUMMY_ALIGN_OFFSET
#         trial_data[i, 3] = 2 if (i > DUMMY_N_TRIALS * 0.8) else 1 if (i in val_idx) else 0
#     t += np.random.randint(*DUMMY_ITI)
#     spikes = np.random.random(
#         (t, DUMMY_N_NEUR)) > DUMMY_SPIKE_THRESH
#     spikes = spikes.astype(float)
#     behavior = np.stack([
#         np.sin(np.arange(t) * 0.01 * np.pi), # pos x
#         np.cos(np.arange(t) * 0.01 * np.pi), # pos y
#         np.cos(np.arange(t) * 0.01 * np.pi), # vel x
#         -np.sin(np.arange(t) * 0.01 * np.pi), # vel y
#     ], axis=-1)
#     save_path = Path(DATA_DIR, DUMMY_RAW_FILE_NAME)
#     np.savez(
#         save_path,
#         spikes=spikes,
#         behavior=behavior,
#         trial_data=trial_data,
#     )
#     return save_path

# @pytest.fixture(scope="session")
# def dummy_nwb_filepath(dummy_true_filepath):
#     if not os.path.exists(DATA_DIR):
#         os.mkdir(DATA_DIR)
    
#     nwb_train = NWBFile(
#         session_description='dummy train data for testing',
#         identifier='train',
#         session_start_time=datetime.now(timezone.utc),
#     )
#     nwb_test = NWBFile(
#         session_description='dummy test data for testing',
#         identifier='test',
#         session_start_time=datetime.now(timezone.utc),
#     )
#     raw_data = np.load(dummy_true_filepath)

#     spikes = raw_data['spikes']
#     behavior = raw_data['behavior']
#     trial_data = raw_data['trial_data']
    
#     first_test = np.nonzero(trial_data[:, 3] == 2)[0][0]
#     test_start = trial_data[0, first_test]

#     train_spikes, test_spikes = np.split(spikes, [test_start])
#     train_behavior, test_behavior = np.split(spikes, [test_start])
#     train_spikes, test_spikes = np.split(spikes, [test_start])

@pytest.fixture(scope="session")
def nlb_filepath():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    download(urls=[DANDISET_URL], output_dir=DATA_DIR, existing='refresh')
    return Path(DATA_DIR, NLB_FILE_NAME)

@pytest.fixture
def nlb_dataset(nlb_filepath):
    dataset = NWBDataset(nlb_filepath)
    return dataset
    
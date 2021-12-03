import os
from pathlib import Path

# HIGH-LEVEL PRAGMAS: ==========================================

PRAGMA_EDOUARD_YANN_CC = 3  # 3 if MISTlab RTX3080, 2 if ComputeCanada, 1 if Edouard, 0 if Yann  # TODO: remove for release
PRAGMA_SERVER_ON_EDOUARD_YANN = 0  # 1 is server on Edouard's PC, 0 if server on Yann's PC
RUN_NAME = "SACv1_SPINUP_4_LIDAR_Yann_test_sane_version"

BUFFERS_MAXLEN = 2000  # Maximum length of the local buffers for RolloutWorkers, Server and TrainerInterface
RW_MAX_SAMPLES_PER_EPISODE = 1000  # If this number of timesteps is reached, the RolloutWorker will reset the episode

PRAGMA_TM2020_TMNF = True  # True if TM2020, False if TMNF
PRAGMA_LIDAR = True  # True if Lidar, False if images
PRAGMA_CUDA_TRAINING = True  # True if CUDA, False if CPU (trainer)
PRAGMA_CUDA_INFERENCE = False  # True if CUDA, False if CPU (rollout worker)

PRAGMA_GAMEPAD = True  # True to use gamepad, False to use keyboard

CONFIG_COGNIFLY = False  # if True, will override config with Cognifly's config

PRAGMA_DCAC = False  # True for DCAC, False for SAC

LOCALHOST_WORKER = False  # set to True for RolloutWorkers on the same machine as the Server
PUBLIC_IP_REDIS = "173.179.182.4" if PRAGMA_SERVER_ON_EDOUARD_YANN else "45.74.221.204"  # IP Edouard

REDIS_IP_FOR_WORKER = PUBLIC_IP_REDIS if not LOCALHOST_WORKER else "127.0.0.1"
REDIS_IP_FOR_TRAINER = PUBLIC_IP_REDIS

# ENVIRONMENT: =======================================================

LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # [88, 88, 88] for tiny road, [55, 55, 55] FOR BASIC ROAD
REWARD_END_OF_TRACK = 0  # bonus reward at the end of the track
CONSTANT_PENALTY = 0  # should be <= 0 : added to the reward at each time step
SLEEP_TIME_AT_RESET = 1.5  # 1.5 to start in a Markov state with the lidar, 0.0 for saving replays
ACT_BUF_LEN = 2
IMG_HIST_LEN = 4

# CRC DEBUGGING AND BENCHMARKING: ==============================

CRC_DEBUG = False  # Only for checking the consistency of the custom networking methods, set it to False otherwise. Caution: difficult to handle if reset transitions are collected.
CRC_DEBUG_SAMPLES = 1000  # Number of samples collected in CRC_DEBUG mode
PROFILE_TRAINER = False  # Will profile each epoch in the Trainer when True
BENCHMARK = False  # The environment will be benchmarked when this is True

# FILE SYSTEM: =================================================

PATH_FILE = Path(__file__)  # TODO: this won't work with PyPI or normal install
print(f"DEBUG: PATH_FILE:{PATH_FILE}")
PATH_DATA = PATH_FILE.absolute().parent.parent / 'data'
print(f"DEBUG: PATH_DATA:{PATH_DATA}")

MODEL_HISTORY = 10  # 0 for not saving history, x for saving model history every x new model received by RolloutWorker

MODEL_PATH_WORKER = str(PATH_DATA / "weights" / (RUN_NAME + ".pth"))
MODEL_PATH_SAVE_HISTORY = str(PATH_DATA / "weights" / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(PATH_DATA / "weights" / (RUN_NAME + "_t.pth"))
CHECKPOINT_PATH = str(PATH_DATA / "checkpoint" / RUN_NAME)
DATASET_PATH = str(PATH_DATA / "dataset")
REWARD_PATH = str(PATH_DATA / "reward" / "reward.pkl")

if PRAGMA_EDOUARD_YANN_CC == 2:  # Override some of these for Compute Canada
    if PRAGMA_SERVER_ON_EDOUARD_YANN == 1:  # Edouard
        MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl_edouard/data/" + (RUN_NAME + "_t.pth")
        CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl_edouard/data/" + RUN_NAME
        REWARD_PATH = r"/home/yannbout/scratch/base_tmrl_edouard/data/reward.pkl"
    else:  # Yann
        MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl/data/" + (RUN_NAME + "_t.pth")
        CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl/data/" + RUN_NAME
        REWARD_PATH = r"/home/yannbout/scratch/base_tmrl/data/reward.pkl"
elif PRAGMA_EDOUARD_YANN_CC == 3:  # Override some of these for MIST Benchbot
    MODEL_PATH_TRAINER = r"/home/ybouteiller/tmrl/data/" + (RUN_NAME + "_t.pth")
    CHECKPOINT_PATH = r"/home/ybouteiller/tmrl/data/" + RUN_NAME
    REWARD_PATH = r"/home/ybouteiller/tmrl/data/reward.pkl"

# WANDB: =======================================================

WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = "tmrl"
WANDB_ENTITY = "tmrl"
WANDB_KEY = "df28d4daa98d2df2557d74caf78e40c68adaf288"

os.environ['WANDB_API_KEY'] = WANDB_KEY

# NETWORKING: ==================================================

PRINT_BYTESIZES = True

PORT_TRAINER = 55555  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = 55556  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 536870912  # 268435456  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = 12  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = 300.0
SOCKET_TIMEOUT_ACCEPT_TRAINER = 300.0
SOCKET_TIMEOUT_CONNECT_ROLLOUT = 300.0
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = 300.0  # socket waiting for rollout workers closed and restarted at this interval
SOCKET_TIMEOUT_COMMUNICATE = 30.0
SELECT_TIMEOUT_OUTBOUND = 30.0
SELECT_TIMEOUT_PING_PONG = 60.0
ACK_TIMEOUT_WORKER_TO_REDIS = 300.0
ACK_TIMEOUT_TRAINER_TO_REDIS = 300.0
ACK_TIMEOUT_REDIS_TO_WORKER = 300.0
ACK_TIMEOUT_REDIS_TO_TRAINER = 300.0
RECV_TIMEOUT_TRAINER_FROM_SERVER = 600.0
RECV_TIMEOUT_WORKER_FROM_SERVER = 600.0
WAIT_BEFORE_RECONNECTION = 10.0
LOOP_SLEEP_TIME = 1.0

# # COGNIFLY: ==================================================== (TODO)
#
# if CONFIG_COGNIFLY:
#
#     if PRAGMA_EDOUARD_YANN_CC == 0:  # Yann  # TODO: CC
#         MODEL_PATH_WORKER = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgn.pth"
#         MODEL_PATH_TRAINER = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgnt.pth"
#         CHECKPOINT_PATH = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgn0"
#
#     WANDB_RUN_ID = "SAC_cognifly_test_2"
#     WANDB_PROJECT = "cognifly"
#
#     TRAIN_MODEL = Mlp
#     POLICY = MlpPolicy
#     BENCHMARK = False
#
#     ACT_BUF_LEN = 4
#     IMGS_OBS = 0
#
#     INT = partial(CogniflyInterfaceTask1, img_hist_len=0)
#
#     from rtgym import DEFAULT_CONFIG_DICT
#     CONFIG_DICT = DEFAULT_CONFIG_DICT
#     CONFIG_DICT["interface"] = INT
#
#     CONFIG_DICT["time_step_duration"] = 0.05
#     CONFIG_DICT["start_obs_capture"] = 0.05
#     CONFIG_DICT["time_step_timeout_factor"] = 1.0
#     CONFIG_DICT["ep_max_length"] = 200
#     CONFIG_DICT["act_buf_len"] = ACT_BUF_LEN
#     CONFIG_DICT["reset_act_buf"] = False
#     CONFIG_DICT["act_in_obs"] = True
#     CONFIG_DICT["benchmark"] = BENCHMARK
#     CONFIG_DICT["wait_on_done"] = True
#
#     SAMPLE_COMPRESSOR = get_local_buffer_sample_cognifly
#     OBS_PREPROCESSOR = obs_preprocessor_cognifly
#     SAMPLE_PREPROCESSOR = None
#
#     MEM = MemoryCognifly
#
#     MEMORY = partial(MEM,
#                      path_loc=DATASET_PATH,
#                      imgs_obs=IMGS_OBS,
#                      act_buf_len=ACT_BUF_LEN,
#                      obs_preprocessor=OBS_PREPROCESSOR,
#                      sample_preprocessor=SAMPLE_PREPROCESSOR,
#                      crc_debug=CRC_DEBUG
#                      )

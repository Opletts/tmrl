import time
from argparse import ArgumentParser, ArgumentTypeError
import logging
import json

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.tools.record import record_reward_dist
from tmrl.tools.check_environment import check_env_tm20lidar
from tmrl.envs import GenericGymEnv
from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.util import partial
# import networks as core
import spinup_ddpg_core as core

def main(args):
    if args.worker:
        config = cfg_obj.CONFIG_DICT
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": config}),
                           actor_module_cls=partial(core.MLPActor),
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           min_samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
        rw.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='runs inference without training')
    arguments = parser.parse_args()
    logging.info(arguments)

    main(arguments)
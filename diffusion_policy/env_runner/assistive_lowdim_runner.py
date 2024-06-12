import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.zarr_recording_wrapper import ZarrRecordingWrapper

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from safety_model.train import *

module_logger = logging.getLogger(__name__)

class AssistiveLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            task_name: str = "Feeding",
            robot_name: str = "Jaco",
            safety_model_path: str = None,
            safety_model_kwargs: dict = None,
            record_zarr: bool = False,
            **kwargs,
        ):
        super().__init__(output_dir)
        self.safety_model = None
        if safety_model_path is not None:
            self.safety_model = torch.load(safety_model_path)
            self.safety_model.override_params(safety_model_kwargs)

        if n_envs is None:
            n_envs = n_train + n_test

        self.record_zarr = record_zarr
        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            import assistive_gym
            assert task_name in assistive_gym.tasks, assistive_gym.tasks
            assert robot_name in assistive_gym.robots, assistive_gym.robots
            env = gym.make(f"assistive_gym:{task_name}{robot_name}-v1")

            return MultiStepWrapper(
                VideoRecordingWrapper(
                    ZarrRecordingWrapper(env, file_path=None),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                horizon=self.safety_model.horizon if self.safety_model else None,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, init_qpos=None, init_qvel=None, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper), env.env
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f"train_episode_{i}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                #TODO(aty): it's actually <TimeLimit<FeedingJacoEnv<FeedingJaco-v1>>>, check that instead!
                #from assistive_gym.envs.env import AssistiveEnv
                #assert isinstance(env.env.env, AssistiveEnv), env.env.env

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render, record_zarr=record_zarr):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper), env.env
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f"test_episode_{i}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                if record_zarr:
                    assert isinstance(env.env.env, ZarrRecordingWrapper), env.env.env
                    filename = pathlib.Path(output_dir).joinpath(
                        'zarr_recording', f"test_episode_{i}.zarr")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.env.file_path = filename

                #TODO(aty): it's actually <TimeLimit<FeedingJacoEnv<FeedingJaco-v1>>>, check that instead!
                #from assistive_gym.envs.env import AssistiveEnv
                #assert isinstance(env.env.env, AssistiveEnv), env.env.env

                # set seed
                assert isinstance(env, MultiStepWrapper), env
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs, (len(this_init_fns), n_envs)

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            if self.safety_model:
                self.safety_model.reset()

            assert obs.shape[0] == n_envs, (n_envs, obs.shape, this_global_slice)
            safe_mask = np.ones((n_envs,), dtype=np.bool_)
            max_uncertainty_scores = np.ones((n_envs,), dtype=np.float64) * (-np.inf)

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval AssistiveLowdimRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                if self.safety_model:
                    try:
                        device = self.safety_model.device
                    except:
                        device = "cpu"

                    obs_list = env.call('get_attr', 'cached_obs')
                    act_list = env.call('get_attr', 'cached_actions')
                    from gym.vector.utils.numpy_utils import concatenate, create_empty_array
                    obs = concatenate(obs_list, None, env.single_observation_space).astype(np.float32)
                    act = concatenate(act_list, None, env.single_action_space).astype(np.float32)
                    if isinstance(self.safety_model, EnsembleStatePredictionSM):
                        is_safe, stats = self.safety_model.check_safety_runner({
                            'obs': torch.from_numpy(obs).to(device),
                            'action': action_dict['action_pred'].to(device),
                        }, return_stats=True)
                        max_uncertainty_scores = np.maximum(list(stats.values())[0], max_uncertainty_scores)
                    elif isinstance(self.safety_model, VariationalAutoEncoderSM):

                        if self.safety_model.in_act_horizon > self.safety_model.in_obs_horizon:
                            n_obs_steps = self.n_obs_steps
                            n_action_steps = self.n_action_steps
                            #assert action_dict['action_pred'].shape[1] == horizon
                            #breakpoint()
                            action_vec = torch.cat([torch.from_numpy(act).to(device), action_dict['action_pred'][:, n_obs_steps:].to(device)], dim=1)
                        else:
                            action_vec = torch.from_numpy(act).to(device)
                        batch = {
                            'obs': torch.from_numpy(obs).to(device),
                            'action': action_vec,
                        }
                        is_safe, stats = self.safety_model.check_safety_runner(batch, return_stats=True)
                        #    {
                        #    'obs': torch.from_numpy(obs).to(device),
                        #    'action': torch.from_numpy(act).to(device),
                        #}, return_stats=True)
                        max_uncertainty_scores = np.maximum(stats["pwkl"], max_uncertainty_scores)
                    else:
                        raise NotImplementedError()

                    # is_dangerous = stats['std'] > 0.08
                    safe_mask &= is_safe

                # Apply safety mask
                action[~safe_mask] = 0.0

                # step env
                obs, reward, done, info = env.step(action)

                # NOTE(aty): Unsafe envs are automatically done (aka halted)
                done = np.all(done | ~safe_mask)

                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

            def to_range(s: slice):
                return range(s.start, s.stop)

            # add "task_halted" to info
            for i_info, i_env in zip(to_range(this_global_slice), to_range(this_local_slice)):
                last_info[i_info]['task_halted'] = not safe_mask[i_env]
                last_info[i_info]['max_uncertainty_score'] = max_uncertainty_scores[i_env]

        # log
        log_data = dict()
        prefix_total_reward_map = collections.defaultdict(list)
        prefix_total_length_map = collections.defaultdict(list)
        prefix_timeout_map = collections.defaultdict(list)
        prefix_halted_map = collections.defaultdict(list)
        prefix_success_map = collections.defaultdict(list)
        prefix_failure_map = collections.defaultdict(list)
        prefix_max_uncertainty_score = collections.defaultdict(list)
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.sum(this_rewards)
            prefix_total_reward_map[prefix].append(total_reward)
            prefix_total_length_map[prefix].append(len(this_rewards))

            task_success = last_info[i]['task_success']
            prefix_success_map[prefix].append(task_success)
            task_halted = last_info[i]['task_halted']
            prefix_halted_map[prefix].append(task_halted)
            task_timeout = (len(this_rewards) == 200) and not task_halted
            prefix_timeout_map[prefix].append(task_timeout)
            task_failed = not (task_success or task_halted or task_timeout)
            prefix_failure_map[prefix].append(task_failed)
            prefix_max_uncertainty_score[prefix].append(last_info[i]['max_uncertainty_score'])
            #mean_force_on_human = last_info[i]['total_force_on_human_sum'] / len(this_rewards)
            #prefix_mean_forces_map[prefix].append(mean_force_on_human)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in prefix_total_reward_map.items():
            value = np.array(value)
            log_data[prefix + 'mean_score'] = np.mean(value)
            log_data[prefix + 'std_score'] = np.std(value)

        for prefix, value in prefix_total_length_map.items():
            value = np.array(value)
            log_data[prefix + 'mean_len'] = np.mean(value)
            log_data[prefix + 'min_len'] = np.min(value)
            log_data[prefix + 'max_len'] = np.max(value)

        #
        #   SUCCESS - task completed successfully.
        #   HALTED  - task failed but the failure is detected and handled correctly.
        #   TIMEOUT - task failed due to the timeout. Policy just got lost. Bad, but not the worst.
        #   FAILURE - task failed. Worst case.
        #
        #   SUCCESS + TIMEOUT + HALTED + FAILURE = 100%
        #
        for prefix, value in prefix_success_map.items():
            success = np.array(value)
            log_data[prefix + 'success_episodes'] = list(success)
            log_data[prefix + 'success'] = np.mean(success)
            log_data[prefix + 'success_std'] = np.std(success)

        for prefix, value in prefix_halted_map.items():
            halted = np.array(value)
            log_data[prefix + 'halted_episodes'] = list(halted)
            log_data[prefix + 'halted'] = np.mean(halted)
            log_data[prefix + 'halted_std'] = np.std(halted)

        for prefix, value in prefix_timeout_map.items():
            timeout = np.array(value)
            log_data[prefix + 'timeout_episodes'] = list(timeout)
            log_data[prefix + 'timeout'] = np.mean(timeout)
            log_data[prefix + 'timeout_std'] = np.std(timeout)

        for prefix, value in prefix_failure_map.items():
            failure = np.array(value)
            log_data[prefix + 'failure_episodes'] = list(failure)
            log_data[prefix + 'failure'] = np.mean(failure)
            log_data[prefix + 'failure_std'] = np.std(failure)

        """
        for prefix, value in prefix_mean_forces_map.items():
            mean_forces = np.array(value)
            name = prefix + f'mean_mean_forces'
            log_data[name] = np.mean(mean_forces)
        """
        for prefix, value in prefix_max_uncertainty_score.items():
            success = np.array(prefix_success_map[prefix]).astype(np.bool_)
            uncertainties = np.array(value)
            log_data[prefix + 'success_uncertainty_score_max_max'] = np.max(uncertainties[success]) if success.sum() > 0 else 0.0
            log_data[prefix + 'success_uncertainty_score_max_min'] = np.min(uncertainties[success]) if success.sum() > 0 else 0.0
            log_data[prefix + 'timeout_uncertainty_score_max_max'] = np.max(uncertainties[~success]) if (~success).sum() > 0 else 0.0
            log_data[prefix + 'timeout_uncertainty_score_max_min'] = np.min(uncertainties[~success]) if (~success).sum() > 0 else 0.0

        return log_data

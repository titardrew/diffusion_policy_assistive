import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.assistive_lowdim_runner import AssistiveLowdimRunner

def test():
    def _test(task_config):
        from omegaconf import OmegaConf
        from pathlib import Path
        ROOT_DIR = Path(__file__).absolute().parent.parent

        cfg_path = ROOT_DIR / "diffusion_policy" / "config" / "task" / task_config
        assert cfg_path.exists()
        cfg = OmegaConf.load(cfg_path)
        cfg['n_obs_steps'] = 1
        cfg['n_action_steps'] = 1
        cfg['past_action_visible'] = False
        runner_cfg = cfg['env_runner']
        runner_cfg['n_train'] = 1
        runner_cfg['n_test'] = 0
        del runner_cfg['_target_']
        runner = AssistiveLowdimRunner(
            **runner_cfg, 
            output_dir='/tmp/test')

        print(cfg['task_name'])
        self = runner
        env = self.env
        env.seed(seeds=self.env_seeds)
        obs = env.reset()
        print(env.observation_space.shape, env.action_space)
        #import ipdb; ipdb.set_trace()
    
    _test("assistive_arm_manipulation_jaco_lowdim.yaml")
    _test("assistive_bed_bathing_jaco_lowdim.yaml")
    _test("assistive_feeding_jaco_lowdim.yaml")
    _test("assistive_scratch_itch_jaco_lowdim.yaml")
    _test("assistive_drinking_jaco_lowdim.yaml")

if __name__ == '__main__':
    test()
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.assistive_lowdim_runner import AssistiveLowdimRunner

def test():
    from omegaconf import OmegaConf
    from pathlib import Path
    ROOT_DIR = Path(__file__).absolute().parent.parent

    cfg_path = ROOT_DIR / "diffusion_policy" / "config" / "task" / "assistive_feeding_jaco_lowdim.yaml"
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

    self = runner
    env = self.env
    env.seed(seeds=self.env_seeds)
    obs = env.reset()

    #import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    test()
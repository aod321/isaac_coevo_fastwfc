import time
from draw import *
from vec_env_fastwfc import PCGVecEnv, VecAdapter
from stable_baselines3 import PPO
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        # start timer
        self.start = time.time()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.training_env.headless == False:
            self.training_env.render()

        if self.n_calls % self.check_freq == 0:
            end = time.time()
            
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Time taken: {}".format(end - self.start))
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                
                result = {
                    "reward": mean_reward,
                    "best mean reward": self.best_mean_reward,
                }
                
                if mean_reward > self.best_mean_reward and self.num_timesteps > 10000:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
        return True

if __name__ == "__main__":
    log_dir = "./training_logs"
    timesteps = 1500000
    check_freq = 1000
    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir, check_freq=check_freq)

    m_env = PCGVecEnv(headless_ = True, compute_device_id=1, graphics_device_id=1)
    m_env = VecMonitor(VecAdapter(m_env), filename=log_dir)
    m_env.reset()

    model_ref = None
    best_step = 0
    model = PPO('CnnPolicy', env=m_env, batch_size = 1024, device="cuda:1")
    # model = PPO.load("./training_logs/best_model5.zip", env=m_env, batch_size = 1024)
    model_ref = model
    best_step = 0
    print("model.n_steps : ", model.n_steps)
    print("model.batch_size : ", model.batch_size)

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    # model.eval_env




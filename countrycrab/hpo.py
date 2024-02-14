# This code is based and adapted on previous code realized by Sergey that can be found here:
# https://github.hpe.com/sergey-serebryakov/ray_tune/blob/master/xgb.py
import os
# select which GPU are you using
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import typing as t
import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from ray import tune
from ray.air import RunConfig
from compiler import get_instance_names
from solver import camsat
from ray.tune import Callback
from ray.tune.experiment import Trial
from enum import Enum


class Mode(Enum):
    MIN = 'min'
    MAX = 'max'


class MLflowCallback(Callback):
    """A callback that Ray Tune can use to report progress to MLflow.

    Usage example:
    >>> run_config = RunConfig(
    ...     local_dir='ray_results',
    ...     callbacks=[MLflowCallback('eval_loss', Mode.MIN)]
    ... )

    Args:
        metric: Metric that is being optimized.
        mode: Either this metric should be minimized or maximized.
    """

    def __init__(self, metric: str, mode: Mode) -> None:
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.best_value: t.Optional = None
        self.trial_index = 0

    def on_trial_result(self, iteration: int, trials: t.List[Trial], trial: Trial, result: t.Dict, **info) -> None:
        self.trial_index += 1
        value = result[self.metric]
        mlflow.log_metric(self.metric, value, self.trial_index)

        if self.best_value is None:
            self.best_value = value
        elif self.mode == Mode.MIN:
            self.best_value = min(self.best_value, value)
        else:
            self.best_value = max(self.best_value, value)
        mlflow.log_metric(f'best_{self.metric}', self.best_value, self.trial_index)

        mlflow.log_metric('trial_time_s', result['time_total_s'], self.trial_index)


def optimize(scheduler_name: t.Optional[str] = None) -> None:
    # i suggest to test that the generated path is correct
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    I_V_opt, I_V_final = get_instance_names(path, k =3)
    I_V_opt_flat = [item for sublist in list(I_V_opt.values()) for item in sublist]
    # test only size 20
    #instance_list = I_V_opt[20]      
    search_space = {
        "noise": tune.uniform(0.2, 2),
        "instance": tune.grid_search(I_V_opt_flat),
        #"instance": tune.grid_search(instance_list),
    }

    # you can select how much of the gpu memory is used by any instance. This is a bit of a trial and error, start high and try to decrease it until it works basically
    resources_per_trial = {'gpu':0.5}
    objective_fn = tune.with_resources(camsat, resources_per_trial)
    # Need this to log RayTune artifacts into MLflow runs' artifact store.

    # set experiment name
    run_config = RunConfig(
        name = 'hpo_3sat_uniform_noise',
        local_dir=local_file_uri_to_path(mlflow.active_run().info.artifact_uri),
        log_to_file=True,
    )

    tuner = tune.Tuner(
        # set params and number of samples
        tune.with_parameters(
            objective_fn,
            params={'max_flips': 100000, 'max_runs': 1000, 'batch_size': 1, 'task': 'hpo'}
        ),
        # Tuning configuration.
        tune_config=tune.TuneConfig(
            metric="tts",
            mode="min",
            num_samples=50,
        ),
        # Hyperparameter search space.
        param_space=search_space,
        # Runtime configuration.
        run_config=run_config
    )
    _ = tuner.fit()


if __name__ == "__main__":
    # set folder where to save results
    tracking_uri = os.path.join(os.path.expanduser('~/'), 'projects/camsat/camsat_v2/data/experiments/tcas/')
    print(tracking_uri) 
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        optimize()
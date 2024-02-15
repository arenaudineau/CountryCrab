# This code is based and adapted on previous code realized by Sergey that can be found here:
# https://github.hpe.com/sergey-serebryakov/ray_tune/blob/master/xgb.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import typing as t
import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow import MlflowClient
from ray import tune
from ray.air import RunConfig
from compiler import get_instance_names
from solver import camsat
import json


def schedule(scheduler_name: t.Optional[str] = None) -> None:     
    with open('config.json', 'r') as f:
        config = json.load(f)
    instance_list = config["instance_list"]

    # the only search space parameters are the instances in the case of the scheduler
    search_space = {
        "instance": tune.grid_search(instance_list),
    }

    resources_per_trial = {'gpu':0.2}
    objective_fn = tune.with_resources(camsat, resources_per_trial)
    # Need this to log RayTune artifacts into MLflow runs' artifact store.
    run_config = RunConfig(
        name = config["experiment_name"],
        local_dir=local_file_uri_to_path(mlflow.active_run().info.artifact_uri),
        log_to_file=True,
    )
    tuner = tune.Tuner(
        
        tune.with_parameters(
            objective_fn,
            params=config
        ),
        # Tuning configuration.
        tune_config=tune.TuneConfig(
            metric="its",
            mode="min",
            num_samples=1,
        ),
        # Hyperparameter search space.
        param_space=search_space,
        # Runtime configuration.
        run_config=run_config
    )
    _ = tuner.fit()


if __name__ == "__main__":
    tracking_uri = os.path.join(os.path.expanduser('~/'), 'projects/camsat/camsat_v2/data/experiments/tcas/')
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        schedule()
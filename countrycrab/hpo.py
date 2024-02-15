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
import json
import argparse


def optimize(config_fname: t.Optional[str] = None) -> None:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_fname)
    with open(config_path, 'r') as f:
        config = json.load(f)
    instance_list = config["instance_list"]   
    min_noise = config["min_noise"]
    max_noise = config["max_noise"]
    search_space = {
        "noise": tune.uniform(min_noise, max_noise),
        "instance": tune.grid_search(instance_list),
    }

    # you can select how much of the gpu memory is used by any instance. This is a bit of a trial and error, start high and try to decrease it until it works basically
    resources_per_trial = {'gpu':0.5}
    objective_fn = tune.with_resources(camsat, resources_per_trial)
    # Need this to log RayTune artifacts into MLflow runs' artifact store.

    # set experiment name
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


def main(tracking_uri, config_fname):
    tracking_uri_exapanded = os.path.join(os.path.expanduser('~/'), tracking_uri)
    mlflow.set_tracking_uri(tracking_uri_exapanded)
    with mlflow.start_run():
        optimize(config_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tracking_uri', type=str, default='experiments/defaults/',
                        help='The tracking URI to use for the experiments, defaults to experiments/defaults/ if not provided.')
    
    parser.add_argument('--config_fname', type=str, default='debug.json',
                    help='The name of the configuration file stored in the config folder, defaults to debug.json if not provided.')


    args = parser.parse_args()

    main(args.tracking_uri)
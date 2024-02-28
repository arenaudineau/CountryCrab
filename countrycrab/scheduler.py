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
from solver import solve
import argparse
import json


def schedule(config_fname: t.Optional[str] = None) -> None:   
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_fname)
    with open(config_path, 'r') as f:  
        config = json.load(f)
    instance_list = config["instance_list"]

    # the only search space parameters are the instances in the case of the scheduler
    search_space = {
        "instance": tune.grid_search(instance_list),
    }

    # set resource per trial based on experiments. A high value is used to ensure that the experiment runs but it can be decreased to optimize the use of the GPU
    resources_per_trial = {'gpu':0.2}
    objective_fn = tune.with_resources(solve, resources_per_trial)
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


def main(tracking_uri, config):
    module_path = os.path.abspath(os.path.join(""))
    tracking_uri_exapanded = module_path+'/data/experiments/' + tracking_uri 
    mlflow.set_tracking_uri(tracking_uri_exapanded)
    with mlflow.start_run():
        schedule(config)
    print('Experiment completed')
    print(tracking_uri_exapanded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tracking_uri', type=str, default='experiments/defaults/',
                        help='The tracking URI to use for the experiments, defaults to experiments/defaults/ if not provided.')
    
    parser.add_argument('--config', type=str, default='debug.json',
                    help='The name of the configuration file stored in the config folder, defaults to debug.json if not provided.')


    args = parser.parse_args()

    main(args.tracking_uri, args.config)
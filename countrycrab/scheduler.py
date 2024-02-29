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
import numpy as np


def schedule(config_fname: t.Optional[str] = None) -> None:   
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_fname)
    with open(config_path, 'r') as f:  
        config = json.load(f)
    instance_list = config["instance_list"]
    if 'min_noise' in config:
        min_noise = config["min_noise"]
        max_noise = config["max_noise"]
        num_samples = config['num_samples']
    elif 'noise' in config:
        min_noise = config['noise']
        max_noise = config['noise']
        num_samples = 1
    else:
        min_noise = 0.8
        max_noise = 0.8
        num_samples = 1

    # load parameters to save them in the configuration    
    # p_solve is the probability of solving the problem
    p_solve = config.get("p_solve", 0.99)
    # task is the type of task to be performed
    task = config.get("task", "debug")
    # max runs is the number of parallel initialization (different inputs)
    max_runs = config.get("max_runs", 100)
    # max_flips is the maximum number of iterations
    max_flips = config.get("max_flips", 1000)
    # noise profile
    noise_dist = config.get("noise_distribution",'normal')
    # architecture specifics
    n_words = config.get("n_words", -1) # -1 means that the number of words is equal to the number of clauses
    n_cores = config.get("n_cores", 1)
    scheduling = config.get("scheduling", "fill_first")
    # heuristics specification
    heuristic_name = config.get("heuristic", 'walksat_m')
    compiler_name = config.get("compiler", 'compile_walksat_m')

    # the only search space parameters are the instances and the noise
    search_space = {
        "noise": tune.grid_search(np.linspace(min_noise, max_noise, num_samples).tolist()), # note that uniform sampling may work better for some experiments
        "instance": tune.grid_search(instance_list),
        "p_solve": p_solve,
        "task": task,
        "max_runs": max_runs,
        "max_flips": max_flips,
        "heuristic": heuristic_name,
        "compiler": compiler_name,
        "noise_distribution": noise_dist,
        "n_cores": n_cores,
        "n_words": n_words,
        "scheduling": scheduling
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
# This code is based and adapted on previous code realized by Sergey that can be found here:
# https://github.hpe.com/sergey-serebryakov/ray_tune/blob/master/xgb.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import typing as t
import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from ray import tune
from ray.air import RunConfig
from compiler import get_instance_names
from solver import camsat


def schedule(scheduler_name: t.Optional[str] = None) -> None:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    I_V_opt, I_V_final = get_instance_names(path, k =3)
    I_V_final_flat = [item for sublist in list(I_V_final.values()) for item in sublist]
    # test only size 20
    #instance_list = [I_V_opt[20][0]]      
    search_space = {
        "instance": tune.grid_search(I_V_final_flat),
        "n_words": 128,
        "n_cores":20,
    }

    resources_per_trial = {'gpu':0.2}
    objective_fn = tune.with_resources(camsat, resources_per_trial)
    # Need this to log RayTune artifacts into MLflow runs' artifact store.
    run_config = RunConfig(
        name = '3sat_hierarchical_exp2',
        local_dir=local_file_uri_to_path(mlflow.active_run().info.artifact_uri),
        log_to_file=True,
    )

    tuner = tune.Tuner(
        
        tune.with_parameters(
            objective_fn,
            params={'max_runs': 1000, 'batch_size': 1, 'task': 'solve','hp_location':'/home/pedretti/projects/camsat/camsat_v2/data/experiments/3sat_hierarchical_hpo.csv','scheduling':'fill_first'}
        ),
        
        # Tuning configuration.
        tune_config=tune.TuneConfig(
            metric="tts_max",
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
    tracking_uri = os.path.join(os.path.expanduser('~/'), 'projects/camsat/camsat_v2/data/experiments/hierarchical/')
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        schedule()
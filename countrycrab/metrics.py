import numpy as np
from countrycrab.qbert.qbert_metrics.sample_its_metric import SampleITSMetric
    
def get_standard_qbert_results(results_cc: numpy.array) -> dict:
    """Convert results from Countrycrab to QBERT metrics expected format.

    Args:
        results_cc (numpy.array): Results from Countrycrab

    Returns:
        results_qbert (dict): Results format for QBERT metrics.
    """
    results_qbert = {}
    for single_try in range(results_cc.shape[0]):
        best_result = min(results_cc[single_try, :])
        try:
            number_iterations = list(results_cc[single_try, :]).index(0) + 1
        except ValueError:
            number_iterations = results_cc.shape[1] + 1
        results_qbert[single_try] = {
            "best_result": best_result,
            "number_iterations": number_iterations,
        }
    return results_qbert


def vector_its_bayesian(violated_constr_mat,config):
    its_metric = SampleITSMetric()
    # problem instance name
    problem_ids = ["<problem instance name>"]
    # problem instance best known solution
    best_known_solutions = [0]
    results_qbert_single_problem = get_standard_qbert_results(violated_constr_mat)
    results_qbert = {"<problem instance name>": results_qbert_single_problem}

    its, its_err = its_metric.calc(results_qbert, problem_ids, best_known_solutions)
    return its, its_err

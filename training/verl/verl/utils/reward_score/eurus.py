from util.evaluation_utils.code_util import evaluate_code
from util.evaluation_utils.math_util import evaluate_math

def eurus_compute_score(ability, solution_str, ground_truth):
    if ability == 'math':
        res, _, _ = evaluate_math(solution_str, ground_truth)
        return float(res)
    elif ability == 'code':
        res, _ = evaluate_code(solution_str, ground_truth)
        return float(res)
    else:
        raise NotImplementedError
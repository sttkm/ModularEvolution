import sys
import os

import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)

from arguments.evogym_modular import get_args

from parallel import BatchBayesianOptimizationParallel
from experiment_utils import initialize_experiment

from evaluator import ppoConfig, EvogymStructureEvaluator

import evogym.envs

# from sample_robot2 import Robot



# def acquisition_thompson(trees, X, indices, min_variance=0.01):
#     values = np.zeros(len(X))
#     tree_selected_count = np.random.multinomial(len(trees), np.ones(len(trees)) / len(trees), size=(len(X)))

#     for i, tree in enumerate(trees):
#         node_mean = tree.tree_.value.flatten()
#         node_variance = tree.tree_.impurity
#         node_variance = np.where(node_variance < min_variance, min_variance, node_variance)
#         node_values = np.random.normal(node_mean, node_variance ** (1/2))

#         leaf_indices = tree.tree_.apply(X)
#         tree_value = node_values[leaf_indices]
    
#         values += tree_value * tree_selected_count[:, i]

#     values /= len(trees)

#     indice_ = np.argmax(values)
#     value = values[indice_]
#     indice = indices[indice_]

#     return indice, value


def acquisition_thompson(trees, X, get_n, min_variance=0.01):
    values = np.zeros(len(X))
    tree_selected_count = np.random.multinomial(len(trees), np.ones(len(trees)) / len(trees), size=(len(X)))

    for i, tree in enumerate(trees):
        node_mean = tree.tree_.value.flatten()
        node_variance = tree.tree_.impurity
        node_variance = np.where(node_variance < min_variance, min_variance, node_variance)
        node_values = np.random.normal(node_mean, node_variance ** (1/2))

        leaf_indices = tree.tree_.apply(X)
        tree_value = node_values[leaf_indices]
    
        values += tree_value * tree_selected_count[:, i]

    values /= len(trees)

    argsort = np.argsort(values)[-get_n:][::-1]
    values = values[argsort]
    return argsort, values



from scipy.stats import norm

def get_ei(trees, X, y_best, min_variance=0.01):
    ### 
    # https://github.com/jungtaekkim/On-Uncertainty-Estimation-by-Tree-based-Surrogate-Models-in-SMO/blob/main/src/tree_based_surrogates.py

    mean = np.zeros(len(X))
    variance = np.zeros(len(X))

    for tree in trees:
        leaf_indices = tree.tree_.apply(X)
        variance_tree = tree.tree_.impurity[leaf_indices]
        mean_tree = tree.tree_.value.flatten()[leaf_indices]

        variance_tree = np.where(variance_tree < min_variance, min_variance, variance_tree)

        mean += mean_tree
        variance += variance_tree + mean_tree ** 2

    mean /= len(trees)
    variance /= len(trees)
    
    variance -= mean ** 2.0
    variance = np.where(variance < 0.0, 0.0, variance)
    std = variance ** 0.5

    improvement = (mean - y_best)
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei


def get_params(trees, X, min_variance=0.01):
    ### 
    # https://github.com/jungtaekkim/On-Uncertainty-Estimation-by-Tree-based-Surrogate-Models-in-SMO/blob/main/src/tree_based_surrogates.py

    mean = np.zeros(len(X))
    variance = np.zeros(len(X))

    for tree in trees:
        leaf_indices = tree.tree_.apply(X)
        variance_tree = tree.tree_.impurity[leaf_indices]
        mean_tree = tree.tree_.value.flatten()[leaf_indices]

        variance_tree = np.where(variance_tree < min_variance, min_variance, variance_tree)

        mean += mean_tree
        variance += variance_tree + mean_tree ** 2

    mean /= len(trees)
    variance /= len(trees)
    
    variance -= mean ** 2.0
    variance = np.where(variance < 0.0, 0.0, variance)
    std = variance ** 0.5
    return mean, std

def bacth_thompson(mean, std, observed_indices, batch_size):
    value_sample = np.random.normal(mean, std, (batch_size, len(mean)))
    indices_obs = []
    values_obs = []
    for b in range(batch_size):
        argsort = np.argsort(value_sample[b])
        i = 1
        while argsort[-i] in observed_indices or argsort[-i] in indices_obs:
            i += 1
        indices_obs.append(argsort[-i])
        values_obs.append(value_sample[b, argsort[-i]])

    ucb = mean + std

    return indices_obs, values_obs, ucb

def acquision(surrogate_model, features, observed_indices, batch_size):
    mean, std = get_params(surrogate_model, features)
    indices_obs, values_obs, ucb = bacth_thompson(mean, std, observed_indices, batch_size)
    return indices_obs, values_obs, ucb



def ei_thompson(trees, X, y_best, observed_indices, batch_size, min_variance=0.01):
    ### 
    # https://github.com/jungtaekkim/On-Uncertainty-Estimation-by-Tree-based-Surrogate-Models-in-SMO/blob/main/src/tree_based_surrogates.py

    mean = np.zeros((batch_size, len(X)))
    variance = np.zeros((batch_size, len(X)))

    tree_mixture_ratio = np.random.dirichlet(np.ones(len(trees)), size=(batch_size,))

    for t_i, tree in enumerate(trees):
        leaf_indices = tree.tree_.apply(X)
        variance_tree = tree.tree_.impurity[leaf_indices]
        mean_tree = tree.tree_.value.flatten()[leaf_indices]

        variance_tree = np.where(variance_tree < min_variance, min_variance, variance_tree)

        mean += np.expand_dims(mean_tree, axis=0) * np.expand_dims(tree_mixture_ratio[:, t_i], axis=1)
        variance += np.expand_dims(variance_tree + np.square(mean_tree), axis=0) * np.expand_dims(tree_mixture_ratio[:, t_i], axis=1)
    
    variance -= np.square(mean)
    variance = np.where(variance < min_variance, min_variance, variance)
    std = np.sqrt(variance)

    improvement = (mean - y_best)
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)


    indices_obs = []
    ei_obs = []
    for b in range(batch_size):
        argsort = np.argsort(ei[b])
        i = 1
        while argsort[-i] in observed_indices or argsort[-i] in indices_obs:
            i += 1
        indices_obs.append(argsort[-i])
        ei_obs.append(ei[b, argsort[-i]])

    return indices_obs, ei_obs


def rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

class SurrogateRandomForest:
    def __init__(self, exp_path):
        self.model = None

        self.save_path = os.path.join(exp_path, "surrogate")
        os.makedirs(self.save_path, exist_ok=True)

    def _make_new_surrogate_model(self, params):
        return RandomForestRegressor(**params, random_state=0, n_jobs=1)
    
    def save_surrogate_model(self, iteration):
        file = os.path.join(self.save_path, f"{iteration}.joblib")
        joblib.dump(self.model, file)

    def load_surrogate_model(self, iteration):
        file = os.path.join(self.save_path, f"{iteration}.joblib")
        self.model = joblib.load(file)

    def _grid_search(self, X, Y):
        kf = KFold(n_splits=5, shuffle=True, random_state=0)

        params = {
            "max_depth": range(2, 11, 1),
            "max_features": ["sqrt", "log2"],
            "min_samples_split": range(2, 11, 1),
            "n_estimators": 2**np.arange(5, 10),
        }
        gridsearch = GridSearchCV(
            RandomForestRegressor(random_state=0),
            params,
            cv=kf,
            scoring=make_scorer(rmse, greater_is_better=False),
            n_jobs=-1,
        )
        gridsearch.fit(X, Y)

        return gridsearch.best_params_, -gridsearch.best_score_

    def get_model(self):
        return self.model.estimators_


    def update(self, X, Y, iteration):
        params, score = self._grid_search(X, Y)

        self.model = self._make_new_surrogate_model(params)

        self.model.fit(X, Y)
        return params, score

        # self.save_surrogate_model(self.model, iteration)




def run_bo(robots, features, surrogate, batch_size, parallel, max_evaluation, save_path, resume=False):

    history_file = os.path.join(save_path, "history.csv")
    surrogate_path = os.path.join(save_path, "surrogate")
    

    robot_keys = list(robots.keys())

    if resume:
        history = pd.read_csv(history_file, index_col=0)
        history = history.astype(dtype={"iteration": "int64", "indice": "int64", "hash": "object", "value": "float64", "surrogate_value": "float64"})
        iter = int(history["iteration"].values[-1]) + 1
        best_value = history["value"].max()
        best_robot_hash = history["hash"][np.argmax(history["value"])]
        observed_indices = history["indice"].values.tolist()
        X_observed = [features[observed_indices]]
        Y_observed = history["value"].values.tolist()
        surrogate.load_surrogate_model(iter - 1)
    else:
        os.makedirs(surrogate_path, exist_ok=True)
        iter = 0
        best_value = -np.inf
        best_robot_hash = None
        history = pd.DataFrame(columns=["iteration", "indice", "hash", "value", "surrogate_value"])
        history = history.astype(dtype={"iteration": "int64", "indice": "int64", "hash": "object", "value": "float64", "surrogate_value": "float64"})
        X_observed = []
        Y_observed = []
        observed_indices = []

    while len(observed_indices) < max_evaluation:
        iter_start_time = time.time()
        print("*" * 20 + f"  Iteration {iter: =4}  " + "*" * 20)
        print()


        acqusition_start_time = time.time()
        if len(observed_indices) == 0:
            print("-----  initial sampling -----")

            indices_obs = np.random.choice(len(robots), batch_size, replace=False)
            observed_indices = list(indices_obs)
            expected_values = [0.0] * batch_size
            # indices_pool = indices_pool - set(observed_indices)

        else:
            print("-----  acquisition  -----")

            model = surrogate.get_model()
            # ei = get_ei(model, features, best_value)
            # print(f"expected improvement  max: {np.max(ei): =+.5f}  mean: {np.mean(ei): =+.5f}  std: {np.std(ei): =.5f}")
            # indices_obs, expected_values = parallel.acquisition(model, features, ei, observed_indices)

            # indices_obs, expected_values, ucb = acquision(model, features, observed_indices, batch_size)
            # print(f"upper confidential bound  max: {np.max(ucb): =+.4f}  mean: {np.mean(ucb): =+.4f}  std: {np.std(ucb): =.4f}")

            # y_base = np.percentile(Y_observed, 0.9)
            indices_obs, expected_values = ei_thompson(model, features, best_value, observed_indices, batch_size)

            observed_indices.extend(indices_obs)

        print("sampled indices     : [" + ", ".join([f"{indice: =10,}" for indice in indices_obs]) + "]")
        print("expected improvement: [" + ", ".join([f"{value: =10.8f}" for value in expected_values]) + "]")
        print(f"acquisition elapsed time: {time.time() - acqusition_start_time: =7.2f} sec")
        print()


        evaluate_start_time = time.time()
        print("-----  evaluate robots  -----")
        keys_obs = [robot_keys[indice] for indice in indices_obs]
        robots_obs = {key: robots[key].robot for key in keys_obs}

        # evaluate robots
        Y_obs = parallel.evaluate(robots_obs)
        Y_obs = [Y_obs[key] for key in keys_obs]

        for i, hash in enumerate(keys_obs):
            value = Y_obs[i]
            if value > best_value:
                best_robot_hash = hash
                best_value = value
            history.loc[len(history)] = [int(iter), int(indices_obs[i]), hash, value, expected_values[i]]

            print(f"indice: {indices_obs[i]: =10,}  robot: [" + hash.rjust(42) + f"]  value: {value: =+6.2f}")
        print(f"evaluation elapsed time: {time.time() - evaluate_start_time: =7.2f} sec")
        print()


        X_obs = features[indices_obs]
        X_observed.append(X_obs)
        Y_observed.append(Y_obs)

        model_update_start_time = time.time()
        print("surrogate model updating", end="")
        params, score = surrogate.update(np.vstack(X_observed), np.hstack(Y_observed), iter)
        surrogate.save_surrogate_model(iter)
        print("\rsurrogate model updated ")
        print(f"RMSE: {score: =.4f}")
        print(", ".join([f"{name}: {p}" for name,p in params.items()]))
        print(f"surrogate model update elapsed time: {time.time() - model_update_start_time: =7.2f} sec")


        print()
        history = history.astype(dtype={"iteration": "int64", "indice": "int64", "hash": "object", "value": "float64", "surrogate_value": "float64"})
        history.to_csv(history_file)
        print(f"evaluated {len(observed_indices): =4} robots")
        print("best robot: [" + best_robot_hash.rjust(42) + f"]  value: {best_value: =+6.2f}")
        print(f"elapsed time: {time.time() - iter_start_time: =7.2f} sec")
        print("\n")

        iter += 1



def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_modular_bo', f'{args.name}')

    if not args.resume:
        initialize_experiment(args.name, save_path, args)


    # robots_file = "robots.pickle"
    # features_file = "features.npy"
    # robots_file = "minimum_robots.pickle"
    # features_file = "minimum_features.npy"
    # robots_file = "fixed_robots2.pickle"
    # features_file = "fixed_features2.npy"
    # robots_file = "fixed_robots3.pickle"
    # features_file = "fixed_features3.npy"
    robots_file = "robot_samples/5-5_10000000/robots_fixed.pickle"
    features_file = "robot_samples/5-5_10000000/features_fixed.npy"


    print("dataset loading ...", end="")
    with open(robots_file, "rb") as f:
        robots = pickle.load(f)
    features = np.load(features_file).astype("float32")
    features[np.isnan(features)] = 0.0
    print("\rdataset loaded      ")

    ppo_config = ppoConfig()

    evaluator = EvogymStructureEvaluator(args.task, save_path, args.ppo_iters, args.evaluation_interval, ppo_config, resume=args.resume)
    evaluate_function = evaluator.evaluate_structure

    parallel = BatchBayesianOptimizationParallel(
        num_workers=args.num_cores,
        batch_size=args.batch_size,
        evaluate_function=evaluate_function,
        acquisition_function=acquisition_thompson,
    )

    surrogate = SurrogateRandomForest(save_path)

    run_bo(robots, features, surrogate, args.batch_size, parallel, args.max_evaluation, save_path, resume=args.resume)

if __name__=='__main__':
    main()

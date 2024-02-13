import sys
import os

import time
import pickle
import numpy as np
import pandas as pd
import itertools

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)

from arguments.evogym_ga import get_args

from parallel import EvaluatorParallel
from experiment_utils import initialize_experiment

from evaluator import ppoConfig, EvogymStructureEvaluator, EvogymStructureConstraint

import evogym.envs
from evogym import is_connected, has_actuator, get_full_connectivity




def clean_robot(robot):
    reduce_h = np.nonzero(np.any(robot > 0, axis=1))[0]
    reduce_w = np.nonzero(np.any(robot > 0, axis=0))[0]
    robot = robot[reduce_h[0]: reduce_h[-1] + 1, reduce_w[0]: reduce_w[-1] + 1]
    return robot

def get_random_robot(size):
    # pd = np.array([0.6, 0.2, 0.2, 0.2])
    pd = np.array([9, 4, 4, 4])
    pd = pd / pd.sum()
    invalid = True
    while invalid:
        robot = np.random.choice([0, 1, 3, 4], size, p=pd)
        invalid = not (has_actuator(robot) and is_connected(robot))

    robot = clean_robot(robot)
    return robot

def mutate_robot(robot, size_limit=(5, 5)):
    robot_ = np.pad(robot, ((1,), (1,)))

    pd = np.array([9, 4, 4, 4])
    pd = pd / pd.sum()

    invalid = True
    while invalid:
        mutate_pos = np.random.binomial(1, 0.1, robot_.shape).astype(bool)
        mutate_num = np.sum(mutate_pos)
        robot_[mutate_pos] = np.random.choice([0, 1, 3, 4], mutate_num, p=pd)

        robot_ = clean_robot(robot_)

        invalid = not (robot_.shape[0] <= size_limit[0] and robot_.shape[1] <= size_limit[1] and is_connected(robot) and has_actuator(robot))

    return robot_



def decode(robot_genome, args):
    body = robot_genome.robot
    connections = get_full_connectivity(body)
    return body, connections

def to_hash(robot):
    hash = ",".join(["".join(map(str, c)) for c in robot])
    return hash

class RobotGenome:
    def __init__(self, key, parent=-1, robot=None):
        self.key = key
        self.parent = parent
        self.robot = robot
        self.fitness = None

    def initialize(self):
        self.robot = get_random_robot((5, 5))

    def mutate(self, key):
        new_robot = mutate_robot(self.robot)
        child = RobotGenome(key, parent=self.key, robot=new_robot)
        return child


def run_ga(generation, pop_size, child_size, fitness_function, max_evaluation, save_path, constraint, resume=False):

    history_file = os.path.join(save_path, "history.csv")
    population_file = os.path.join(save_path, "population.pickle")
    constraint_file = os.path.join(save_path, "constraint.pickle")

    if resume:
        history = pd.read_csv(history_file, index_col=0)
        history = history.astype(dtype={"generation": "int64", "key": "int64", "hash": "object", "reward": "float64", "parent": "int64"})
        g = int(history["generation"].values[-1]) + 1
        best_fitness = history["reward"].max()
        best_robot_key = history["key"][np.argmax(history["reward"])]
        best_robot_hash = history["hash"][np.argmax(history["reward"])]

        indexer = itertools.count(history["key"].max() + 1)
        with open(population, "rb") as f:
            population = pickle.load(f)
        constraint.load_hashes(constraint_file)

    else:
        g = 0
        evaluation = 0
        best_fitness = -np.inf
        best_robot_key = None
        best_robot_hash = None
        history = pd.DataFrame(columns=["generation", "key", "hash", "reward", "parent"])
        history = history.astype(dtype={"generation": "int64", "key": "int64", "hash": "object", "reward": "float64", "parent": "int64"})

        indexer = itertools.count()
        population = {}

    
    while g < generation and evaluation < max_evaluation:
        iter_start_time = time.time()
        print("*" * 20 + f"  Generation {g: =4}  " + "*" * 20)
        print()
        current_keys = list(population.keys())

        # reproduce
        children = {}
        print()
        if g == 0 and len(population) == 0:
            for i in range(pop_size):
                key = next(indexer)
                valid = False
                while not valid:
                    robot = RobotGenome(key)
                    robot.initialize()
                    valid = constraint.eval_constraint(robot, None)
                children[key] = robot
        else:
            for _ in range(child_size):
                key = next(indexer)
                parent_key = np.random.choice(current_keys)
                valid = False
                while not valid:
                    child = population[parent_key].mutate(key)
                    valid = constraint.eval_constraint(child, None)
                children[key] = child


        # evaluation
        evaluate_start_time = time.time()
        print("-----  evaluate robots  -----")

        fitness_function(children, None, g)
        evaluation += len(children)

        print(f"evaluation elapsed time: {time.time() - evaluate_start_time: =7.2f} sec")
        print()

        # update best
        for key, robot in children.items():
            fitness = robot.fitness
            hash = to_hash(robot.robot)
            if fitness > best_fitness:
                best_robot_key = key
                best_robot_hash = hash
                best_fitness = fitness
            history.loc[len(history)] = [g, key, hash, fitness, robot.parent]

            print(f"key: {key: =4}  robot: [" + hash.rjust(42) + f"]  fitness: {fitness: =+6.2f}")


        print()
        # update population
        evaluation += len(children)
        new_keys = list(children.keys())
        population.update(children)
        population = dict(sorted(population.items(), key=lambda z: z[1].fitness)[-pop_size:])
        next_keys = list(population.keys())
        drop_keys = sorted(list(set(current_keys) - set(next_keys)))
        survived_keys = sorted(list(set(next_keys) & set(new_keys)))
        print("replacement")
        print("news : [" + ", ".join([f"{k: =4}" for k in survived_keys]))
        print("drops: [" + ", ".join([f"{k: =4}" for k in drop_keys]))


        print()
        history = history.astype(dtype={"generation": "int64", "key": "int64", "hash": "object", "reward": "float64", "parent": "int64"})
        history.to_csv(history_file)
        with open(population_file, "wb") as f:
            pickle.dump(population, f)
        constraint.save_hashes(constraint_file)

        print(f"evaluated {evaluation: =4} robots")
        print(f"best  key: {best_robot_key: 4}  robot: [" + best_robot_hash.rjust(42) + f"]  value: {best_fitness: =+6.2f}")
        print(f"elapsed time: {time.time() - iter_start_time: =7.2f} sec")
        print("\n")

        g += 1



    
def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_ga', f'{args.name}')

    initialize_experiment(args.name, save_path, args)

    constraint = EvogymStructureConstraint(decode)


    ppo_config = ppoConfig()
    evaluator = EvogymStructureEvaluator(args.task, save_path, args.ppo_iters, args.evaluation_interval, ppo_config)
    evaluate_function = evaluator.evaluate_structure

    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode,
        # parallel=False
    )

    run_ga(args.generation, args.pop_size, args.child_size, parallel.evaluate, args.max_evaluation, save_path, constraint, resume=args.resume)

if __name__=='__main__':
    main()
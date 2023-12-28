import sys
import os

import json
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)
import neat_cppn

from arguments.evogym_cppn import get_args

from parallel import EvaluatorParallel
from experiment_utils import initialize_experiment

from evaluator import ppoConfig, EvogymStructureEvaluator, EvogymStructureConstraint

import evogym.envs
from evogym import is_connected, has_actuator, hashable, get_full_connectivity

class EvogymStructureDecoder:
    def __init__(self, size):
        self.size = size

        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
        x = x.flatten()
        y = y.flatten()

        center = (np.array(size) - 1) / 2
        d = np.sqrt(np.square(x - center[0]) + np.square(y - center[1]))

        self.inputs = np.vstack([x, y, d]).T

    def decode(self, genome, config):
        cppn = neat_cppn.FeedForwardNetwork.create(genome, config)

        states = []
        for inp in self.inputs:
            state = cppn.activate(inp)
            states.append(state)

        output = np.vstack(states)
        material = np.argmax(output, axis=1)

        body = np.reshape(material, self.size)
        body[body>1] = body[body>1] + 1
        connections = get_full_connectivity(body)
        return (body, connections)


def main():
    args = get_args()

    save_path = os.path.join('out', 'evogym_cppn', f'{args.name}')

    initialize_experiment(args.name, save_path, args)


    decoder = EvogymStructureDecoder(args.shape)
    decode_function = decoder.decode

    constraint = EvogymStructureConstraint(decode_function)
    constraint_function = constraint.eval_constraint

    ppo_config = ppoConfig()

    evaluator = EvogymStructureEvaluator(args.task, save_path, args.ppo_iters, args.evaluation_interval, ppo_config)
    evaluate_function = evaluator.evaluate_structure

    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )


    config_file = os.path.join(CURR_DIR, 'config', 'robot_cppn.cfg')
    custom_config = [
        ('NEAT', 'pop_size', args.pop_size),
    ]
    config = neat_cppn.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'evogym_cppn.cfg')
    config.save(config_out_file)


    pop = neat_cppn.Population(config, constraint_function=constraint_function)

    reporters = [
        neat_cppn.SaveResultReporter(save_path),
        neat_cppn.StdOutReporter(True),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    pop.run(
        fitness_function=parallel.evaluate,
        constraint_function=constraint_function,
        n=args.generation,
        max_evaluation=args.max_evaluation
    )

if __name__=='__main__':
    main()

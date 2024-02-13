
import multiprocessing.pool
import multiprocessing as mp

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

# class Pool(mp.pool.Pool):
#     Process = NoDaemonProcess

class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

class EvaluatorParallel:
    def __init__(self, num_workers, decode_function, evaluate_function, revaluate=False, timeout=None, parallel=True, print_progress=True):
        self.num_workers = num_workers
        self.decode_function = decode_function
        self.evaluate_function = evaluate_function
        self.revaluate = revaluate
        self.timeout = timeout
        self.parallel = parallel
        self.pool = NonDaemonPool(num_workers) if parallel and num_workers>0 else None
        self.print_progress = print_progress

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def evaluate(self, genomes, config, generation):

        size = len(genomes)
        if config is not None:
            config = config.genome_config

        if self.parallel:
            phenomes = {key: self.decode_function(genome, config) for key,genome in genomes.items()}

            jobs = {}
            for key,phenome in phenomes.items():
                # if already assinged fitness, skip evaluation
                if not self.revaluate and getattr(genomes[key], 'fitness', None) is not None:
                    continue

                args = (key, phenome, generation)
                jobs[key] = self.pool.apply_async(self.evaluate_function, args=args)

            # assign the result back to each genome
            for i,(key,genome) in enumerate(genomes.items()):
                if key not in jobs:
                    continue

                if self.print_progress:
                    print(f'\revaluating genomes ... {i: =4}/{size: =4}', end='')

                reward = jobs[key].get(timeout=self.timeout)
                setattr(genome, "fitness", reward)

            if self.print_progress:
                print('\revaluating genomes ... done')

        else:
            for i,(key,genome) in enumerate(genomes.items()):
                phenome = self.decode_function(genome, config)
                if self.print_progress:
                    print(f'\revaluating genomes ... {i: =4}/{size: =4}', end='')

                args = (key, phenome, generation)
                reward = self.evaluate_function(*args)
                setattr(genome, "fitness", reward)

            if self.print_progress:
                print('\revaluating genomes ... done')


import random
import numpy as np

class BatchBayesianOptimizationParallel:
    def __init__(self, num_workers, batch_size, evaluate_function, acquisition_function, timeout=None, parallel=True, print_progress=True):
        self.num_workers = num_workers
        self.evaluate_function = evaluate_function
        self.acquisition_function = acquisition_function
        self.timeout = timeout
        self.parallel = parallel
        self.batch_size = batch_size
        self.pool = NonDaemonPool(num_workers) if parallel and num_workers>0 else None
        self.print_progress = print_progress

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def evaluate(self, candidates):

        size = len(candidates)
        values = {}

        if self.parallel:

            jobs = {}
            for key,candidate in candidates.items():

                args = (key, candidate)
                jobs[key] = self.pool.apply_async(self.evaluate_function, args=args)

            for i, (key, candidate) in enumerate(candidates.items()):
                if key not in jobs:
                    continue
                    
                if self.print_progress:
                    print(f'\revaluating robots ... {i: =4} / {size: =4}', end='')

                value = jobs[key].get(timeout=self.timeout)
                values[key] = value
                
            if self.print_progress:
                print('\revaluating robots ... done      ')

        else:
            for i, (key, candidate) in enumerate(candidates.items()):
                if self.print_progress:
                    print(f'\revaluating robots ... {i: =4} / {size: =4}', end='')

                args = (key, candidate)
                value = self.evaluate_function(*args)
                values[key] = value

            if self.print_progress:
                print('\revaluating robots ... done      ')

        return values
    
    # @classmethod
    # def _get_split_list(cls, indices, batch_size):
    #     N = len(indices)
    #     indices_ = random.sample(indices, N)  
    #     split_indices = [indices_[i::batch_size] for i in range(batch_size)]
    #     return split_indices

    # def acquisition(self, surrogate_model, features, indices_set):

    #     split_indices = self._get_split_list(indices_set, self.batch_size)

    #     indices = []
    #     values = []
    #     jobs = {}
    #     for i in range(self.batch_size):
    #         batch_indices = split_indices[i]
    #         batch_features = features[batch_indices]

    #         args = (surrogate_model, batch_features, batch_indices)
    #         jobs[i] = self.pool.apply_async(self.acquisition_function, args=args)

    #     for i in range(self.batch_size):
    #         print(f"\racquisition points ... {i: =2} / {self.batch_size: =2}", end="")

    #         indice, value = jobs[i].get(timeout=self.timeout)

    #         indices.append(indice)
    #         values.append(value)

    #     print(f"\racquisition points ... done   ")
    #     return indices, values
    
    def acquisition(self, surrogate_model, features, ei, observed_indices):

        high_ei_indices = np.argsort(ei)[-len(features)//10:]
        high_ei_featrues = features[high_ei_indices]

        indices_obs = []
        values_obs = []
        jobs = {}
        for i in range(self.batch_size):
            args = (surrogate_model, high_ei_featrues, len(observed_indices) + i + 1)
            jobs[i] = self.pool.apply_async(self.acquisition_function, args=args)

        for i in range(self.batch_size):
            print(f"\racquisition points ... {i: =2} / {self.batch_size: =2}", end="")

            indices, values = jobs[i].get(timeout=self.timeout)
            indices = high_ei_indices[indices]
            # print(indices, values)

            k = 0
            while k < len(indices):
                indice = indices[k]
                value = values[k]
                if indice not in observed_indices and indice not in indices_obs:
                    break
                k += 1
            assert k < len(indices)

            indices_obs.append(indice)
            values_obs.append(value)

        print(f"\racquisition points ... done   ")
        return indices_obs, values_obs

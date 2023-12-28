
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

        if self.parallel:
            phenomes = {key: self.decode_function(genome, config.genome_config) for key,genome in genomes.items()}

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

                results = jobs[key].get(timeout=self.timeout)
                for attr, data in results.items():
                    setattr(genome, attr, data)

                if self.print_progress:
                    print(f'\revaluating genomes ... {i+1: =4}/{size: =4}', end='')
            if self.print_progress:
                print('evaluating genomes ... done')

        else:
            for i,(key,genome) in enumerate(genomes.items()):
                phenome = self.decode_function(genome, config.genome_config)

                args = (key, phenome, generation)
                results = self.evaluate_function(*args)
                for attr, data in results.items():
                    setattr(genome, attr, data)
                if self.print_progress:
                    print(f'\revaluating genomes ... {i+1: =4}/{size: =4}', end='')
            if self.print_progress:
                print('evaluating genomes ... done')

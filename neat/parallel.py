"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool, Lock
import sys

# Create a global lock
global_lock = Lock()

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)
        self.processes = set(self.pool._pool)

    def __del__(self):
        self.pool.terminate()
        # self.pool.close()
        self.pool.join()

    def end(self, args=None):
        print("End Process: ", args)
        self.__del__()

    def evaluate(self, genomes, config):
        if any(map(lambda p: not p.is_alive(), self.processes)):
            self.pool._inqueue._rlock.release()
            raise RuntimeError("Some worker process has exited!")
        try:
            # Genomes is a list of all the genomes/neural networks. We want to divide this work onto the num_workers
            # Our eval_function accepts a list of genomes, for which it calculates and displays all the cars in parallel

            # Create a list of lists. The inner list contains all the genomes. The outer list contains all the tasks we want to parallelize.
            # Right now we naively split by num_workers
            cars_per_process = len(genomes)//self.num_workers
            genome_tasks = [genomes[i:i+cars_per_process] for i in range(0, len(genomes), cars_per_process)]

            jobs = []
            
            print("Multiprocessing Genomes: ", genomes)
            for genome_list in genome_tasks:
                print("Genome list: ", genome_list)
                # jobs.append(self.pool.apply_async(self.eval_function, (genome_list, config), error_callback=self.end))
                jobs.append(self.pool.apply_async(self.eval_function, (genome_list, config)))


            # Use this to terminate all process pools on close https://stackoverflow.com/a/69322020

            # 2D List of fitnesses. The outer list contains all the tasks that were run. The inner list has all the fitness values of the genomes that were evaluated
            fitnesses = []
            for i, job in enumerate(jobs):
                fitnesses = job.get(timeout=self.timeout)

                # Go through all the fitnesses, and assign them to their genomes.
                # This cannot be done in the eval_function since spawning a new process creates a new memory reference for the genome. When you update that new genome, the changes will not be reflected in NEAT population.run()
                for j, g in enumerate(genome_tasks[i]):
                    if fitnesses[j] == None:
                        self.pool._inqueue._rlock.release()
                        raise RuntimeError("Some worker process has exited!")
                        return False
                    g[1].fitness = fitnesses[j]
            return True
        except KeyboardInterrupt:
            with global_lock:
                self.pool.terminate()
                self.pool.join()
        

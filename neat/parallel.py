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
        eval_function takes 2 arguments, a list of genomes and the config object,
        and returns a list of floats (the fitness of each genome).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)
        self.processes = set(self.pool._pool)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    """
    This function returns True if it ran successfully, and None if not successful. This function will return None if the program was cancelled by the user.
    """
    def evaluate(self, genomes, config):
        try:
            # Genomes is a list of all the genomes/neural networks. We want to divide this work onto the num_workers
            # Our eval_function accepts a list of genomes, for which it calculates and displays all the cars in parallel

            # Create a list of lists. The inner list contains all the genomes. The outer list contains all the tasks we want to parallelize.
            # Right now we naively split by num_workers
            cars_per_process = len(genomes)//self.num_workers
            genome_tasks = [genomes[i:i+cars_per_process] for i in range(0, len(genomes), cars_per_process)]

            jobs = []
            
            for genome_list in genome_tasks:
                jobs.append(self.pool.apply_async(self.eval_function, (genome_list, config)))


            # Use this to terminate all process pools on close https://stackoverflow.com/a/69322020

            for i, job in enumerate(jobs):
                fitnesses = job.get(timeout=self.timeout)

                # Go through all the fitnesses, and assign them to their genomes.
                # This cannot be done in the eval_function since spawning a new process creates a new memory reference for the genome. When you update that new genome, the changes will not be reflected in NEAT population.run()
                for j, g in enumerate(genome_tasks[i]):

                    # We return None from the eval_function if there are any exceptions or the user cancels the program.
                    if fitnesses[j] == None:
                        # may need to release the lock if causing problems
                        # self.pool._inqueue._rlock.release()
                        return False
                    g[1].fitness = fitnesses[j]
            return True
        except:
            self.pool.terminate()
            self.pool.join()
            return None
        

import os, time, sys
import concurrent.futures
from concurrent import futures

def run_process(process):
    print(process)
    os.system('python {}'.format(process))

   
if __name__ == "__main__":        
    processes = ['testing.py', 'testing2.py']
   
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(run_process, processes)  
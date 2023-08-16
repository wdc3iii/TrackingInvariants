import numpy as np

class Logger:

    def __init__(self, log_path, header:str=None):
        self.log = open(log_path, 'w')
        if header is not None:
            self.log.write(header)

    def close(self) -> None:
        self.log.close()

    def write(self, data) -> None:
        self.log.write(np.array2string(data, separator=',')[1:-1].replace(' ', '').replace("\n", '') + "\n")
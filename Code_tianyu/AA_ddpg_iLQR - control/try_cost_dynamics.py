import numpy as np
from iLQR_controller import iLQR, fd_Cost, fd_Dynamics, myCost
import numdifftools as nd

if __name__ == '__main__':
    k=range(50,0,-1)
    for i in k:
        print(i)
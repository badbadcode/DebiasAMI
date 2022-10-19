import numpy as np
from utils.config import Config


if __name__=='__main__':
    data_name = "AMI"
    delta_T = np.load("data/AMI EVALITA 2018/deltaT_sens.npy", allow_pickle=True)
    delta_Y = np.load("data/AMI EVALITA 2018/deltaY_sens.npy", allow_pickle=True)

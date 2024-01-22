# Imports

from collections import deque
from copy import deepcopy

import pandas as pd
from DBSCAN_CFL import dbscan_cfl
from Agglomerative_CFL import run_federated_learning
from DataLoad import *

mean_db_result = pd.DataFrame()
std_db_result = pd.DataFrame()
mean_ag_result = pd.DataFrame()
std_ag_result = pd.DataFrame()
agglo_frame = pd.DataFrame()
agglo_result = pd.DataFrame()
diff_frame = pd.DataFrame()
N_CLIENTS = [10, 15, 20]
# N_CLIENTS = [20]
DIRICHLET_ALPHA = 1.0
mean_history = deque(maxlen=5)
std_history = deque(maxlen=5)
torch.manual_seed(42)
np.random.seed(42)
agglo_init = True
# dataset = 'CIFAR10'
dataset = 'EMNIST'
change_detection_period = 20
for n_clients in N_CLIENTS:

    if dataset == 'CIFAR10':
        clients, server = cifar10(n_clients, DIRICHLET_ALPHA)
        COMMUNICATION_ROUNDS = 100
        if n_clients == 10:
            EPS_1 = 1.0
            EPS_2 = 2.6
        if n_clients == 15:
            EPS_1 = 0.6
            EPS_2 = 1.6
        if n_clients == 20:
            EPS_1 = 0.5
            EPS_2 = 1.6
        c_limit = 40
    else:
        clients, server = emnist(n_clients, DIRICHLET_ALPHA)
        COMMUNICATION_ROUNDS = 40
        EPS_1 = 0.4
        EPS_2 = 1.6
        c_limit = 20

    clients_ag = deepcopy(clients)
    server_ag = deepcopy(server)

    frame, final_result = dbscan_cfl(clients, server, COMMUNICATION_ROUNDS, EPS_1, EPS_2, n_clients, mean_db_result, std_db_result, c_limit)

    agglo_frame["Difference"] = 0
    if agglo_init:
        agglo_frame, agglo_result = run_federated_learning(COMMUNICATION_ROUNDS, clients_ag, server_ag, EPS_1, EPS_2, n_clients, mean_ag_result, std_ag_result, c_limit)
        merged_frame = pd.merge(frame, agglo_frame, left_index=True, right_index=True, suffixes=('_frame1', '_frame2'))

        plot_difference(merged_frame)

    print('Mean of difference from DBSCAN: ', frame["Difference"].mean())
    print('Mean of difference from Agglomerative: ', agglo_frame["Difference"].mean())

# plot_diff(N_CLIENTS, diff_frame)

# Display the final result
print("DBSCAN mean and std: ", final_result)
if not agglo_result.empty:
    print("Agglomerative mean and std: ", agglo_result)

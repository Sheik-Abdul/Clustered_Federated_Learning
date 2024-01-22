# Imports

import numpy as np
import pandas as pd
import torch

from helper import ExperimentLogger, display_train_stats


def run_federated_learning(COMMUNICATION_ROUNDS, clients, server, EPS_1, EPS_2, N_CLIENTS, mean_ag_result, std_ag_result, c_limit):
    source = 'Agglomerative CFL'
    torch.manual_seed(42)
    np.random.seed(42)

    cfl_stats = ExperimentLogger()

    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)

        participating_clients = server.select_clients(clients, frac=1.0)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=1)
            client.reset()

        similarities = server.compute_pairwise_similarities(clients)

        acc_clients = [client.evaluate() for client in clients]
        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])

            # if math.fmod(c_round, 10) == 0:
            if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc) > 2 and c_round > c_limit:

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                c1, c2 = server.agglomerative_cluster_clients(similarities)
                cluster_indices_new = [c1, c2]

                cfl_stats.log({"split": c_round})


            else:

                add_cluster = not any(any(i in idcs for idcs in cluster_indices_new) for i in idc)

                if add_cluster:
                    cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new

        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        cfl_stats.log({"acc_clients": acc_clients, "mean_norm": mean_norm, "max_norm": max_norm,
                       "rounds": c_round, "clusters": cluster_indices})

        display_train_stats(cfl_stats, EPS_1, EPS_2, COMMUNICATION_ROUNDS, source)

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([N_CLIENTS, len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["Client {}".format(i) for i in range(results.shape[0])])

    selected_columns = ["FL Model"] + ["Model {}".format(i - 1) for i in
                                       range(frame.shape[1] - len(cluster_indices), frame.shape[1])]

    # Overwriting the existing DataFrame with the selected columns
    frame = frame[selected_columns]
    mean_frame = frame[frame != 0].mean(axis=1)
    std_frame = frame[frame != 0].std(axis=1)

    # Concatenate the results for the current iteration
    mean_ag_result = pd.concat([mean_ag_result, mean_frame], axis=1)
    std_ag_result = pd.concat([std_ag_result, std_frame], axis=1)
    frame["Difference"] = -(frame["FL Model"] - frame.iloc[:, -(len(cluster_indices)):].sum(axis=1))

    # Take the mean across columns to get the overall mean and std
    final_mean = mean_ag_result.mean(axis=1)
    final_std = std_ag_result.mean(axis=1)

    # Combine mean and std into a single DataFrame
    final_result = pd.DataFrame({"mean": final_mean, "std": final_std})

    print("Agglo: ")
    print(frame)
    print('Clusters : ', cluster_indices)
    return frame, final_result

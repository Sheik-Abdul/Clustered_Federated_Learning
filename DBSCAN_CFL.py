# Imports
import math
import numpy as np
from helper import ExperimentLogger, display_train_stats
from DataLoad import *


def transform_similarity(similarity):
    return np.round(0.5 * (1 + similarity), 5)


def dbscan_cfl(clients, server, COMMUNICATION_ROUNDS, EPS_1, EPS_2, n_clients, mean_db_result, std_db_result, c_limit):
    source = 'DBSCAN CFL'
    cfl_stats = ExperimentLogger()

    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)
                client.cache_last_layer()

        participating_clients = server.select_clients(clients, frac=1.0)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=1)
            client.reset()

        similarities = server.compute_pairwise_similarities(clients)
        transformed_similarity = transform_similarity(similarities)
        acc_clients = [client.evaluate() for client in clients]
        cluster_indices_new = []

        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])

            # if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
            # changes_detected, mean_history, std_history = last_layer_change_det(clients, mean_history, std_history)
            # if c_round == change_detection_period and changes_detected:
            if math.fmod(c_round, c_limit) == 0:
                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                # c = cluster_grid(transformed_similarity)
                # c = cluster_change_det(transformed_similarity)
                # c = last_layer_change_det(clients)
                c = cluster(transformed_similarity, 0, 0)
                cluster_indices_new = c

                cfl_stats.log({"split": c_round})

            else:
                add_cluster = not any(any(i in idcs for idcs in cluster_indices_new) for i in idc)

                if add_cluster:
                    cluster_indices_new += [idc]

        for client in clients:
            client.cache_last_layer()
        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        cfl_stats.log({"acc_clients": acc_clients, "mean_norm": mean_norm, "max_norm": max_norm,
                       "rounds": c_round, "clusters": cluster_indices})

        display_train_stats(cfl_stats, EPS_1, EPS_2, COMMUNICATION_ROUNDS, source)

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([n_clients, len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = create_frame(results, cluster_indices)

    mean_db_result, std_db_result = mean_std_frame(frame, mean_db_result, std_db_result)

    frame["Difference"] = -(frame["FL Model"] - frame.iloc[:, -(len(cluster_indices)):].sum(axis=1))
    print('DBSCAN: ')
    print(frame)
    print('Clusters : ', cluster_indices)
    # diff_frame = store_diff(frame, diff_frame, n_clients)
    final_mean = mean_db_result.mean(axis=1)
    final_std = std_db_result.median(axis=1)
    final_result = pd.DataFrame({"mean": final_mean, "std": final_std})
    return frame, final_result

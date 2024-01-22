# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
import ruptures as rpt
import torch

from cifar_models import ConvNetCifar
from data_utils import split_noniid, CustomSubset
from emnist_models import ConvNetEmnist
from fl_devices import Client, Server, pairwise_angles_layer


def emnist(N_CLIENTS, DIRICHLET_ALPHA):
    data = datasets.EMNIST(root=".", split="byclass", download=True)

    mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                     'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                     'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                     'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')

    idcs = np.random.permutation(len(data))
    train_idcs, test_idcs = idcs[:10000], idcs[10000:20000]
    # trainset, testset = data.train_data, data.test_data
    # train_idcs = np.random.permutation(len(trainset))
    # test_idcs = np.random.permutation(len(testset))
    train_labels = data.train_labels.numpy()

    client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    client_data = [CustomSubset(data, idcs) for idcs in client_idcs]
    test_data = CustomSubset(data, test_idcs, transforms.Compose([transforms.ToTensor()]))

    plt.figure(figsize=(20, 3))
    plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
             bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(N_CLIENTS)])
    plt.xticks(np.arange(62), mapp)
    plt.legend()
    plt.show()

    for i, client_datum in enumerate(client_data):
        if i < len(client_data) / 2:
            client_datum.subset_transform = transforms.Compose([transforms.RandomRotation((180, 180)),
                                                                transforms.ToTensor()])
        else:
            client_datum.subset_transform = transforms.Compose([transforms.ToTensor()])

    clients = [Client(ConvNetEmnist, lambda x: torch.optim.SGD(x, lr=0.1, momentum=0.9), dat, idnum=i)
               for i, dat in enumerate(client_data)]
    server = Server(ConvNetEmnist, test_data)
    return clients, server


def cifar10(N_CLIENTS, DIRICHLET_ALPHA):
    trainset = datasets.CIFAR10(root='.', train=True,
                                download=True)
    testset = datasets.CIFAR10(root='.', train=False,
                               download=True)
    train_idcs = np.random.permutation(len(trainset))
    test_idcs = np.random.permutation(len(testset))

    labels = [label for _, label in trainset]
    train_labels = np.array(labels)

    client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    client_data = [CustomSubset(trainset, idcs) for idcs in client_idcs]
    test_data = CustomSubset(testset, test_idcs, transforms.Compose([transforms.ToTensor()]))

    for i, client_datum in enumerate(client_data):
        if i < len(client_data) / 2:
            client_datum.subset_transform = transforms.Compose([transforms.RandomRotation((180, 180)),
                                                                transforms.ToTensor()])
        else:
            client_datum.subset_transform = transforms.Compose([transforms.ToTensor()])

    clients = [Client(ConvNetCifar, lambda x: torch.optim.SGD(x, lr=0.1, momentum=0.9), dat, idnum=i)
               for i, dat in enumerate(client_data)]
    server = Server(ConvNetCifar, test_data)
    return clients, server


def cluster_grid(S):
    # Grid search for eps and min_samples
    best_eps, best_min_samples, best_score = 0.3, 2, -1
    eps = np.linspace(np.min(S), np.max(S), 10)
    min_samples = 2
    scr = False
    for eps in eps:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(S)

        # Check the number of unique labels
        unique_labels = np.unique(cluster_labels)

        # Ensure there are more than one unique label
        if len(unique_labels) > 1:
            # Calculate silhouette score
            score = silhouette_score(S, cluster_labels)

            # Update best parameters if the score is higher
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
                print('Scores updated')
                scr = True
                print(unique_labels)

        if scr:
            print("Best eps:", best_eps)
            print("Best min_samples:", best_min_samples)
            scr = False

    return cluster(S, best_eps, best_min_samples)


def cluster_change_det(S):
    min_samples = 2
    neighbors = NearestNeighbors(metric='precomputed', n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(S)
    distances, indices = neighbors_fit.kneighbors(S)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    print("distances: ", distances)
    print('type: ', type(distances))
    algo = rpt.Dynp(model="l2")
    algo.fit(distances)
    result = algo.predict(n_bkps=1)
    print("Detected changes: ", result)
    elbow_point_index = min(result)
    elbow_point_value = distances[elbow_point_index]

    plot_knee(distances, elbow_point_index)

    if elbow_point_index is None or elbow_point_index == 0 or elbow_point_index == 1 or elbow_point_index == len(
            distances) or elbow_point_index == len(distances) - 1:
        elbow_point_index = 2
        elbow_point_value = np.mean(distances[elbow_point_index])
        print('Handled null')

    print(f"The elbow point is at index {elbow_point_index} with value {elbow_point_value}", " len(distances)",
          len(distances))

    return cluster(S, elbow_point_value, min_samples)


# def last_layer_change_det(clients):
#     # Extract weights from the last layer history
#     last_layer_weights_history = [client.last_layer_history for client in clients]
#     flattened_last_weights = [np.array(weights).flatten() for weights in last_layer_weights_history]
#     flattened_last_weights = np.std(flattened_last_weights)
#     # Extract current weights
#     current_weights = [client.dW['fc1.weight'].detach().cpu().numpy() for client in clients]
#     flattened_current_weights = [np.array(weights).flatten() for weights in current_weights]
#     flattened_current_weights = np.std(flattened_current_weights)
#
#     # Calculate the Euclidean norm of the differences between current and cached weights
#     # differences = [np.linalg.norm(current - cached) for current, cached in zip(current_weights,
#     #                                                                            last_layer_weights_history)]
#     differences = flattened_last_weights - flattened_current_weights
#     print(differences)
#     # Use a threshold to determine if there are significant changes
#     threshold = 0.003
#     changes_detected = differences > threshold
#     return changes_detected
#     # changes_detected = [diff > threshold for diff in differences]
#     # return any(changes_detected)

def last_layer_change_det(clients, mean_history, std_history):
    last_layer_weights_history = [np.array(client.last_layer_history).flatten() for client in clients]
    # Calculate mean and standard deviation
    new_mean = np.mean(last_layer_weights_history)
    new_std = np.std(last_layer_weights_history)

    # Update mean and std history with new values
    mean_history.append(new_mean)
    std_history.append(new_std)

    if detect_significant_change(mean_history, std_history):
        # Clear other values and add the new significant value
        mean_history = [new_mean]
        std_history = [new_std]

    # Check if the standard deviation is significantly higher than the mean
    changes_detected = np.mean(mean_history) < 2 * np.median(std_history)

    return changes_detected, mean_history, std_history


def detect_significant_change(mean_history, std_history):
    # Check for significant change based on moving average and adaptive threshold
    if len(mean_history) >= 2 and len(std_history) >= 2:
        mean_change = abs(mean_history[-1] - mean_history[-2])
        std_change = abs(std_history[-1] - std_history[-2])

        # Set adaptive thresholds as a percentage of historical standard deviations
        mean_threshold = 0.3 * np.std(mean_history)
        std_threshold = 0.3 * np.std(std_history)

        return mean_change > mean_threshold or std_change > std_threshold

    return False


def cluster(S, eps, min_samples):
    hdb = HDBSCAN(min_cluster_size=2)
    cluster_labels = hdb.fit_predict(S)
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # cluster_labels = dbscan.fit_predict(S)
    #
    # plot_clusters(dbscan, S, cluster_labels)

    client_clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in client_clusters:
            client_clusters[label] = []
        client_clusters[label].append(i)

    return [np.array(cluster) for cluster in client_clusters.values()]


def plot_clusters(dbscan, S, cluster_labels):
    unique_labels = set(cluster_labels)
    core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_labels == k

        xy = S[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = S[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()


def create_frame(results, cluster_indices):
    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["Client {}".format(i) for i in range(results.shape[0])])
    selected_columns = ["FL Model"] + ["Model {}".format(i - 1) for i in
                                       range(frame.shape[1] - len(cluster_indices), frame.shape[1])]
    frame = frame[selected_columns]
    return frame


def mean_std_frame(frame, mean_db_result, std_db_result):
    mean_frame = frame[frame != 0].mean(axis=1)
    std_frame = frame[frame != 0].std(axis=1)

    # Concatenate the results for the current iteration
    mean_db_result = pd.concat([mean_db_result, mean_frame], axis=1)
    std_db_result = pd.concat([std_db_result, std_frame], axis=1)
    return mean_db_result, std_db_result


def plot_difference(merged_frame):
    fig, ax = plt.subplots(figsize=(12, 6))
    merged_frame[['Difference_frame1', 'Difference_frame2']].plot(kind='bar', ax=ax)
    ax.legend()
    # Customize the plot
    plt.title('Mean Comparison between Dataframes')
    plt.xlabel('Clients')
    plt.ylabel('Mean')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def store_diff(frame, diff_frame, n_clients):
    diff_frame[f'difference_{n_clients}'] = frame['Difference']
    return diff_frame


def plot_diff(num_clients_values, diff_frame):
    fig, ax = plt.subplots(figsize=(12, 6))
    for num_clients in num_clients_values:
        diff_frame[[f'difference_{num_clients}']].plot(kind='bar', ax=ax)
    ax.legend()
    # Customize the plot
    plt.title('Mean Comparison between Clients')
    plt.xlabel('Clients')
    plt.ylabel('Mean')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_knee(distances, elbow_point_index):
    plt.plot(distances)
    plt.vlines(x=elbow_point_index, ymin=np.min(distances), ymax=np.max(distances), colors='green', ls=':', lw=2,
               label='detected change!')
    plt.title('Knee Point Detection')
    plt.xlabel('Data Points')
    plt.ylabel('Distances')
    plt.show()

import random
from collections import deque

import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch._lazy import to_cpu
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import ruptures as rpt

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_op(model, loader, optimizer, epochs=1):
    model.train()
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()

    return running_loss / samples


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct / samples


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()


def flatten_layer(parameters):
    return torch.cat([torch.from_numpy(value.flatten()) for value in parameters])


def pairwise_angles_layer(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten_layer(source1)
            s2 = flatten_layer(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()


def get_final_emnist_layer(client):
    final_layer_weights = client.dW['fc1.weight']
    return to_cpu(final_layer_weights)


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key: value for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, batch_size=128, train_frac=0.8):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data = data
        n_train = int(len(data) * train_frac)
        n_eval = len(data) - n_train
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)

        self.id = idnum

        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.last_layer_history = deque(maxlen=5)


    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)

    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"] *= 0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats

    def reset(self):
        copy(target=self.W, source=self.W_old)

    def cache_last_layer(self):
        # Cache the current last layer
        final_layer_weights = get_final_emnist_layer(self)
        self.last_layer_history.append(final_layer_weights)


class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data):
        super().__init__(model_fn, data)
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False)
        self.model_cache = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients) * frac))

    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])

    def agglomerative_cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()

        return c1, c2

    def dbscan_cluster_clients(self, S, eps, min_samp):
        min_samples = 2
        neighbors = NearestNeighbors(metric='precomputed', n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(S)
        distances, indices = neighbors_fit.kneighbors(S)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        print("distances: ", distances)
        algo = rpt.Dynp(model="l2")
        algo.fit(distances)
        result = algo.predict(n_bkps=1)
        print("Detected changes: ", result)
        elbow_point_index = min(result)
        elbow_point_value = distances[elbow_point_index]

        plt.plot(distances)
        plt.vlines(x=elbow_point_index, ymin=min(distances), ymax=max(distances), colors='green', ls=':', lw=2,
                   label='detected change!')
        plt.title('Knee Point Detection')
        plt.xlabel('Data Points')
        plt.ylabel('Distances')
        plt.show()

        if elbow_point_index is None or elbow_point_index == 0 or elbow_point_index == 1 or elbow_point_index == 9 or elbow_point_index == 8:
            elbow_point_index = 2
            elbow_point_value = np.mean(distances[elbow_point_index])
            print('Handled null')

        print(f"The elbow point is at index {elbow_point_index} with value {elbow_point_value}", " len(distances)",
              len(distances))

        dbscan = DBSCAN(eps=eps, min_samples=min_samp, metric='precomputed')
        cluster_labels = dbscan.fit_predict(S)

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

        client_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in client_clusters:
                client_clusters[label] = []
            client_clusters[label].append(i)

        return [np.array(cluster) for cluster in client_clusters.values()]

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            reduce_add_average(targets=[client.W for client in cluster],
                               sources=[client.dW for client in cluster])

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]),
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

    def get_final_emnist_layer(self):
        final_layer = self.model.fc1
        final_layer_weights = final_layer.weight
        return self.to_cpu(final_layer_weights)

    def get_final_cifar_layer(self):
        return self.model.fc3

import os
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from scripts.utils import accuracy
import torch.nn.functional as F


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm


class Trainer(object):
    def __init__(self, model, embedding_model, optimizers, loss_func, device, embedding_grad_clip, meta_dataset,
                 writer, log_interval, save_interval, model_type, save_folder, total_iter, collect_accuracies,
                 num_support_sample_per_calss, transductive):

        self._model = model
        self._embedding_model = embedding_model
        self._prototypes = {}
        self._features_sum = {}
        self._optimizers = optimizers
        self._loss_func = loss_func
        self._device = device
        self._embedding_grad_clip = embedding_grad_clip
        self._grads_mean = []
        self.to(device)
        self._reset_measurements()
        self._collect_accuracies = collect_accuracies
        self._meta_dataset = meta_dataset
        self._writer = writer
        self._log_interval = log_interval
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._total_iter = total_iter
        self._reset_measurements()
        self._num_class = meta_dataset.output_size
        self._num_query = 15
        self._num_support = num_support_sample_per_calss
        self._transductive = transductive


    def _reset_measurements(self):
        self._count_iters = 0.0
        self._cum_loss = 0.0
        self._cum_accuracy = 0.0

    def _update_measurements(self, task, loss, preds):
        self._count_iters += 1.0
        self._cum_loss += loss.data.cpu().numpy()
        if self._collect_accuracies:
            self._cum_accuracy += accuracy(preds, task.y).data.cpu().numpy()

    def _pop_measurements(self):
        measurements = {}
        loss = self._cum_loss / self._count_iters
        measurements['loss'] = loss
        if self._collect_accuracies:
            accuracy = self._cum_accuracy / self._count_iters
            measurements['accuracy'] = accuracy
        self._reset_measurements()
        return measurements

    def _state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizers': [optimizer.state_dict() for optimizer in self._optimizers]
        }
        if self._embedding_model:
            state.update(
                {'embedding_model_state_dict':
                 self._embedding_model.state_dict()})
        return state

    def _calculate_proto(self, features, labels):
        proto = torch.zeros(self._meta_dataset.output_size, features.size(1), device=self._device)
        for i in range(labels.size(0)):
            proto[labels[i], :] += features[i, :]

        for i in range(proto.size(0)):
            n_rep = (labels == i).nonzero().numel()
            if n_rep != 0:
                proto[i, :] = proto[i, :] / n_rep

        return proto

    def _eucldian_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x-y, 2).sum(2)

    def _calculate_loss(self, features, prototypes, labels):

        target_inds = labels.view(self._num_class, self._num_query, 1).long()
        target_inds.requires_grad = False
        target_inds = target_inds.to(self._device)
        dists = self._eucldian_dist(features, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        log_p_y = log_p_y.to(self._device)
        loss_val = F.nll_loss(log_p_y, labels)
        loss_val.clone().detach()
        _, y_hat = log_p_y.max(1)  # y_hat gives the index for the maximum value of log_p
        acc_val = torch.eq(y_hat, labels).float().mean()

        return loss_val, acc_val

    def _update_proto(self, task, prototypes, params, embeddings, features_sum, num_support):
        features = self._model(task, params=params, embeddings=embeddings)
        dists = self._eucldian_dist(features, prototypes)
        p_y = F.softmax(-dists, dim=1)

        for j in range(self._num_class):
            scores = p_y[:, j]
            scores = scores.unsqueeze(1).expand(scores.size(0), features.size(1))
            weighted_score = (scores*features).sum(dim=0)
            scores_sum = p_y[:, j].sum()
            prototypes[j, :] = (features_sum[j] + weighted_score) / (self._num_support + scores_sum)
        return prototypes

    def run(self, is_training):
        if not is_training:
            acc_val = []
            loss_val = []

        if self._meta_dataset.name == 'MultimodalFewShot':
            accuracies = [[] for i in range(self._meta_dataset.num_dataset)]

        for i, (train_tasks, val_tasks) in enumerate(iter(self._meta_dataset), start=1):

            # Save model
            if (i % self._save_interval == 0 or i == 1) and is_training:
                save_name = 'model_{0}_{1}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._state_dict(), f)

            params = self._model.param_dict

            for optimizer in self._optimizers:
                optimizer.zero_grad()

            # Compute prototypes for each task
            cnt = 1
            for task in train_tasks:
                embeddings = None
                if self._embedding_model:
                    embeddings = self._embedding_model(task)
                features = self._model(task, params=params, embeddings=embeddings)
                self._prototypes[str(cnt)] = self._calculate_proto(features, task.y)
                features_sum = []
                for j in range(self._num_class):
                    features_sum.append(features[j*self._num_support:(j+1)*self._num_support-1, :].sum(dim=0))
                self._features_sum[str(cnt)] = features_sum

                cnt += 1

            # Evaluate each task on query samples
            cnt = 1
            losses = []
            for task in val_tasks:
                embeddings = None
                if self._embedding_model:
                    embeddings = self._embedding_model(task)

                if self._transductive:
                    self._prototypes[str(cnt)] = self._update_proto(task, self._prototypes[str(cnt)],
                                                                    params, embeddings, self._features_sum[str(cnt)],
                                                                    self._num_support)

                features = self._model(task, params=params, embeddings=embeddings)
                loss, acc = self._calculate_loss(features, self._prototypes[str(cnt)], task.y)

                if self._meta_dataset.name == 'MultimodalFewShot':
                    accuracies[self._meta_dataset.dataset_names.index(task.task_info)].append(acc.item())

                cnt += 1
                if is_training:
                    self._cum_accuracy += acc.item()
                    self._cum_loss += loss.item()
                    self._count_iters += 1
                    losses.append(loss)
                else:
                    loss_val.append(loss.item())
                    acc_val.append(acc.item())

            if is_training:
                # update model parameters
                mean_loss = torch.mean(torch.stack(losses))
                mean_loss = mean_loss.to(self._device)
                measurements = self._pop_measurements()
                mean_loss.backward()

                if i % 10000 == 0:
                    self._optimizers[0].param_groups[0]['lr'] *= 0.5
                self._optimizers[0].step()
                if len(self._optimizers) > 1:
                    if self._embedding_grad_clip > 0:
                        _grad_norm = clip_grad_norm_(self._embedding_model.parameters(), self._embedding_grad_clip)
                    else:
                        _grad_norm = get_grad_norm(self._embedding_model.parameters())
                    # grad_norm
                    self._grads_mean.append(_grad_norm)
                    if i % 10000 == 0:
                        self._optimizers[1].param_groups[0]['lr'] *= 0.5
                    self._optimizers[1].step()

                if (i == 1) or (i % self._log_interval == 0):
                    print('-------------------')
                    print('Meta-Batch Number: {}'.format(i))
                    if self._meta_dataset.name == 'MultimodalFewShot':
                        accuracy_str = []
                        for cnt, acc in enumerate(accuracies):
                            accuracy_str.append('{}: {}'.format(
                                self._meta_dataset.dataset_names[cnt],
                                'NaN' if len(acc) == 0 else '{:.3f}%'.format((100*np.mean(acc)))
                            ))

                        print('Individual accuracies: {}'.format(' '.join(accuracy_str)))

                    print('Average loss: {:.3f}, Average accuracy: {:.3f}%'
                          .format(measurements['loss'], 100*measurements['accuracy']))
            else:
                if i % self._log_interval == 0:  # report results for (log_interval*meta_batch_size) tasks
                    print('Evaluation Results ===>>> Batch Number: {}'.format(i))

                    if self._meta_dataset.name == 'MultimodalFewShot':
                        accuracy_str = []
                        for cnt, acc in enumerate(accuracies):
                            accuracy_str.append('{}: {:.3f}+_{:.3f}'.format(
                                self._meta_dataset.dataset_names[cnt],
                                100 * np.mean(acc),
                                100 * self.compute_confidence_interval(acc)))

                        print('Individual accuracies: {}'.format(' '.join(accuracy_str)))

                        accuracies = [[] for i in range(self._meta_dataset.num_dataset)]

                    print('Loss: {:.3f}+_{:.3f} || Accuracy: {:.3f}+_{:.3f}%'.format(
                                np.mean(loss_val), self.compute_confidence_interval(loss_val),
                                100*np.mean(acc_val), 100*self.compute_confidence_interval(acc_val)
                    ))
                    loss_val = []
                    acc_val = []
                    print('---------------------------------')

    def compute_confidence_interval(self, value):
        """
        Compute 95% +- confidence intervals over tasks
        change 1.960 to 2.576 for 99% +- confidence intervals
        """
        return np.std(value) * 1.960 / np.sqrt(len(value))

    def train(self):
        self.run(is_training=True)

    def eval(self):
        self.run(is_training=False)

    def write_tensorboard(self, pre_val_measurements, pre_train_measurements,
                          post_val_measurements, post_train_measurements,
                          iteration, embedding_grads_mean=None):
        for key, value in pre_val_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_val'.format(key), value, iteration)
        for key, value in pre_train_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_train'.format(key), value, iteration)
        for key, value in post_train_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_train'.format(key), value, iteration)
        for key, value in post_val_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_val'.format(key), value, iteration)
        if embedding_grads_mean is not None:
            self._writer.add_scalar(
                'embedding_grads_mean', embedding_grads_mean, iteration)

    def log_output(self, pre_val_measurements, pre_train_measurements,
                   post_val_measurements, post_train_measurements,
                   iteration, embedding_grads_mean=None):
        log_str = 'Iteration: {}/{} '.format(iteration, self._total_iter)
        for key, value in sorted(pre_val_measurements.items()):
            log_str = (log_str + '{} meta_val before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(pre_train_measurements.items()):
            log_str = (log_str + '{} meta_train before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_train_measurements.items()):
            log_str = (log_str + '{} meta_train after: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_val_measurements.items()):
            log_str = (log_str + '{} meta_val after: {:.3f} '
                                 ''.format(key, value))
        if embedding_grads_mean is not None:
            log_str = (log_str + 'embedding_grad_norm after: {:.3f} '
                                 ''.format(embedding_grads_mean))
        print(log_str)

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        if self._embedding_model:
            self._embedding_model.to(device, **kwargs)

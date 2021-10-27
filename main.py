import os
import json
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from scripts.datasets.omniglot import OmniglotMetaDataset
from scripts.datasets.miniimagenet import MiniimagenetMetaDataset
from scripts.datasets.cifar100 import Cifar100MetaDataset
from scripts.datasets.bird import BirdMetaDataset
from scripts.datasets.aircraft import AircraftMetaDataset
from scripts.datasets.multimodal_few_shot import MultimodalFewShotDataset
from scripts.models.conv_net import ConvModel
from scripts.models.gated_conv_net import GatedConvModel
from scripts.models.conv_embedding_model import ConvEmbeddingModel
from scripts.trainer import Trainer
from scripts.utils import optimizer_to_device, get_git_revision_hash


def main(args):
    is_training = not args.eval
    run_name = 'train' if is_training else 'eval'

    if is_training:
        writer = SummaryWriter('./train_dir/{0}/{1}'.format(
            args.output_folder, run_name))
        with open('./train_dir/{}/config.txt'.format(
            args.output_folder), 'w') as config_txt:
            for k, v in sorted(vars(args).items()):
                config_txt.write('{}: {}\n'.format(k, v))
    else:
        writer = None

    save_folder = './train_dir/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    config_name = '{0}_config.json'.format(run_name)
    with open(os.path.join(save_folder, config_name), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        try:
            config.update({'git_hash': get_git_revision_hash()})
        except:
            pass
        json.dump(config, f, indent=2)

    # Define Dataset
    _num_tasks = 1
    if args.dataset == 'omniglot':
        dataset = OmniglotMetaDataset(
            root='data',
            img_side_len=28,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'cifar':
        dataset = Cifar100MetaDataset(
            root='data',
            img_side_len=32,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'multimodal_few_shot':
        dataset_list = []
        if 'omniglot' in args.multimodal_few_shot:
            dataset_list.append(OmniglotMetaDataset(
                root='data',
                img_side_len=args.common_img_side_len,
                img_channel=args.common_img_channel,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_samples_per_class,
                num_total_batches=args.num_batches,
                num_val_samples=args.num_val_samples,
                meta_batch_size=args.meta_batch_size,
                train=is_training,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
            )
        if 'miniimagenet' in args.multimodal_few_shot:
            dataset_list.append( MiniimagenetMetaDataset(
                root='data',
                img_side_len=args.common_img_side_len,
                img_channel=args.common_img_channel,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_samples_per_class,
                num_total_batches=args.num_batches,
                num_val_samples=args.num_val_samples,
                meta_batch_size=args.meta_batch_size,
                train=is_training,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
            )
        if 'cifar' in args.multimodal_few_shot:
            dataset_list.append(Cifar100MetaDataset(
                root='data',
                img_side_len=args.common_img_side_len,
                img_channel=args.common_img_channel,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_samples_per_class,
                num_total_batches=args.num_batches,
                num_val_samples=args.num_val_samples,
                meta_batch_size=args.meta_batch_size,
                train=is_training,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
            )
        if 'bird' in args.multimodal_few_shot:
            dataset_list.append(BirdMetaDataset(
                root='data',
                img_side_len=args.common_img_side_len,
                img_channel=args.common_img_channel,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_samples_per_class,
                num_total_batches=args.num_batches,
                num_val_samples=args.num_val_samples,
                meta_batch_size=args.meta_batch_size,
                train=is_training,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
            )
        if 'aircraft' in args.multimodal_few_shot:
            dataset_list.append(AircraftMetaDataset(
                root='data',
                img_side_len=args.common_img_side_len,
                img_channel=args.common_img_channel,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_samples_per_class,
                num_total_batches=args.num_batches,
                num_val_samples=args.num_val_samples,
                meta_batch_size=args.meta_batch_size,
                train=is_training,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
            )
        assert len(dataset_list) > 0
        print('Multimodal Few Shot Datasets: {}'.format(
            ' '.join([dataset.name for dataset in dataset_list])))
        dataset = MultimodalFewShotDataset(
            dataset_list,
            num_total_batches=args.num_batches,
            mix_meta_batch=args.mix_meta_batch,
            mix_mini_batch=args.mix_mini_batch,
            txt_file=args.sample_embedding_file+'.txt' if args.num_sample_embedding > 0 else None,
            train=is_training,
        )
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'miniimagenet':
        dataset = MiniimagenetMetaDataset(
            root='data',
            img_side_len=84,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    else:
        raise ValueError('Unrecognized dataset {}'.format(args.dataset))

    # Define Model
    if args.model_type == 'conv':
        model = ConvModel(
            input_channels=dataset.input_size[0],
            output_size=dataset.output_size,
            num_channels=args.num_channels,
            img_side_len=dataset.input_size[1],
            use_max_pool=args.use_max_pool,
            verbose=args.verbose)
    elif args.model_type == 'gatedconv':
        model = GatedConvModel(
            input_channels=dataset.input_size[0],
            output_size=dataset.output_size,
            use_max_pool=args.use_max_pool,
            num_channels=args.num_channels,
            img_side_len=dataset.input_size[1],
            condition_type=args.condition_type,
            condition_order=args.condition_order,
            verbose=args.verbose)
    else:
        raise ValueError('Unrecognized model type {}'.format(args.model_type))
    model_parameters = list(model.parameters())

    # Define Embedding Type
    if args.embedding_type == '':
        embedding_model = None
    elif args.embedding_type == 'ConvGRU':
        embedding_model = ConvEmbeddingModel(
             input_size=np.prod(dataset.input_size),
             output_size=dataset.output_size,
             embedding_dims=args.embedding_dims,
             hidden_size=args.embedding_hidden_size,
             num_layers=args.embedding_num_layers,
             convolutional=args.conv_embedding,
             num_conv=args.num_conv_embedding_layer,
             num_channels=args.num_channels,
             rnn_aggregation=(not args.no_rnn_aggregation),
             embedding_pooling=args.embedding_pooling,
             batch_norm=args.conv_embedding_batch_norm,
             avgpool_after_conv=args.conv_embedding_avgpool_after_conv,
             linear_before_rnn=args.linear_before_rnn,
             num_sample_embedding=args.num_sample_embedding,
             sample_embedding_file=args.sample_embedding_file+'.'+args.sample_embedding_file_type,
             img_size=dataset.input_size,
             verbose=args.verbose)
        embedding_parameters = list(embedding_model.parameters())
    else:
        raise ValueError('Unrecognized embedding type {}'.format(
            args.embedding_type))

    # Define Optimizer
    if embedding_model:
      optimizers = ( torch.optim.Adam(model_parameters, lr=args.lr),
                     torch.optim.Adam(embedding_parameters, lr=args.lr) )
    else:
      optimizers = ( torch.optim.Adam(model_parameters, lr=args.lr), )

    # Load a saved model, if there was a checkpoint command in terminal for meta-test
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        if 'optimizer' in checkpoint:
          pass
        else:
          optimizers[0].load_state_dict(checkpoint['optimizers'][0])
          optimizer_to_device(optimizers[0], args.device)

          if embedding_model:
            embedding_model.load_state_dict(
                checkpoint['embedding_model_state_dict'])
            optimizers[1].load_state_dict(checkpoint['optimizers'][1])
            optimizer_to_device(optimizers[1], args.device)

    # To resume training from a saved checkpoint (also, need to change index in run function of trainer)
    if args.resume:
        latest_chk = 'TYPE DIRECTORY OF SAVED MDOEL TO BE LOADED FOR RESUMING TRAINING'
        checkpoint = torch.load(latest_chk)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        if 'optimizer' in checkpoint:
            pass
        else:
            optimizers[0].load_state_dict(checkpoint['optimizers'][0])
            optimizer_to_device(optimizers[0], args.device)

            if embedding_model:
                embedding_model.load_state_dict(
                    checkpoint['embedding_model_state_dict'])
                optimizers[1].load_state_dict(checkpoint['optimizers'][1])
                optimizer_to_device(optimizers[1], args.device)

    # Define Trainer
    trainer = Trainer(
        model, embedding_model, optimizers, loss_func=loss_func, device=args.device,
        embedding_grad_clip=args.embedding_grad_clip, meta_dataset=dataset, writer=writer,
        log_interval=args.log_interval, save_interval=args.save_interval,
        model_type=args.model_type, save_folder=save_folder,
        total_iter=args.num_batches//args.meta_batch_size, collect_accuracies=collect_accuracies,
        num_support_sample_per_calss=args.num_samples_per_class, transductive=args.transductive
    )

    if is_training:
        trainer.train()
    else:
        trainer.eval()


if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(description='KML for Multimodal Meta-Learning')

    parser.add_argument('--kml-model', type=str2bool, default=True,
                        help='use KML for task-level modulating the meta-learner and producing task-aware layers')
    parser.add_argument('--vanilla-model', type=str2bool, default=False,
                        help='use vanilla meta-learner')

    # Model
    parser.add_argument('--hidden-sizes', type=int,
        default=[256, 128, 64, 64], nargs='+',
        help='number of hidden units per layer')
    parser.add_argument('--model-type', type=str, default='gatedconv',
                        help='type of the model')
    parser.add_argument('--condition-type', type=str, default='affine',
        choices=['affine', 'sigmoid', 'softmax'],
        help='type of the conditional layers')
    parser.add_argument('--condition-order', type=str, default='low2high',
        help='order of the conditional layers to be used')
    parser.add_argument('--use-max-pool', type=str2bool, default=False,
                        help='choose whether to use max pooling with convolutional model')
    parser.add_argument('--num-channels', type=int, default=32,
                        help='number of channels in convolutional layers')
    parser.add_argument('--disable-norm', action='store_true',
        help='disable batchnorm after linear layers in a fully connected model')
    parser.add_argument('--bias-transformation-size', type=int, default=0,
        help='size of bias transformation vector that is concatenated with '
             'input')

    # Embedding
    parser.add_argument('--embedding-type', type=str, default='',
        help='type of the embedding')
    parser.add_argument('--embedding-hidden-size', type=int, default=128,
        help='number of hidden units per layer in recurrent embedding model')
    parser.add_argument('--embedding-num-layers', type=int, default=2,
        help='number of layers in recurrent embedding model')
    parser.add_argument('--embedding-dims', type=int, nargs='+', default=0,
        help='dimensions of the embeddings')

    # Randomly sampled embedding vectors
    parser.add_argument('--num-sample-embedding', type=int, default=0,
        help='number of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file', type=str, default='embeddings',
        help='the file name of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file-type', type=str, default='hdf5')

    # Inner loop
    '''parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')'''
    '''parser.add_argument('--fast-lr', type=float, default=0.05,
        help='learning rate for the 1-step gradient update of MAML')'''
    '''parser.add_argument('--inner-loop-grad-clip', type=float, default=20.0,
        help='enable gradient clipping in the inner loop')'''
    '''parser.add_argument('--num-updates', type=int, default=5,
        help='how many update steps in the inner loop')'''

    # Optimization
    parser.add_argument('--num-batches', type=int, default=300000,
                        help='number of meta-training batches')
    parser.add_argument('--meta-batch-size', type=int, default=10,
                        help='number of tasks per batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='KML_ProtoNet_3mode_5w1s',
                        help='name of the output folder')
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='how many DataLoader workers to use')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of batches between printing info')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='number of batches between model saves')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to saved parameters when evaluating the model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='path to saved parameters for resuming the training')
    parser.add_argument('--transductive', type=str2bool, default=True,
                        help='use transductive scenario (update prototype with soft labels from query set)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='multimodal_few_shot',
                        help='which dataset to use')
    parser.add_argument('--data-root', type=str, default='data',
                        help='path to store datasets')
    parser.add_argument('--num-train-classes', type=int, default=1100,
                        help='how many classes for training')
    parser.add_argument('--num-classes-per-batch', type=int, default=5,
                        help='how many classes per task (number of ways)')
    parser.add_argument('--num-samples-per-class', type=int, default=1,
                        help='how many samples per class for training (number of shots)')
    parser.add_argument('--num-val-samples', type=int, default=15,
                        help='how many samples per class for validation')
    parser.add_argument('--img-side-len', type=int, default=28,
                        help='width and height of the input images')
    parser.add_argument('--input-range', type=float, default=[-5.0, 5.0],
        nargs='+', help='input range of simple functions')
    parser.add_argument('--phase-range', type=float, default=[0, np.pi],
        nargs='+', help='phase range of sinusoids')
    parser.add_argument('--amp-range', type=float, default=[0.1, 5.0],
        nargs='+', help='amp range of sinusoids')
    parser.add_argument('--slope-range', type=float, default=[-3.0, 3.0],
        nargs='+', help='slope range of linear functions')
    parser.add_argument('--intersect-range', type=float, default=[-3.0, 3.0],
        nargs='+', help='intersect range of linear functions')
    parser.add_argument('--noise-std', type=float, default=0.0,
        help='add gaussian noise to mixed functions')
    parser.add_argument('--oracle', action='store_true',
        help='concatenate phase and amp to sinusoid inputs')
    parser.add_argument('--task-oracle', action='store_true',
        help='uses task id for prediction in some models')

    # Combine few-shot learning datasets
    parser.add_argument('--multimodal_few_shot', type=str,
                        default=['omniglot', 'cifar', 'miniimagenet'],
                        choices=['omniglot', 'cifar', 'miniimagenet', 'bird', 'aircraft'],
                        nargs='+')
    parser.add_argument('--common-img-side-len', type=int, default=84)
    parser.add_argument('--common-img-channel', type=int, default=3,
                        help='3 for RGB and 1 for grayscale')
    parser.add_argument('--mix-meta-batch', type=str2bool, default=True)
    parser.add_argument('--mix-mini-batch', type=str2bool, default=False)

    parser.add_argument('--alternating', action='store_true')
    parser.add_argument('--classifier-schedule', type=int, default=10)
    parser.add_argument('--embedding-schedule', type=int, default=10)
    parser.add_argument('--conv-embedding', type=str2bool, default=True)
    parser.add_argument('--conv-embedding-batch-norm', type=str2bool, default=True)
    parser.add_argument('--conv-embedding-avgpool-after-conv', type=str2bool, default=True)
    parser.add_argument('--num-conv-embedding-layer', type=int, default=4)
    parser.add_argument('--no-rnn-aggregation', type=str2bool, default=True)
    parser.add_argument('--embedding-pooling', type=str,
                        choices=['avg', 'max'], default='avg', help='')
    parser.add_argument('--linear-before-rnn', action='store_true')
    parser.add_argument('--embedding-grad-clip', type=float, default=0.0)
    parser.add_argument('--verbose', type=str2bool, default=False)

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # Make sure num sample embedding < num sample tasks
    args.num_sample_embedding = min(args.num_sample_embedding, args.num_batches)
    # ToDO: Make the calculation automatic based on the kernels dims of CNN
    args.embedding_dims = [32, 27, 32, 192, 96, 64, 384, 192, 128, 768, 384, 256]

    assert not (args.kml_model and args.vanilla_model)

    if args.kml_model is True:
        print('Use KML with Meta-Learner')
        args.model_type = 'gatedconv'
        args.embedding_type = 'ConvGRU'

    if args.vanilla_model is True:
        print('Use vanilla Meta-Learner')
        args.model_type = 'conv'
        args.embedding_type = ''

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    # print args
    if args.verbose:
        print('='*10 + ' ARGS ' + '='*10)
        for k, v in sorted(vars(args).items()):
            print('{}: {}'.format(k, v))
        print('='*26)

    main(args)

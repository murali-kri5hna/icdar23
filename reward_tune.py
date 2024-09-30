# Import required libraries and modules
import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from pytorch_metric_learning import samplers
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, const_scheduler, cosine_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from reranking import sgr

from aug import Erosion, Dilation
from utils.triplet_loss import TripletLoss, RewardTripletLoss
from utils.m_n_sampler import MWritersPerNClusterSampler
#from utils.fast_ap_reward import FastAPReward
from pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
from pytorch_metric_learning.losses.fast_ap_loss import FastAPLoss
from pytorch_metric_learning.losses.ranked_list_loss import RankedListLoss
#from pytorch_metric_learning import losses
from pytorch_metric_learning import distances, miners, reducers

from utils.Smooth_AP_loss import SmoothAP
from utils.Smooth_AP_loss_Brown import SmoothAP as SmoothAP_brown
from utils.bb_map_loss import MapLoss
from utils.ranked_list_reward import RankedListReward

#import blackbox_backprop as bb

from backbone import resnets
from backbone.model import Model

import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#multiprocessing.set_start_method("spawn") # fork

from main import prepare_logging, train_val_split, test, get_optimizer, validate, compute_page_features, save_page_features

from visualization.data_table import cluster_plot, addtoTable

# Training function for one epoch
def train_one_epoch(model, train_ds, Loss, optimizer, scheduler, epoch, args, logger, reward_baseline, **kwargs):
    
    model.train()
    model = model.cuda()  # Move the model to GPU

    # If reward tuning is enabled, add dropout to the model
    if args['reward_tuning']:
        model.fc = torch.nn.Dropout(p=0.3)

    # Labels of the training dataset
    labels = train_ds.dataset.labels
    
    # Sampler setup depending on whether reward tuning is active
    if args['reward_tuning']:
        sampler = MWritersPerNClusterSampler(np.array(train_ds.dataset.labels[args['train_options']['n_label']])[train_ds.indices],
                                             np.array(train_ds.dataset.labels[args['train_options']['m_label']])[train_ds.indices],
                                             np.array(train_ds.dataset.labels['page'])[train_ds.indices],
                                             args['train_options']['sampler_m'], 
                                             args['train_options']['sampler_n'],
                                             batch_size=args['train_options']['batch_size'],
                                             length_before_new_iter=args['train_options']['length_before_new_iter'])

    else:
        sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    
    # DataLoader with the created sampler
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)
    
    pbar = tqdm(train_triplet_loader)  # Progress bar for training
    pbar.set_description('Epoch {} Training'.format(epoch))  # Set progress bar description
    iters = len(train_triplet_loader)  # Total iterations in the epoch
    logger.log_value('Epoch', epoch, commit=False)  # Log epoch number

    # Accumulation step for gradient accumulation (used for large batch sizes)
    if args['train_options']['accumulation_steps']:
        accumulation_steps = args['train_options']['accumulation_steps']

    reward_baseline = 0  # Initialize reward baseline
    
    # Loop through the training data
    for i, (samples, labels) in enumerate(pbar):
        idx = i  # Current iteration index
        it = iters * epoch + i  # Overall iteration count

        # Adjust learning rate based on the scheduler
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']

        # Separate writer and page labels
        if len(labels) == 3:
            writers, pages = labels[1], labels[2]
        else:
            writers, pages = labels[0], labels[1]

        samples = samples.cuda()
        samples.requires_grad = True  # Ensure the input requires gradient

        # Define the training label to use (e.g., writer or cluster)
        if args['train_label'] == 'cluster':
            l = labels[0]
        if args['train_label'] == 'writer':
            l = labels[1]

        # Define the reward label to use for reward-based training
        if args['reward_label'] == 'cluster':
            reward_label = labels[0]

        if args['reward_label'] == 'writer':
            reward_label = labels[1]

        l = l.cuda()  # Move the label to GPU
        reward_label = reward_label.cuda()

        # Pass the input samples through the model to get embeddings
        emb = model(samples)
        
        # Select loss function based on the arguments
        if args['train_options']['loss'] == "triplet":
            loss = Loss(emb, l, emb, l)
    
        if args['train_options']['loss'] == "fastAP":
            loss = Loss(emb, l)
    
        if args['train_options']['loss'] == "smoothAP":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "smoothAP_brown":
            loss = Loss(emb)

        if args['train_options']['loss'] == "rewardTriplet":
            loss = Loss(emb, l, emb, l, reward_label, reward_baseline, logger=logger)

        if args['train_options']['loss'] == "RankedListLoss":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "RankedListReward":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "bbMapLoss":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "triplet_pml":
            for key, value in kwargs.items():
                if key == 'mining_func':
                    mining_func = value
            loss = Loss(emb, l)  # Loss with mining function for hard triplets
        
        logger.log_value(f'loss', loss.item())  # Log the loss

        # If reward tuning is enabled
        if args['reward_tuning']:
            feats = emb.detach().cpu().numpy()  # Detach and convert embeddings to numpy
            page_features, page_writers = compute_page_features(feats, np.array(writers), np.array(pages))

            norm = 'powernorm'  # Normalization type
            pooling = SumPooling(norm)  # Apply sum pooling
            descs = pooling.encode(page_features)

            _eval = Retrieval()  # Instantiate the retrieval evaluator

            # Calculate distances and mean Average Precision (mAP) as the reward
            dists = _eval.calc_distances(descs, page_writers, use_precomputed_distances=False)
            map = _eval.calc_only_map_from_distances(page_writers, dists)
            logger.log_value(f'map_reward', map)  # Log the map reward

            # Update reward baseline after each accumulation step
            if idx % accumulation_steps == 0:
                reward_baseline = map

            # Adjust loss using gradient ascent based on map reward
            loss = loss * (-(map - reward_baseline))
            
            if idx % accumulation_steps != 0:
                logger.log_value(f'reward_tuned_loss', loss.item())
                logger.log_value(f'map_with_baseline', map - reward_baseline)
            
            logger.log_value(f'lr', optimizer.param_groups[0]['lr'])  # Log learning rate
    
            if idx % accumulation_steps == 0:  # Perform optimizer step at every accumulation
                optimizer.zero_grad()
    
            # Backpropagate the loss
            if accumulation_steps > 1:
                loss = loss / (accumulation_steps - 1)
            
            loss.backward()
    
            if (idx + 1) % accumulation_steps == 0:  # Update optimizer
                optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)  # Clip gradients to prevent exploding gradients
        
        # Regular (non-reward tuning) training
        else:
            logger.log_value(f'lr', optimizer.param_groups[0]['lr'])  # Log learning rate
    
            optimizer.zero_grad()
            loss.backward()  # Backpropagation
    
            optimizer.step()  # Update optimizer
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)  # Gradient clipping
    
    torch.cuda.empty_cache()  # Empty cache to free memory
    
    if args['reward_tuning']:
        model.fc = torch.nn.Identity()
    return model
        


def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])

    if args['optimizer_options']['scheduler'] == 'const':
        lr_schedule = const_scheduler(args['optimizer_options']['base_lr'], epochs, niter_per_ep, args['optimizer_options']['warmup_epochs'])
    if args['optimizer_options']['scheduler'] == 'cosine':
        lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)
    

    best_epoch = -1
    best_map = validate(model, val_ds, args)
    reward_baseline = best_map
    #best_map = -1
    

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)
    best_map = -1

    #reward_loss = TripletLoss(margin=args['train_options']['margin'])
    
    #loss = FastAPLoss()
    #reward = FastAPReward()
    #reward_loss = RewardTripletLoss(margin=args['train_options']['margin'])
    if args['train_options']['loss'] == "triplet":
        loss = TripletLoss(margin=args['train_options']['margin'])

    mining_func = None
    if args['train_options']['loss'] == "triplet_pml":
        distance = distances.CosineSimilarity()
        #reducer = reducers.ThresholdReducer(low=0)
        loss = TripletMarginLoss(margin=0.35, distance=distance)#, reducer=reducer)
        #mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="hard")
        #loss = TripletMarginLoss(margin=args['train_options']['margin'])

    if args['train_options']['loss'] == "rewardTriplet":
        loss = RewardTripletLoss(margin=args['train_options']['margin'])

    if args['train_options']['loss'] == "fastAP":
        loss = FastAPLoss(num_bins=10)

    if args['train_options']['loss'] == "smoothAP":
        loss = SmoothAP(args['train_options']['sigmoid_temperature'])

    if args['train_options']['loss'] == "smoothAP_brown":
        m_per_class = args['train_options']['sampler_m']
        anneal = args['train_options']['sigmoid_temperature']
        batch_size =  args['train_options']['batch_size']
        num_classes = int(batch_size/m_per_class)
        feat_dims = 64*100

        loss = SmoothAP_brown(anneal, batch_size, num_classes, feat_dims)

    if args['train_options']['loss'] == "RankedListLoss":
        loss = RankedListLoss(margin=args['train_options']['margin'], Tn=0, Tp= 0 )

    if args['train_options']['loss'] == "RankedListReward":
        loss = RankedListReward(margin=args['train_options']['margin'], Tn=0, Tp= 0 )

    if args['train_options']['loss'] == "bbMapLoss":
        loss = MapLoss(lambda_val=args['train_options']['lambda_val'], 
                       margin=args['train_options']['bb_margin'],
                       interclass_coef=args['train_options']['interclass_coef'], 
                       batch_memory=args['train_options']['batch_memory'])
        
    print(f'''Using {args['train_options']['loss']} Loss''')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger, reward_baseline) #, mining_func= mining_func)
        mAP = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')

        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'model.pt'))

        epochs_to_save = np.arange(0,10)
        if epoch in epochs_to_save:
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, f'model_epoch_{epoch}.pt'))
            

        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval() 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def add_np_embeddingsto_table(logger, args):
    loaded_fs = np.load('/cluster/qy41tewa/rl-map/dataset/pfs_tf_smoothAP_256.npz')
    pfs_tf = loaded_fs['pfs_tf']
    writer = loaded_fs['writer']
    addtoTable(f"features", pfs_tf, writer, args, logger, num_writers=720)
    
    if args['rerank']:
        print(f'Reranking...')
        pfs_tf_rerank = sgr.sgr_reranking(torch.tensor(pfs_tf), k=1, layer=1, gamma=0.4)[0]
        addtoTable("features_reranked", pfs_tf_rerank, writer, args, logger, num_writers=720)
        


def reward_tune(args):

    #breakpoint()
    
    logger = prepare_logging(args)
    logger.update_config(args)

    backbone = getattr(resnets, args['model']['name'], None)()
    #backbone_writer = getattr(resnets, args['model_writer']['name'], None)()
    if not backbone:
        print("Unknown backbone!")
        raise

    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f'''Using {args['model'].get('encoding', 'netvlad')} as encoding.''')
    print('----------')

    random = args['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random)

    #random_writer = args['model_writer'].get('encoding', None) == 'netrvlad'
    #model_writer = Model(backbone, dim=64, num_clusters=args['model_writer']['num_clusters'], random=random)
    
    model.train()
    model = model.cuda()
    
    #model_writer.train()
    #model_writer = model.cuda()

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 
        #model.classification = True

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
    #else:
    #    raise ValueError('No checkpoint provided. Train model and provide checkpoint for reward finetuning.')


    
    if args['train']:
        

        tfs = []

        if args.get('grayscale', None):
            tfs.extend([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3)
            ])
        else:
            tfs.append(transforms.ToTensor())
    
        if args.get('data_augmentation', None) == 'morph':
            tfs.extend([transforms.RandomApply(
                [Erosion()],
                p=0.3
            ),
            transforms.RandomApply(
                [Dilation()],
                p=0.3
            )])
    
        transform = transforms.Compose(tfs)
    
        if args['trainset']:
            d = WriterZoo.get(**args['trainset'])
            train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=['cluster', 'writer', 'page'])
        
        if args.get('use_test_as_validation', False):
            val_ds = WriterZoo.get(**args['testset'])
            if args.get('grayscale', None):
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=3)
                ])
            else:
                test_transform = transforms.ToTensor()
            val_ds = val_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])
    
            train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
            val_ds = torch.utils.data.Subset(val_ds, range(len(val_ds)))
        else:
            train_ds, val_ds = train_val_split(train_dataset)

        if args['plot_emb']:
            cluster_plot(model, train_ds, args, logger)
            return
        
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)
        save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))

    if args['test']:
        if args['reward_tuning']:
            model.fc = torch.nn.Identity()
        save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))
        test(model, logger, args, name='Test')

#    if args['train_writer']:
#        model_writer, optimizer_writer = train(model_writer, train_ds, val_ds, args, logger, optimizer)

    
    if args['add_np_embeddingsto_table']:
        add_np_embeddingsto_table(logger, args)

    if args['save_page_features']:
        save_page_features(model, logger, args, name='Test')

    logger.finish()

def set_subelement(data, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml', help='Path to the configuration file')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=2174, type=int,
                        help='seed')
    parser.add_argument('--train', default=False,
                        help='enables training')
    parser.add_argument('--plot-emb', default=False,
                        help='only plots the embeddings using checkpoint model')
    parser.add_argument('--test', default=False,
                        help='only test the checkpoint model')
    parser.add_argument('--updates', nargs='+', required=False, help='List of subelement paths and values in the format subelement=value')

    args = parser.parse_args()
        
    config = load_config(args)[0]

    if args.updates:
        for update in args.updates:
            subelement, value = update.split('=', 1)
            set_subelement(config, subelement, value)

    if os.environ.get('DATAINTMP'):
         #WriterZoo.datasets[config['trainset']['dataset']]['basepath'] = '/scratch/qy41tewa/rl-map/dataset/icdar19/color/'
        WriterZoo.datasets[config['trainset']['dataset']]['basepath'] = '/scratch/qy41tewa/dataset/'

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    
    reward_tune(config)
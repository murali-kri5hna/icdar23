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
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler, const_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from dataloading.writerdataset import WriterDataset

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling
from reranking import sgr

from aug import Erosion, Dilation

from utils.triplet_loss import TripletLoss, RewardTripletLoss
from utils.m_n_sampler import NPerClusterSampler
#from utils.fast_ap_reward import FastAPReward
from pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
from pytorch_metric_learning.losses.fast_ap_loss import FastAPLoss
from pytorch_metric_learning.losses.ranked_list_loss import RankedListLoss
#from pytorch_metric_learning import losses
from utils.Smooth_AP_loss import SmoothAP
from utils.Smooth_AP_loss_Brown import SmoothAP as SmoothAP_brown
from utils.bb_map_loss import MapLoss
from utils.ranked_list_reward import RankedListReward

from pytorch_metric_learning import distances, miners, reducers

from backbone import resnets
from backbone.model import Model, WriterModelFC, WriterResModel

from visualization.data_table import addtoTable

import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def compute_page_features(_features, writer, pages):
    _labels = list(zip(writer, pages))

    labels_np = np.array(_labels)
    features_np = np.array(_features)
    writer = labels_np[:, 0]
    page = labels_np[:, 1]

    page_features = []
    page_writer = []
    for w, p in tqdm(set(_labels), 'Page Features'):
        idx = np.where((writer == w) & (page == p))
        page_features.append(features_np[idx])
        page_writer.append(w)

    return page_features, page_writer


def encode_per_class(model, args, poolings=[]):
    testset = args['testset']
    ds = WriterZoo.datasets[testset['dataset']]['set'][testset['set']]
    basepath = WriterZoo.datasets[testset['dataset']]['basepath']
    path = os.path.join(basepath, ds['path'])
    regex = ds['regex']

    pfs_per_pooling = [[] for i in poolings]

    regex_w = regex.get('writer')
    regex_p = regex.get('page')

    srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    logging.info(f'Found {len(srcs)} images')
    writer = [int('_'.join(re.search(regex_w, Path(f).name).groups())) for f in srcs]
    page = [int('_'.join(re.search(regex_p, Path(f).name).groups())) for f in srcs]

    labels = list(zip(writer, page))

    if args.get('grayscale', None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        transform = transforms.ToTensor()

    np_writer = np.array(writer)
    np_page = np.array(page)

    print(f'Found {len(list(set(labels)))} pages.')

    writers = []
    for w, p in tqdm(set(labels), 'Page Features'):
        idx = np.where((np_writer == w) & (np_page == p))[0]
        fps = [srcs[i] for i in idx]
        ds = FilepathImageDataset(fps, pil_loader, transform)
        loader =  torch.utils.data.DataLoader(ds, batch_size=args['test_batch_size'], num_workers=4)

        feats = []
        for img in loader:
            img = img.cuda()

            with torch.no_grad():
                feat = model(img)
                feat = torch.nn.functional.normalize(feat)
            feats.append(feat.detach().cpu().numpy())

        feats = np.concatenate(feats)

        for i, pooling in enumerate(poolings):
            enc = pooling.encode([feats])
            pfs_per_pooling[i].append(enc)
        writers.append(w)

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]
    
    return pfs_per_pooling, writers


def inference(model, ds, args):
    '''
    Inference on the dataset and return the features, writers and pages.
    '''

    model.eval()
    
    # breakpoint()
    
    loader = torch.utils.data.DataLoader(ds, batch_size=args['test_batch_size'], num_workers=4)

    feats = []
    pages = []
    writers = []

    for sample, labels in tqdm(loader, desc='Inference'):
        if len(labels) == 3:
            w,p = labels[1], labels[2]
        else:
            w,p = labels[0], labels[1]

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()

        with torch.no_grad():
            emb = model(sample)
            emb = torch.nn.functional.normalize(emb)
        feats.append(emb.detach().cpu().numpy())
    
    feats = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages = np.concatenate(pages)   

    return feats, writers, pages

def test(model, logger, args, name='Test'):


    best_map = -1
    best_top1 = -1

    table = []
    columns = ['mAP', 'Top1']
    
    #breakpoint()
    
    loaded_fs = np.load(f'''{args['np_embeddings_path']}''')
    pfs_tf = loaded_fs['pfs_tf']
    writer = loaded_fs['writer']

    if args['rerank']:
            name = args['train_options']
            print(f'Reranking...')
            pfs_tf_rerank = sgr.sgr_reranking(torch.tensor(pfs_tf), k=1, layer=1, gamma=0.4)[0]

            if args['addWriterEmbeddingsToTable']:
                addtoTable(f'{poolings[0]}_features_reranked', pfs_tf_rerank, writer, args, logger, num_writers=720)

    _eval = Retrieval()
    print(f'Calculate mAP..')

    res, _ = _eval.eval(pfs_tf, writer)
    res_rerank, _ = _eval.eval(pfs_tf_rerank, writer)

    pca_dim = 512

    p = f'{pca_dim}' if pca_dim != -1 else 'full'
    meanavp = res['map']
    meanavp_rerank = res_rerank['map']

    if meanavp > best_map:
        best_map = meanavp
        best_top1 = res['top1']
        best_pooling = f'{poolings[0]}-{p}'
        #pk.dump(pca, open(os.path.join(logger.log_dir, 'pca.pkl'), "wb"))
        
    table.append([meanavp, res['top1']])
    table.append([f'{poolings[0]}-{p}-reranking', meanavp_rerank, res_rerank['top1']])
    print(f'{poolings[0]}-{p}-{name} MAP: {meanavp}')
    print(f'''{poolings[0]}-{p}-{name} Top-1: {res['top1']}''')
    print(f'{poolings[0]}-{p}-{name}-rerank MAP: {meanavp_rerank}')
    print(f'''{poolings[0]}-{p}-{name}-rerank Top-1: {res_rerank['top1']}''')
        
    
    logger.log_table(table, 'Results', columns)
    logger.log_value(f'Best-mAP', best_map)
    logger.log_value(f'Best-Top1', best_top1)
    print(f'Best-Pooling: {best_pooling}')

###########

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args['optimizer_options']['base_lr'],
                    weight_decay=args['optimizer_options']['wd'])
    return optimizer

def validate(model, val_ds, args):
    #desc, writer, pages = inference(model, val_ds, args)
    #print('Inference done')
    #pfs, writer = compute_page_features(desc, writer, pages)

    batch_size = 1
    model.cuda()
    model.eval()
    

    # Create the DataLoader
    #breakpoint()
    val_loader = torch.utils.data.DataLoader(val_ds, 
                                               batch_size=batch_size,
                                               num_workers=8)
    
    total_enc = []
    writers = []
    
    for encs, writer in tqdm(val_loader):
        encs = encs.cuda()
        writer = writer.cuda()
        encs = model(encs)
        encs = torch.nn.functional.normalize(encs)
        
        total_enc.append(encs.detach().cpu().numpy())
        writers.append(writer.detach().cpu().numpy())

    #breakpoint()
    
    total_enc = np.concatenate(total_enc)
    writers = np.concatenate(writers)
        
    torch.cuda.empty_cache()

    _eval = Retrieval()
    res, _ = _eval.eval(total_enc, writers)
    meanavp = res['map']

    return meanavp

                                        

    norm = 'powernorm'
    pooling = SumPooling(norm)
    descs = pooling.encode(pfs)

    _eval = Retrieval()
    res, _ = _eval.eval(descs, writer)
    meanavp = res['map']

    return meanavp


def train_one_epoch(model, train_ds, Loss, optimizer, scheduler, epoch, args, logger, **kwargs):

    model.train()
    model = model.cuda()

    # set up the triplet stuff
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels)[train_ds.indices], args['train_options']['sampler_m'], batch_size=args['train_options']['batch_size']) #length_before_new_iter=args['train_options']['length_before_new_iter'])
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=8)
    
    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    for i, (samples, label) in enumerate(pbar):

        #breakpoint()
        
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']
   
        samples = samples.cuda()
        samples.requires_grad=True

        l = label.cuda()

        emb = model(samples)


        if args['train_options']['loss'] == "triplet":
            loss = Loss(emb, l, emb, l)
    
        if args['train_options']['loss'] == "fastAP":
            loss = Loss(emb, l)
    
        if args['train_options']['loss'] == "smoothAP":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "smoothAP_brown":
            loss = Loss(emb)

        if args['train_options']['loss'] == "rewardTriplet":
            loss = Loss(emb, l, emb, l, reward_label, reward_baseline)
            reward = Loss.reward(emb, reward_label)
            logger.log_value(f'reward', reward.item())

        if args['train_options']['loss'] == "RankedListLoss":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "RankedListReward":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "bbMapLoss":
            loss = Loss(emb, l)

        if args['train_options']['loss'] == "triplet_pml":
            indices_tuple = mining_func(embeddings, labels)
            loss = Loss(emb, l, indices_tuple)
        
        logger.log_value(f'loss', loss.item())

        
        logger.log_value(f'loss', loss.item())
        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])

        # compute gradient and update weights
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    torch.cuda.empty_cache()
    return model

def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)

    if args['optimizer_options']['scheduler'] == 'const':
        lr_schedule = const_scheduler(args['optimizer_options']['base_lr'], epochs, niter_per_ep, args['optimizer_options']['warmup_epochs'])
    
    best_epoch = -1
    best_map = validate(model, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)

    if args['train_options']['loss'] == "triplet":
        loss = TripletLoss(margin=args['train_options']['margin'])

    mining_func = None
    if args['train_options']['loss'] == "triplet_pml":
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss = TripletMarginLoss(margin=0.1, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=0.1, distance=distance, type_of_triplets="hard"
        )
        #loss = TripletMarginLoss(margin=args['train_options']['margin'])

    if args['train_options']['loss'] == "rewardTriplet":
        loss = RewardTripletLoss(margin=args['train_options']['margin'])

    if args['train_options']['loss'] == "fastAP":
        loss = FastAPLoss()

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
        loss = RankedListLoss(margin=args['train_options']['margin'], Tn=1, Tp= -1 )

    if args['train_options']['loss'] == "RankedListReward":
        loss = RankedListReward(margin=args['train_options']['margin'], Tn=1, Tp= -1 )

    if args['train_options']['loss'] == "bbMapLoss":
        loss = MapLoss(lambda_val=args['train_options']['lambda_val'], 
                       margin=args['train_options']['bb_margin'],
                       interclass_coef=args['train_options']['interclass_coef'], 
                       batch_memory=args['train_options']['batch_memory'])

    print(f'''Using {args['train_options']['loss']} Loss''')
    

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger, mining_func = mining_func)
        mAP = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')


        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'writer_model.pt'))


        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'writer_model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval() 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger = getLogger(args["logger"])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def train_val_split(dataset, prop = 0.9):
    authors = list(set(dataset.labels))
    random.shuffle(authors)

    train_len = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors = authors[train_len:]

    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs = []

    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.labels[i]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    train = torch.utils.data.Subset(dataset, train_idxs)
    val = torch.utils.data.Subset(dataset, val_idxs)

    return train, val


def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)
    
    random = args['model'].get('encoding', None) == 'netrvlad'
    model = WriterModelFC(features=512, num_clusters=100, dim=64, random=True)
    #model = WriterResModel(backbone=resnets.resnet20(), num_clusters=100, dim=64, random=True)
    model.train()
    model = model.cuda()
    optimizer = get_optimizer(args, model)

    cluster_model = Model(backbone=resnets.resnet56(), dim=64, num_clusters=100, random=True)


    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        cluster_model.load_state_dict(checkpoint['model_state_dict'])    
        cluster_model.eval() 

        model.nv = cluster_model.nv

        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

    if args['train']:
        train_dataset = None
        
        loaded_fs = np.load(f'''{args['np_embeddings_path']}''')
        pfs_tf = loaded_fs['pfs_tf']
        writer = loaded_fs['writer']

        train_dataset = WriterDataset(pfs_tf, writer)
        
        train_ds, val_ds = train_val_split(train_dataset)
        
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

    # testing
    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'writer_model.pt'))
    
    if args['test']:
        test(model, logger, args, name='Test')
    
    logger.finish()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=2174, type=int,
                        help='seed')

    args = parser.parse_args()

    config = load_config(args)[0]

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    
    main(config)

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
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.utils import GPU, seed_everything, load_config, save_model, const_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling

from aug import Erosion, Dilation
from utils.triplet_loss import TripletLoss
from utils.arc_face_loss import ArcFaceLoss

from backbone import resnets
from backbone.model import Model, RewardtuneModelFC, RewardtuneModel 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from main import prepare_logging, train_val_split, get_optimizer

def calculate_map_recall(softmax_outputs, one_hot_labels):
    """
    Calculate the mean Average Precision (mAP) and recall given softmax outputs and one-hot encoded labels.

    Parameters:
    - softmax_outputs: Tensor of shape (N, C), where N is the number of samples and C is the number of classes.
    - one_hot_labels: Tensor of shape (N, C), where N is the number of samples and C is the number of classes.

    Returns:
    - mAP: Mean Average Precision
    - recall: Recall for each class
    """
    # Ensure the tensors are on the CPU
    softmax_outputs = softmax_outputs.cpu().detach().numpy()
    one_hot_labels = one_hot_labels.cpu().detach().numpy()
    
    num_classes = one_hot_labels.shape[1]
    
    average_precisions = []
    recalls = []
    
    for i in range(num_classes):
        # Extract the true labels and predicted scores for class i
        true_labels = one_hot_labels[:, i]
        predicted_scores = softmax_outputs[:, i]
        
        # Compute precision-recall pairs and average precision
        precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
        average_precision = average_precision_score(true_labels, predicted_scores)
        
        # Append results
        average_precisions.append(average_precision)
        recalls.append(recall.max())  # Using the maximum recall value
    
    # Calculate mean Average Precision (mAP)
    mAP = np.mean(average_precisions)
    return mAP, recalls

def train_one_epoch(model, classifier, train_ds, cross_entropy_loss, optimizer, scheduler, epoch, args, logger):

    model.eval()
    model = model.cuda()

    classifier.train()
    classifier.cuda()
    # set up the triplet stuff
    
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    data_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)
    
    pbar = tqdm(data_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(data_loader)
    logger.log_value('Epoch', epoch, commit=False)


    for i, (samples, labels) in enumerate(pbar):
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']
        
        if len(labels) == 3:
            writers, pages = labels[1], labels[2]
        else:
            writers, pages = labels[0], labels[1]

        samples = samples.cuda()
        # samples.requires_grad=True

        with torch.no_grad():
            emb = model(samples)
            emb.requires_grad=True
        
        outputs = classifier(emb)
        loss = cross_entropy_loss(outputs, writers.cuda())
        logger.log_value(f'loss', loss.item())

        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 2)
        optimizer.step()
    
    torch.cuda.empty_cache()
    return classifier

def inference(model, classifier, ds, args):
    '''
    Inference on the dataset and return the prediction, writers and pages.
    '''

    model.eval()
    classifier.eval()

    model = torch.nn.Sequential(model, classifier)

    loader = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=args['test_batch_size'])

    prediction = []
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
        prediction.append(emb.detach().cpu().numpy())
    
    prediction = np.concatenate(prediction)
    writers = np.concatenate(writers)
    pages = np.concatenate(pages)   

    return prediction, writers, pages

def validate(model, classifier, val_ds, args):
    prediction, writer, pages = inference(model, classifier, val_ds, args)
    print('Inference done')

    meanavp, recall = calculate_map_recall(prediction, writer)

    return meanavp

def test(model, classifier, logger, args, name='Test'):
    test_ds = WriterZoo.get(**args['testset'])
    if args.get('grayscale', None):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        test_transform = transforms.ToTensor()
    test_ds = test_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])

    prediction, writer, pages = inference(model, classifier, test_ds, args)
    mAP, recall = calculate_map_recall(prediction, writer)    

    logger.log_value(f'mAP', mAP)
    print(f'Val-mAP: {mAP}')

def train(model, classifier, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = const_scheduler(args['optimizer_options']['base_lr'], epochs, niter_per_ep, args['optimizer_options']['warmup_epochs'])

    best_epoch = -1
    best_map = validate(model, classifier, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)

    # loss = TripletLoss(margin=args['train_options']['margin'])
    # loss = ArcFaceLoss(num_classes=6400, embedding_size=6400)
    loss = torch.nn.CrossEntropyLoss()
    print('Using Cross Entropy Loss')

    # sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    # data_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)

    # optimizer = torch.optim.Adam(classifier.parameters(), lr=args['optimizer_options']['base_lr'], weight_decay=args['optimizer_options']['wd'])

    for epoch in range(epochs):
        classifier = train_one_epoch(model, classifier, train_ds, loss, optimizer, lr_schedule, epoch, args, logger)
        mAP = validate(model, classifier, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')

        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(classifier, optimizer, epoch, os.path.join(logger.log_dir, 'classifier.pt'))

        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'classifier.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    classifier.load_state_dict(checkpoint['model_state_dict'])    
    classifier.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def classification_train(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    backbone = getattr(resnets, args['model']['name'], None)()
    if not backbone:
        print("Unknown backbone!")
        raise

    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f'''Using {args['model'].get('encoding', 'netvlad')} as encoding.''')
    print('----------')

    random = args['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random)
    classifier = RewardtuneModelFC(num_writers=args['classifier']['num_writers'])
    
    # model.train()
    # model = model.cuda()

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

    optimizer_netvlad = get_optimizer(args, model)
    optimizer_classifier = get_optimizer(args, classifier)

    if args['checkpoint_netvlad']:
        print(f'''Loading NetVLAD model from {args['checkpoint_netvlad']}''')
        checkpoint = torch.load(args['checkpoint_netvlad'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 

        optimizer_netvlad.load_state_dict(checkpoint['optimizer_state_dict'])   
    else:
        print('No NetVLAD checkpoint provided. It is necessacary for training the classifier. Exiting...')
        return

    if args['checkpoint_classifier']:
        print(f'''Loading Classifier model from {args['checkpoint_classifier']}''')
        checkpoint = torch.load(args['checkpoint_classifier'])
        classifier.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 

        optimizer_classifier.load_state_dict(checkpoint['optimizer_state_dict'])   
    else:
        print('No Classifier checkpoint provided. Training from scratch...')

    if not args['only_test']:
        classifier, optimizer_classifier = train(model, classifier, train_ds, val_ds, args, logger, optimizer_classifier)

    save_model(classifier, optimizer_classifier, args['train_options']['epochs'], os.path.join(logger.log_dir, 'classifier.pt'))
    test(model, classifier, logger, args, name='Test')
    logger.finish()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml', help='Path to the configuration file')
    parser.add_argument('--checkpoint-netvlad', default=None, type=str, metavar='PATH',
                        help='path to latest netvlad checkpoint (default: none)')
    parser.add_argument('--checkpoint-classsifier', default=None, type=str, metavar='PATH',
                        help='path to latest netvlad checkpoint (default: none)')
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
    
    classification_train(config)
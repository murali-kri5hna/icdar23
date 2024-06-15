import logging, os, argparse, math, random, re, glob

import torch
import torch.optim as optim
import numpy as np

from torchvision import transforms

from utils.utils import load_config, getLogger, save_model, const_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from aug import Erosion, Dilation

from backbone import resnets
from backbone.model import Model

from main import prepare_logging, train_val_split, test, get_optimizer, validate




# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize previous MaP for reward shaping
previous_map = 0

# Fine-tuning loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_reward = 0.0
    
    for inputs, labels in train_loader:  # Assuming train_loader is defined
        optimizer.zero_grad()
        
        # Forward pass through NetVLAD
        features = model(inputs)
        
        # Compute reward
        current_map = compute_map(features, labels)
        reward = shaped_reward(current_map, previous_map)
        
        # Compute negative reward for gradient ascent
        loss = -reward
        loss.backward()
        
        optimizer.step()
        running_reward += reward
        
        # Update previous MaP
        previous_map = current_map
    
    print(f"Epoch {epoch + 1}, Reward: {running_reward / len(train_loader)}")

    # Evaluate model with MaP
    with torch.no_grad():
        val_features, val_labels = [], []
        for inputs, labels in val_loader:  # Assuming val_loader is defined
            features = model(inputs)
            val_features.append(features)
            val_labels.append(labels)
        
        val_features = torch.cat(val_features)
        val_labels = torch.cat(val_labels)
        current_map = compute_map(val_features, val_labels)
    
    print(f"Validation MaP: {current_map}")

def train_one_epoch(model, train_ds, triplet_loss, optimizer, scheduler, epoch, args, logger):

    model.train()
    model = model.cuda()

    # set up the triplet stuff
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)
    
    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    for i, (samples, label) in enumerate(pbar):
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

        if args['train_label'] == 'cluster':
            l = label[0]
        if args['train_label'] == 'writer':
            l = label[1]

        l = l.cuda()

        emb = model(samples)

        loss = triplet_loss(emb, l, emb, l)
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
    lr_schedule = const_scheduler(optimizer, args['train_options']['lr'], niter_per_ep)

    best_epoch = -1
    best_map = validate(model, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)

    # loss = TripletLoss(margin=args['train_options']['margin'])
    # print('Using Triplet Loss')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger)
        mAP = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')


        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'model.pt'))


        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval() 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer



def reward_tune(args):
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
    model.train()
    model = model.cuda()

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
        train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=['writer', 'page'])
    
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

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
    else:
        print('No checkpoint provided. Train model and provide checkpoint for reward finetuning.')

    model, optimizer = train(model, optimizer, train_ds, val_ds, args['finetune_options']['epochs'], logger)

    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))
    test(model, logger, args, name='Test')
    logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml', help='Path to the configuration file')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()
        
    config = load_config(args)
    
    reward_tune(config)
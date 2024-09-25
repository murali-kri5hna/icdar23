from tqdm import tqdm

import numpy as np
import torch

from pytorch_metric_learning import samplers


def addtoTable(name, embeddings, labels, args, logger, num_writers=720):

    #breakpoint()
    
    num_stored = min(len(labels), num_writers)
    labels = list(labels)

    converted_embeddings = list(map(lambda embedding: embedding.tolist(), embeddings))
        
    columns = ["Embedding"] + ["Writer"] #, "Page"]
    data = list(zip(converted_embeddings[:num_stored], labels[:num_stored]))
    data = list(map(list,data))

    #data = [[arr.tolist(), label] for arr, label in data]

    logger.log_table(data, name, columns)


def cluster_plot(model, train_ds, args, logger):
    model.eval()
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))

   # sampler = MNSampler(labels[args['train_label']], 
   #                     args['train_options']['sampler_m'], 
   #                     args['train_options']['sampler_n'], 
   #                     labels['writer'], 
   #                     labels['page'],
   #                     length_before_new_iter=args['train_options']['length_before_new_iter'])
    
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)

    pbar = tqdm(train_triplet_loader)
    data = []
    total_embs = []
    total_labels = []

    for i, (samples, labels) in enumerate(pbar):

        samples = samples.cuda()
        samples.requires_grad=False

        emb = model(samples)
        ## Referring to the validation step, we are utilizing the steps in inference and compute_page_features
        emb = torch.nn.functional.normalize(emb) 
        feats = emb.detach().cpu().numpy()
        total_embs.append(feats)
        total_labels.append([label.detach().cpu().numpy() for label in labels])
        

    total_embs = list(np.concatenate(total_embs))
    total_labels = list(np.concatenate(total_labels, axis=1))
    
    columns = ["Embedding"] + ["Cluster", "Writer", "Page"]
    data = list(zip(total_embs,*total_labels))
    data = list(map(list,data))

    logger.log_table(data, "NetVLAD Embeddings", columns)

import torch
import time
#from torch.utils.benchmark import Timer, Measurement
#from memory_profiler import memory_usage

from pytorch_metric_learning.losses.fast_ap_loss import FastAPLoss
from Smooth_AP_loss import SmoothAP
from Smooth_AP_loss_o import SmoothAP as SmoothAP1
from Smooth_AP_loss2 import SmoothAP as SmoothAP2
from Smooth_AP_loss3 import SmoothAP as SmoothAP3
from Smooth_AP_loss4 import SmoothAP as SmoothAP4


def measure_time_and_memory(loss_fn, embeddings, labels, device):
    #torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    #with torch.autocast(device_type=device):
    #    loss = loss_fn(embeddings, labels=labels)
    loss = loss_fn(embeddings, labels=labels)
    loss.backward()
    end_time = time.time()
    #memory_allocated = torch.mps.current_allocated_memory() / 1024**2
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    
    elapsed_time = end_time - start_time
    return elapsed_time, memory_allocated, loss #, memory_allocated

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)  # For reproducibility
    batch_size = 512
    num_classes = 16
    feat_dims = 10
    anneal = 0.01
    labels = torch.tensor([i // int(batch_size/num_classes) for i in range(batch_size)])

    inputs = torch.randn(batch_size, feat_dims, requires_grad=True).to(device)
    #inputs = torch.randint(-5,5,(batch_size, feat_dims)) + inputs
    loss_fns = [
        #SmoothAP(anneal=anneal, batch_size=batch_size, num_id=num_classes, feat_dims=feat_dims),
        #SmoothAP1(anneal=anneal, batch_size=batch_size, num_id=num_classes, feat_dims=feat_dims),
        #SmoothAP2(anneal=anneal, batch_size=batch_size, num_id=num_classes, feat_dims=feat_dims),
        SmoothAP3(anneal=anneal, batch_size=batch_size, num_id=num_classes, feat_dims=feat_dims),
        SmoothAP4(anneal=anneal, batch_size=batch_size, num_id=num_classes, feat_dims=feat_dims),
        FastAPLoss()
    ]

    for i, loss_fn in enumerate(loss_fns, start=1):
        elapsed_time, memory_allocated, loss = measure_time_and_memory(loss_fn, inputs, labels, device)
        print(f"SmoothAPLoss{i}: Time = {elapsed_time:.4f}s, Memory = {memory_allocated:.2f}MB, Loss = {loss}")


if __name__ == "__main__":
    main()
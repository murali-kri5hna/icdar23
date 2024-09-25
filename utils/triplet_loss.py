import torch
from torch import nn
from utils.fast_ap_reward import FastAPReward

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs_col, targets_col, inputs_row, targets_row):
        loss_list, n = self.compute_loss_list(inputs_col, targets_col, inputs_row, targets_row)
        loss = sum(loss_list) / n
        return loss
    
    def compute_loss_list(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_
    
        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]
    
                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]
    
                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]
    
                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]
    
                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)
        return loss, n   


class RewardTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(RewardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs_col, targets_col, inputs_row, targets_row, reward_labels, reward_baseline, **kwargs):
        loss_list, n = self.compute_loss_list(inputs_col, targets_col, inputs_row, targets_row)
        rewards = FastAPReward(num_bins=1600).compute_reward(inputs_col, reward_labels)
        rewards_tensor = torch.tensor(rewards).cuda()
        loss_tensor = torch.stack(loss_list).cuda()

        #reward_baseline = sum(rewards_tensor) / n
        #breakpoint()
        
        #rewards_tensor = (1 - 10 * rewards_tensor) # - reward_baseline)
        rewards_tensor = (1 - rewards_tensor) # - reward_baseline)
        for key, value in kwargs.items():
                if key == 'logger':
                    logger = value
                    reward = sum(rewards) / n
                    logger.log_value(f'reward', reward.item())
        
        #multiplied_loss_list = [loss * reward for loss, reward in zip(loss_list, rewards_tensor)]
        multiplied_loss_list = loss_tensor*rewards_tensor
        
        loss = torch.sum(multiplied_loss_list) / n
        return loss

    def reward(self, inputs_col, targets_col):
        n = inputs_col.size(0)
        rewards = FastAPReward(num_bins=1600).compute_reward(inputs_col, targets_col)
        return sum(rewards)/n

    def compute_loss_list(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_
    
        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]
    
                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]
    
                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]
    
                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]
    
                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)
        return loss, n 

if __name__ == '__main__':
    labels = torch.concat([torch.arange(0,10) for i in range(10)]).cuda()
    x = torch.rand(100,128).cuda()
    loss = TripletLoss()
    print(loss(x, labels, x, labels))
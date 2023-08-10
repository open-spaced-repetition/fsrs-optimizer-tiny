
from tinygrad.tensor import Tensor
from typing import List, Optional
import numpy as np
import tinygrad.nn.optim as optim
from tqdm.auto import tqdm
import pandas as pd
import math

np.random.seed(2023)


def power_forgetting_curve(t, s):
    return (1 + t / (9 * s)) ** -1

class FSRS:
    def __init__(self, w: List[float]):
        self.s = np.array(w[:4], dtype=np.float32)
        self.w = Tensor(w[4:])

    def stability_after_success(self, s: Tensor, new_d: Tensor, r: Tensor, rating: Tensor) -> Tensor:
        hard_penalty = Tensor.where(rating == 2, self.w[11], 1)
        easy_bonus = Tensor.where(rating == 4, self.w[12], 1)
        new_s = s * (1 + Tensor.exp(self.w[4]) *
                        (11 - new_d) *
                        Tensor.pow(s, -self.w[5]) *
                        (Tensor.exp((1 - r) * self.w[6]) - 1) * 
                        hard_penalty * 
                        easy_bonus)
        return new_s.realize()

    def stability_after_failure(self, s: Tensor, new_d: Tensor, r: Tensor) -> Tensor:
        new_s = self.w[7] * \
                Tensor.pow(new_d, -self.w[8]) * \
                (Tensor.pow(s + 1, self.w[9]) - 1) * \
                Tensor.exp((1 - r) * self.w[10])
        return new_s.realize()

    def step(self, i, X: Tensor, stability: Optional[Tensor]=None, difficulty: Optional[Tensor]=None):
        if i == 0:
            keys = np.array([[1,2,3,4]] * X[:,1].shape[0])
            # first learn, init memory states
            new_s = np.zeros(X[:,1].shape[0], dtype=np.float32)
            index = np.nonzero((X[:,1].unsqueeze(1) == keys).numpy())
            new_s[index[0]] = self.s[index[1]]
            new_s = Tensor(new_s, requires_grad=False)
            new_d = self.w[0] - self.w[1] * (X[:,1] - 3)
            new_d = new_d.clip(1, 10)
        else:
            r = power_forgetting_curve(X[:,0], stability).realize()
            new_d = difficulty - self.w[2] * (X[:,1] - 3)
            new_d = self.mean_reversion(self.w[0], new_d)
            new_d = new_d.clip(1, 10)
            condition = X[:,1] > 1
            new_s = Tensor.where(condition, self.stability_after_success(stability, new_d, r, X[:,1]), self.stability_after_failure(stability, new_d, r))
        new_s = new_s.clip(0.1, 36500)
        return new_s.realize(), new_d.realize()

    def forward(self, inputs: Tensor, stability: Optional[Tensor]=None, difficulty: Optional[Tensor]=None) -> Tensor:
        '''
        :param inputs: shape[seq_len, batch_size, 2]
        '''
        ouptut = []
        for i, X in enumerate(inputs):
            stability, difficulty = self.step(i, X, stability, difficulty)
            ouptut.append(stability)
        return Tensor.stack(ouptut)

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return (self.w[3] * init + (1-self.w[3]) * current).realize()
    
    def weight_clip(self):
        # min_bounds = Tensor([1, 0.1, 0.1, 0, 0, 0.1, 0.01, 0.5, 0.01, 0.01, 0.01, 0, 1])
        # max_bounds = Tensor([10, 5, 5, 0.5, 2, 0.8, 1.5, 5, 0.2, 0.9, 2, 1, 10])
        # self.w = self.w.clip(min_bounds, max_bounds)
        # t is your tensor
        w = self.w.chunk(13, 0)

        w[0] = w[0].clip(1, 10)
        w[1] = w[1].clip(0.1, 5)
        w[2] = w[2].clip(0.1, 5)
        w[3] = w[3].clip(0, 0.5)
        w[4] = w[4].clip(0, 2)
        w[5] = w[5].clip(0.1, 0.8)
        w[6] = w[6].clip(0.01, 1.5)
        w[7] = w[7].clip(0.5, 5)
        w[8] = w[8].clip(0.01, 0.2)
        w[9] = w[9].clip(0.01, 0.9)
        w[10] = w[10].clip(0.01, 2)
        w[11] = w[11].clip(0, 1)
        w[12] = w[12].clip(1, 10)

        self.w = Tensor.cat(*w)

def pad_sequence(sequences, padding_value=0):
    """
    Args:
        sequences: list of numpy arrays, each of shape (seq_length_i,), could be different seq_length_i for each array
        padding_value: int, value for padded elements

    Returns:
        padded_sequences: numpy array of shape (max_seq_length, batch_size)
    """
    batch_size = len(sequences)
    max_seq_length = max([len(seq) for seq in sequences])
    try:
        feature_size = sequences[0].shape[1]
        padded_sequences = np.full((max_seq_length, batch_size, feature_size), padding_value, dtype=np.float32)
    except:
        padded_sequences = np.full((max_seq_length, batch_size), padding_value, dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_length = len(seq)
        padded_sequences[:seq_length, i] = seq

    return padded_sequences

class RevlogDataset:
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError('Training data is inadequate.')
        self.seq_len = df['ys'].map(len).values
        self.features = pad_sequence(df['features'].values).transpose(1, 0, 2)
        self.delta_ts = pad_sequence(df['delta_ts'].values).transpose(1, 0)
        self.ys = pad_sequence(df['ys'].values).transpose(1, 0)
    def __getitem__(self, idx):
        return self.features[idx], self.delta_ts[idx], self.ys[idx], self.seq_len[idx]

    def __len__(self):
        return len(self.ys)

from collections import defaultdict

class RevlogSampler:
    def __init__(self, data_source: RevlogDataset):
        self.data_source = data_source
        lengths = np.array(data_source.seq_len)
        batches = defaultdict(list)
        for i, seq_len in enumerate(lengths):
            batches[seq_len].append(i)
        self.batch_indices = [np.array(batches[seq_len]) for seq_len in sorted(batches)]
        self.batch_nums = len(self.batch_indices)

    def __iter__(self):
        yield from (self.batch_indices[idx] for idx in range(self.batch_nums))

    def __len__(self):
        return len(self.data_source)


def loss_fn(y_pred, y_true):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cosine_annealing_lr(lr, step_count, T_max, eta_min = 0):
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * step_count / T_max)) / (1 + math.cos(math.pi * (step_count - 1) / T_max))
    return lr

if __name__ == "__main__":
    model = FSRS([1.14, 1.01, 5.43, 14.11, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61])
    batch = Tensor([[[0, 3], [0, 3], [0, 3]],[[1, 3], [1, 2], [1, 1]]])
    stabilities = model.forward(batch)
    print(stabilities.numpy())
    dataset = pd.read_csv("./seq2seq.tsv", sep='\t')
    dataset = dataset.applymap(lambda x: np.array(list(map(int, x.split(',')))))
    dataset = dataset[dataset['t_history'].map(len) > 1]
    dataset['features'] = dataset[['t_history', 'r_history']].apply(lambda x: np.stack((x[0], x[1]), axis=1), axis=1)
    ds = RevlogDataset(dataset)
    sampler = RevlogSampler(ds)
    optim = optim.Adam([model.w], lr=1e-3)
    n_epochs = 5
    epoch_len = len(ds)
    total_iterations = n_epochs * epoch_len
    step_count = 0
    pbar = tqdm(desc="train", colour="red", total=total_iterations)
    for epoch in range(n_epochs):
        loss_list = []
        for i, index in enumerate(sampler):
            optim.zero_grad()
            features, delta_ts, ys, seq_lens = ds[index]
            real_batch_size = seq_lens.shape[0]
            max_seq_len = max(seq_lens)
            features = features.transpose(1, 0, 2)[:max_seq_len]
            delta_ts = delta_ts.transpose(1, 0)[:max_seq_len]
            ys = ys.transpose(1, 0)[:max_seq_len]
            # tqdm.write(f"{features.shape}, {delta_ts.shape}, {ys.shape}, {seq_lens.shape}")
            stabilities = model.forward(Tensor(features))
            retentions = power_forgetting_curve(Tensor(delta_ts), stabilities)
            loss = loss_fn(retentions, Tensor(ys))
            loss_list.extend(loss.numpy().reshape(1, -1)[0])
            # tqdm.write(f"{loss.numpy()}")
            avg_loss = loss.sum()
            # tqdm.write(f"{avg_loss.numpy()}")
            avg_loss.backward()
            optim.step()
            model.weight_clip()
            step_count += real_batch_size
            optim.lr = cosine_annealing_lr(optim.lr, step_count, total_iterations)
            pbar.update(real_batch_size)
        tqdm.write(f"loss: {np.mean(loss_list)}")
        tqdm.write(f"{list(model.w.numpy())}")


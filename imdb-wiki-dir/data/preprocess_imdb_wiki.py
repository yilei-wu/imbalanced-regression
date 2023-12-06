########################################################################################
# MIT License

# Copyright (c) 2021 Yuzhe Yang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################################################################################
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = './'


def visualize_dataset(db="imdb_wiki"):
    file_path = join(BASE_PATH, f"{db}.csv")
    data = pd.read_csv(file_path)
    _, ax = plt.subplots(figsize=(6, 3), sharex='all', sharey='all')
    ax.hist(data['age'], range(max(data['age'])))
    ax.set_xlim([0, 120])
    plt.title(f"{db.upper()} (total: {data.shape[0]})")
    plt.tight_layout()
    plt.show()


def make_balanced_testset(db="imdb_wiki", max_size=150, seed=666, verbose=True, vis=True, save=True):
    file_path = join(BASE_PATH, f"{db}.csv")
    df = pd.read_csv(file_path)
    df['age'] = df.age.astype(int)
    val_set, test_set = [], []
    import random
    random.seed(seed)
    for value in range(121):
        curr_df = df[df['age'] == value]
        curr_data = curr_df['path'].values
        random.shuffle(curr_data)
        curr_size = min(len(curr_data) // 5, max_size)
        val_set += list(curr_data[:curr_size])
        test_set += list(curr_data[curr_size:curr_size * 2])
    if verbose:
        print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
    assert len(set(val_set).intersection(set(test_set))) == 0
    combined_set = dict(zip(val_set, ['val' for _ in range(len(val_set))]))
    combined_set.update(dict(zip(test_set, ['test' for _ in range(len(test_set))])))
    df['split'] = df['path'].map(combined_set)
    df['split'].fillna('train', inplace=True)
    if verbose:
        print(df)
    if save:
        df.to_csv(str(join(BASE_PATH, f"{db}.csv")), index=False)
    if vis:
        _, ax = plt.subplots(3, figsize=(6, 9), sharex='all')
        df_train = df[df['split'] == 'train']
        # df_train = df_train[(df_train['age'] <= 20) | (df_train['age'] > 50)]
        ax[0].hist(df_train['age'], range(max(df['age'])))
        ax[0].set_title(f"[{db.upper()}] train: {df_train.shape[0]}")
        ax[1].hist(df[df['split'] == 'val']['age'], range(max(df['age'])))
        ax[1].set_title(f"[{db.upper()}] val: {df[df['split'] == 'val'].shape[0]}")
        ax[2].hist(df[df['split'] == 'test']['age'], range(max(df['age'])))
        ax[2].set_title(f"[{db.upper()}] test: {df[df['split'] == 'test'].shape[0]}")
        ax[0].set_xlim([0, 120])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    make_balanced_testset()
    visualize_dataset(db="imdb_wiki")

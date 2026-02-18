import numpy as np
import pandas as pd
import torch
import os

from umap import UMAP
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset
import shap

def load_or_compute_shap_values(
        TRAIN_CSV_PATH,
        BATCH_DIR,
        classifier,
        test_loader,
        trainset,
        pixels,
        in_channels=3,
        batch_size=4
    ):
    if os.path.exists(TRAIN_CSV_PATH):
        print('Loading Shapley values...')
        shap_values_train_df = pd.read_csv(TRAIN_CSV_PATH)
    else:
        print('Making Shapley values...')
        images = []
        max_num = 101
        for batch in iter(test_loader):
            batch_image, _ = batch
            images.extend(batch_image)
            if len(images) >= max_num:
                images = images[:max_num + 1]
                break
        background = np.array(images)[:100]
        print(background.shape)
        background = torch.Tensor(background)
        e = shap.DeepExplainer(classifier.to('cpu'), background)

        shap_values_train = []
        train_labels = []
        batch_ind = 1
        shap_batch = []
        batch_labels = []
        items_per_batch = 500
        num_batches = int(np.ceil(len(trainset) / items_per_batch))
        print(f'We need {num_batches} batches to cover the dataset')
        batches_present = 0

        while batch_ind <= num_batches:
            if os.path.exists(f'{BATCH_DIR}/batch_{batch_ind}.npy'):
                print(f'Batch {batch_ind} exists!')
                shap_batch = np.load(f'{BATCH_DIR}batch_{batch_ind}.npy')
                shap_values_train.extend(shap_batch)
                batch_labels = np.load(f'{BATCH_DIR}batch_{batch_ind}_labels.npy')
                train_labels.extend(batch_labels)
                batches_present += 1
                batch_ind += 1
            else:
                break

        if num_batches != batches_present:
            if batch_ind == 1:
                train_loader = DataLoader(
                    trainset,
                    batch_size=batch_size,
                    shuffle=False
                )
            else:
                train_indecies = np.arange(len(trainset))
                checkpoint = items_per_batch * batches_present
                print(f'We have {checkpoint} SHAP values so far!')
                remaining_indecies = train_indecies[checkpoint:]
                print(f'We need to calculate {len(remaining_indecies)} more SHAP values!')
                shap_subset = Subset(trainset, remaining_indecies)
                train_loader = DataLoader(
                    shap_subset,
                    batch_size=4,
                    shuffle=False
                )

            pbar = tqdm(train_loader, desc=f"Processing train data (current batch: {batch_ind}/{num_batches})", unit="data")
            for data, target in pbar:
                data_size = data.shape[0]
                classification, _ = classifier.compute_class(data)
                shap_values_full = e.shap_values(data)
                shap_values_full = shap_values_full.reshape((data_size, -1, 10))
                train_labels.extend(target.tolist())
                for i in range(len(shap_values_full)):
                    input_prediction = classification[i]
                    shap_values = np.array([shap_values_full[i][y][input_prediction] for y in
                                            range(shap_values_full[i].shape[0])]).flatten()
                    shap_values_train.append(shap_values)
                    shap_batch.append(shap_values)
                    batch_labels.append(target.tolist()[i])

                if len(shap_batch) >= items_per_batch:
                    shap_batch = np.array(shap_batch)
                    batch_labels = np.array(batch_labels)
                    np.save(f'{BATCH_DIR}batch_{batch_ind}.npy', shap_batch)
                    np.save(f'{BATCH_DIR}batch_{batch_ind}_labels.npy', batch_labels)
                    batch_ind += 1
                    shap_batch = []
                    batch_labels = []
                    pbar.set_description(f"Processing train data (current batch: {batch_ind}/{num_batches})")
            if len(trainset) % items_per_batch != 0:
                np.save(f'{BATCH_DIR}/batch_{batch_ind}.npy', shap_batch)
                np.save(f'{BATCH_DIR}/batch_{batch_ind}_labels.npy', batch_labels)

        shap_values_train = np.array(shap_values_train)
        shap_values_train_df = pd.DataFrame(shap_values_train, columns=[f'feature_{x}' for x in range(pixels * pixels * in_channels)])
        shap_values_train_df = shap_values_train_df.join(pd.DataFrame(train_labels, columns=['class']))
        shap_values_train_df.to_csv(TRAIN_CSV_PATH, index_label='ind')
    print('Done!\n')
    return shap_values_train_df


def load_2d_shap(path, shap_values_df, seed, use_umap=False):
    if not os.path.exists(path):
        print('Making reduced points...')
        train_labels = shap_values_df['class']
        to_remove = ['ind', 'class'] if 'ind' in shap_values_df.columns else ['class']
        to_reduce = shap_values_df.drop(to_remove, axis=1, errors='ignore')
        print(to_reduce)
        to_reduce = to_reduce.to_numpy()
        scaler = MinMaxScaler()
        if use_umap:
            print('This is running')
            reducer = UMAP(n_components=2, random_state=seed)
        else:
            reducer = TSNE(n_components=2, random_state=seed)
        reduced_dataset = scaler.fit_transform(reducer.fit_transform(to_reduce))
        reduced_data_df = pd.DataFrame(reduced_dataset, columns=['x', 'y']).join(pd.DataFrame(train_labels, columns=['class']))
        reduced_data_df.to_csv(path, index=False)
    else:
        print('Loading reduced points...')
        reduced_data_df = pd.read_csv(path)
    print('Done!\n')
    return reduced_data_df


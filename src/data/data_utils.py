import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

        
# image files laod
def get_files(dir, format):
    assert format is not None, "File format is None"
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(format):
                paths.append(os.path.join(root, file))
    return paths


def save_npy(file, arr):
    with open(file, 'wb') as f:
        np.save(f, arr)
    return


def random_split(arr, split_ratio=0.8, random_seed=43):
    assert split_ratio > 0.6, "split ratio should > 0.6"
    n = len(arr)
    split = int(np.floor(n * split_ratio))

    np.random.seed(random_seed)
    np.random.shuffle(arr)
    train_arr, test_arr = arr[:split], arr[split:]
    return train_arr, test_arr


def denormalize(image, norm_range_max, norm_range_min):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def trunc(mat, trunc_max, trunc_min):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat


def save_fig(x, y, pred, path, original_result, pred_result, trunc_max, trunc_min):
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                        original_result[1],
                                                                        original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                        pred_result[1],
                                                                        pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(path)
    plt.close()

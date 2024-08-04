import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def color_histogram(img, bins=64):
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    rhist = np.bincount(img[:, :, 0].ravel(), minlength=bins)
    ghist = np.bincount(img[:, :, 1].ravel(), minlength=bins)
    bhist = np.bincount(img[:, :, 2].ravel(), minlength=bins)
    hist = np.stack([rhist, ghist, bhist], 0)
    hist = hist / (img.shape[0] * img.shape[1])
    return hist

def edge_direction_histogram(img, bins=64):
    if img.ndim == 3:
        img = img.sum(2)
    img = img.astype(float)
    gx = img[:, 2:] - img[:, :-2]
    gx = gx[:-2, :] + 2 * gx[1:-1, :] + gx[2:, :]
    gy = img[2:, :] - img[:-2, :]
    gy = gy[:, :-2] + 2 * gy[:, 1:-1] + gy[:, 2:]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gx, -gy)
    dirs = (bins * angle / (2 * np.pi)).astype(int) % bins
    hist = np.bincount(dirs.ravel(), weights=magnitude.ravel(), minlength=bins)
    hist = hist / max(1e-16, hist.sum())
    return hist

def cooccurrence_matrix(img, bins=8, distance=10):
    if img.ndim == 3:
        img = img.mean(2)
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)
    mat += mat.T
    mat = mat / mat.sum()
    return mat

def rgb_cooccurrence_matrix(img, quantization=3, distance=10):
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    if img.max() > 1:
        img = img / 255.0
    img = (img * (quantization - 1)).astype(int)
    bins = quantization ** 3
    img = (img * np.array([[[1, quantization, quantization ** 2]]])).sum(2)
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)
    mat += mat.T
    mat = mat / mat.sum()
    return mat

def _cooccurrence_matrix_dir(values, bins, di, dj):
    m, n = values.shape
    codes = values[:m - di, :n - dj] + bins * values[di:, dj:]
    entries = np.bincount(codes.ravel(), minlength=bins ** 2)
    return entries.reshape(bins, bins)

def process_images_in_directory(directory):
    features = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img = plt.imread(img_path)
                print(f"Processing {img_path}...")
                
                h = color_histogram(img)
                e = edge_direction_histogram(img)
                c = cooccurrence_matrix(img)
                r = rgb_cooccurrence_matrix(img)
                
                features.append((img_path, h, e, c, r))
                
                # Plotting histograms
                print(f"Plotting histograms for {img_path}...")
                plot_histograms(file, h, e, c, r)
                
                # Saving features to files (optional)
                np.savetxt(f"{img_path}_color_histogram.txt", h.reshape(1, -1), fmt="%.4g")
                np.savetxt(f"{img_path}_edge_direction_histogram.txt", e.reshape(1, -1), fmt="%.4g")
                np.savetxt(f"{img_path}_cooccurrence_matrix.txt", c.reshape(1, -1), fmt="%.4g")
                np.savetxt(f"{img_path}_rgb_cooccurrence_matrix.txt", r.reshape(1, -1), fmt="%.4g")
    
    return features

def plot_histograms(img_name, color_hist, edge_hist, cooc_matrix, rgb_cooc_matrix):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    axs[0, 0].bar(range(color_hist.shape[1]), color_hist[0], color='r', alpha=0.6, label='Red')
    axs[0, 0].bar(range(color_hist.shape[1]), color_hist[1], color='g', alpha=0.6, label='Green')
    axs[0, 0].bar(range(color_hist.shape[1]), color_hist[2], color='b', alpha=0.6, label='Blue')
    axs[0, 0].set_title(f'Color Histogram for {img_name}')
    axs[0, 0].legend()

    axs[0, 1].bar(range(len(edge_hist)), edge_hist, color='gray')
    axs[0, 1].set_title(f'Edge Direction Histogram for {img_name}')
    
    im = axs[1, 0].imshow(cooc_matrix, cmap='hot', interpolation='nearest')
    axs[1, 0].set_title(f'Co-occurrence Matrix for {img_name}')
    fig.colorbar(im, ax=axs[1, 0])

    im2 = axs[1, 1].imshow(rgb_cooc_matrix, cmap='hot', interpolation='nearest')
    axs[1, 1].set_title(f'RGB Co-occurrence Matrix for {img_name}')
    fig.colorbar(im2, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

def save_features_to_file(filename, features):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)

def load_features_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

if __name__ == "__main__":
    train_dir = "D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\cake\\cake-images\\train"
    test_dir = "D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\cake\\cake-images\\test"
    
    train_features_file = 'train_features.pkl'
    test_features_file = 'test_features.pkl'
    
    train_features = load_features_from_file(train_features_file)
    if train_features is None:
        train_features = process_images_in_directory(train_dir)
        save_features_to_file(train_features_file, train_features)
    
    test_features = load_features_from_file(test_features_file)
    if test_features is None:
        test_features = process_images_in_directory(test_dir)
        save_features_to_file(test_features_file, test_features)

import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def print_images(cant_images, images, title, random_seed=1973, cols=5):
    """
    Muestra un número arbitrario de imágenes aleatorias.
    """
    np.random.seed(random_seed)

    images = np.asarray(images)

    if images.ndim == 1:
        if images.shape[0] == 0:
            raise ValueError("El argumento 'images' no puede estar vacío.")
        images = images.reshape(1, -1)

    if images.ndim == 2:
        n_samples, n_pixels = images.shape
        if n_pixels == 784:
            images = images.reshape(n_samples, 28, 28)
        else:
            raise ValueError(
                "Las imágenes planas deben tener 784 píxeles (28x28) o un formato compatible."
            )

    if cant_images < 1:
        raise ValueError("cant_images debe ser un entero mayor o igual a 1.")

    n_samples = images.shape[0]
    if n_samples == 0:
        raise ValueError("El dataset de imágenes no puede estar vacío.")

    replace = cant_images > n_samples
    sample_indices = np.random.choice(n_samples, size=cant_images, replace=replace)
    sampled_images = images[sample_indices]

    ncols = min(cols, cant_images)
    nrows = int(np.ceil(cant_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, ax in enumerate(axes.flat[:cant_images]):
        image = sampled_images[idx]
        selected_index = int(sample_indices[idx])

        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        ax.axis("off")
        ax.set_title(f"Imagen {idx}", fontsize=10)

    for ax in axes.flat[cant_images:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plt.show()

def stratified_split(X, y, train_size=0.8, random_seed=1973):
    np.random.seed(random_seed)

    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    labels = np.unique(y)

    for label in labels:
        class_X = X[y==label]
        class_y = y[y==label]

        n_samples = len(class_X)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * train_size)

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train_list.append(class_X[train_indices])
        y_train_list.append(class_y[train_indices])
        
        X_val_list.append(class_X[val_indices])
        y_val_list.append(class_y[val_indices])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    
    # Vuelvo a mezclar para que no quede ordenado por clase
    train_shuffle_idx = np.random.permutation(len(X_train))
    val_shuffle_idx = np.random.permutation(len(X_val))
    
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]
    X_val = X_val[val_shuffle_idx]
    y_val = y_val[val_shuffle_idx]
    
    return X_train, X_val, y_train, y_val

def set_seed(seed=1973):
    """Fija todas las semillas para asegurar reproducibilidad en PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Forzar a PyTorch a usar algoritmos determinísticos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_stratified_sample(X, y, n_samples=3000, random_seed=1973):
    """
    Extrae una muestra estratificada exacta de n_samples.
    """
    np.random.seed(random_seed)
    classes = np.unique(y)
    n_classes = len(classes)
    samples_per_class = n_samples // n_classes
    
    indices = []

    for c in classes:
        c_indices = np.where(y == c)[0]
        # elijo aleatoriamente samples_per_class índices de la clase c sin reposicion
        chosen_indices = np.random.choice(c_indices, samples_per_class, replace=False)
        indices.extend(chosen_indices)
        
    # mezclo para que no quede ordenado por clase
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    return X[indices], y[indices]
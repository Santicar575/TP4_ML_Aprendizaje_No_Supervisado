import numpy as np

def standardize_data(X, mean=None, std=None):
    """
    Estandariza los datos restando la media y dividiendo por la desviación estándar.
    """
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8 # Evitar division por cero 

    X_std = (X - mean) / std
    return X_std, mean, std


def pca_fit(X_std):
    """
    Aplica PCA usando SVD y devuelve los componentes y la varianza explicada.
    """
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    
    explained_variance = (S ** 2) / (X_std.shape[0] - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    return Vt, explained_variance_ratio

def pca_transform(X_std, components, n_components):
    """
    Reduce la dimensionalidad proyectando los datos sobre los primeros n_components.
    """
    # selecciono los primeros n componentes con mayor varianza explicada
    Vt_truncated = components[:n_components]
    
    # proyecto los datos (X * V^T)
    X_reduced = np.dot(X_std, Vt_truncated.T)
    return X_reduced

def pca_inverse_transform(X_reduced, components, n_components, mean, std):
    """
    Reconstruye los datos desde el espacio latente de PCA al espacio original.
    """

    Vt_truncated = components[:n_components]
    
    # Proyecto al espacio original
    X_std_rec = np.dot(X_reduced, Vt_truncated)
    
    # vuelvo a la escala original
    X_rec = (X_std_rec * std) + mean
    
    return X_rec
import numpy as np

def k_means(X, n_clusters, n_inits=5, max_iters=1000, tol=1e-4, random_seed=1973):
    best_loss = np.inf
    best_centroids = None
    best_labels = None

    for init in range(n_inits):
        np.random.seed(random_seed + init)  # Cambiar la semilla en cada inicializacion para obtener resultados diferentes
        
        # inicializo los centroides de forma aleatoria
        centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        prev_loss = np.inf

        for iter in range(max_iters):
            # Asignar cada punto al centroide mas cercano
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Calcular la nueva posicion de los centroides
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                points_in_cluster = X[labels == k]
                if len(points_in_cluster) > 0:
                    new_centroids[k] = points_in_cluster.mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.choice(X.shape[0])]

            # Calcular la loss
            loss = np.sum((X - centroids[labels]) ** 2)

            # Verificar convergencia
            if abs(prev_loss - loss) < tol:
                break

            prev_loss = loss
            centroids = new_centroids

        if loss < best_loss:
            best_loss = loss
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_loss

def multivariate_normal_logpdf(X, mu, sigma):
    """
    Calcula la Log-PDF de una normal multivariada
    """
    n_samples, n_features = X.shape
    diff = X - mu
    
    sign, logdet = np.linalg.slogdet(sigma)
    
    if sign <= 0:
        return np.full(n_samples, -np.inf)
    
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.full(n_samples, -np.inf)
    
    quad_term = np.sum((diff @ inv_sigma) * diff, axis=1)
    
    log_pdf = -0.5 * (n_features * np.log(2 * np.pi) + logdet + quad_term)
    
    return log_pdf

def logsumexp(a, axis=1):

    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    out += a_max
    
    return np.squeeze(out, axis=axis)

def GMM(X, n_clusters, init_means=None, n_inits=5, max_iters=100, tol=1e-4, random_seed=1973):
    """
    Implementación de GMM usando el algoritmo Expectation-Maximization
    """
    n_samples, n_features = X.shape
    emp_cov = np.cov(X, rowvar=False) + np.eye(n_features) * 1e-6                           

    best_log_likelihood = -np.inf
    best_mu_k, best_sigma_k, best_pi_k, best_labels = None, None, None, None

    n_runs = 1 if init_means is not None else n_inits

    for init in range(n_runs):
        np.random.seed(random_seed + init)

        if init_means is not None:
            mu_k = init_means.astype(float).copy()
        else:
            mu_k = X[np.random.choice(n_samples, n_clusters, replace=False)].astype(float)

        sigma_k = np.array([emp_cov.copy() for _ in range(n_clusters)])
        pi_k = np.ones(n_clusters) / n_clusters
        prev_log_likelihood = -np.inf
        for i in range(max_iters):
            # Expectation step
            log_resp = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                log_resp[:, k] = np.log(pi_k[k]) + multivariate_normal_logpdf(X, mu_k[k], sigma_k[k])
            log_prob_norm = logsumexp(log_resp, axis=1)
            log_likelihood = np.sum(log_prob_norm)
            responsibilities = np.exp(log_resp - log_prob_norm[:, np.newaxis])
            
            # Maximization step
            Nk = np.sum(responsibilities, axis=0)
            for k in range(n_clusters):
                if Nk[k] < 1e-8:
                    # Resetear si hay un cluster vacio para evitar problemas numericos
                    mu_k[k] = X[np.random.choice(n_samples)]
                    sigma_k[k] = emp_cov
                    pi_k[k] = 1.0 / n_clusters
                    continue

                mu_k[k] = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / Nk[k]
                diff = X - mu_k[k]
                sigma_k[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / Nk[k]
                sigma_k[k] += np.eye(n_features) * 1e-6 # para evitar singularidades
            
            pi_k = Nk / n_samples

            # Verificar convergencia
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_mu_k = mu_k
            best_sigma_k = sigma_k
            best_pi_k = pi_k
            best_labels = np.argmax(responsibilities, axis=1)

    return best_mu_k, best_sigma_k, best_pi_k, best_labels, best_log_likelihood

def silhouette_score(X, labels):
    """
    Calcula el Silhouette Score global
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    
    X_sq = np.sum(X**2, axis=1)
    dist_matrix = np.sqrt(np.clip(X_sq[:, np.newaxis] + X_sq - 2 * np.dot(X, X.T), 0, None))
    
    s_i = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Distancias desde el punto 'i' a todos los demas puntos
        dists_i = dist_matrix[i]
        
        # distancia intra-cluster
        same_cluster_mask = (labels == cluster_i)
        same_cluster_mask[i] = False # Nos excluimos a nosotros mismos
        
        # si el cluster tiene un solo punto, por convención el silhouette es 0
        if np.sum(same_cluster_mask) == 0:
            s_i[i] = 0.0
            continue
            
        a_i = np.mean(dists_i[same_cluster_mask])
        
        # distancia al cluster vecino mas cercano 
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == cluster_i:
                continue
                
            other_cluster_mask = (labels == other_cluster)
            avg_dist_to_other = np.mean(dists_i[other_cluster_mask])
            
            if avg_dist_to_other < b_i:
                b_i = avg_dist_to_other
                
        # Silhouette para el punto 'i' 
        s_i[i] = (b_i - a_i) / max(a_i, b_i)
        
    # El score global es el promedio de todos los puntos
    return np.mean(s_i)

import numpy as np

def find_elbow_point(x, y):
    """
    Encuentra el punto del codo matemáticamente calculando la máxima 
    distancia perpendicular desde cada punto a la línea que une los extremos.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Coordenadas del primer y ultimo punto
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Vector de la linea
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        p1_to_p = p - p1
        proj = np.dot(p1_to_p, line_unitvec) * line_unitvec
        perp_vec = p1_to_p - proj
        distances.append(np.linalg.norm(perp_vec))
        
    # El codo es el indice de la distancia maxima
    elbow_idx = np.argmax(distances)
    return x[elbow_idx], y[elbow_idx]

def calculate_H_P(D, beta):
    """Funcion auxiliar para calcular Entropía y Probabilidades """
    P = np.exp(-D * beta)
    
    P = np.maximum(P, 1e-12) 
    
    sumP = np.sum(P)
    sumP = np.maximum(sumP, 1e-12) # para que el logaritmo no explote
    
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    
    return H, P / sumP

def compute_pairwise_distances(X):
    """Calcula la matriz de distancias cuadradas euclidianas."""
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return np.maximum(D, 0)

def get_perplexity_and_p(D, perplexity=30.0, tol=1e-5):
    n = D.shape[0]
    P = np.zeros((n, n))
    beta = np.ones((n, 1)) 
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        
        H, thisP = calculate_H_P(Di, beta[i])
        
        tries = 0
        while np.abs(H - logU) > tol and tries < 50:
            if H > logU:
                betamin = beta[i].copy()
                beta[i] = beta[i] * 2.0 if betamax == np.inf else (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                beta[i] = beta[i] / 2.0 if betamin == -np.inf else (beta[i] + betamin) / 2.0
            
            H, thisP = calculate_H_P(Di, beta[i])
            tries += 1
            
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P


def tsne(X, no_dims=2, perplexity=30.0, max_iter=300):
    """
    Algoritmo t-SNE principal.
    """
    n, d = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500 # Learning rate
    min_gain = 0.01
    
    Y = np.random.randn(n, no_dims) * 1e-4
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    
    
    D = compute_pairwise_distances(X)
    P = get_perplexity_and_p(D, perplexity)
    
    P = P + P.T
    P = P / np.sum(P)
    P = P * 4.0 
    P = np.maximum(P, 1e-12)
    
    print(f"t-SNE: Iniciando Descenso de Gradiente ({max_iter} iteraciones)...")
    for iter in range(max_iter):
        # Probabilidades en baja dimensión (Distribución t-Student)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            
        momentum = initial_momentum if iter < 20 else final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, 0) # Centrar
        
        if iter == 100:
            P = P / 4.0
            
        if (iter + 1) % 50 == 0:
            # Calcular la divergencia KL para mostrar progreso
            C = np.sum(P * np.log(P / Q))
            print(f"Iteración {iter+1}: Error (KL Divergence) = {C:.4f}")
            
    return Y

def align_clusters_to_classes(labels_pred, true_labels, n_classes=10):
    """Remapea cada cluster al color de la clase real mayoritaria que contiene."""
    clusters = np.unique(labels_pred)
    n_clusters = len(clusters)

    # Matriz de contingencia: filas = clusters, columnas = clases reales
    cont = np.zeros((n_clusters, n_classes), dtype=int)
    for i, c in enumerate(clusters):
        mask = labels_pred == c
        cont[i, :] = np.bincount(true_labels[mask], minlength=n_classes)

    # Lista de (coincidencias, idx_cluster, idx_clase) ordenada de mayor a menor
    pares = [(cont[i, j], i, j) for i in range(n_clusters) for j in range(n_classes)]
    pares.sort(reverse=True)

    mapping = {}
    clusters_usados = set()
    clases_usadas = set()

    for _, i, j in pares:
        c = clusters[i]
        if c not in clusters_usados and j not in clases_usadas:
            mapping[c] = j
            clusters_usados.add(c)
            clases_usadas.add(j)

    clases_libres = [j for j in range(n_classes) if j not in clases_usadas]
    for c in clusters:
        if c not in mapping:
            mapping[c] = clases_libres.pop() if clases_libres else int(cont[list(clusters).index(c)].argmax())

    new_labels = np.array([mapping[l] for l in labels_pred])
    return new_labels

def plot_cluster_quality(labels_pred, true_labels, title, k, class_names):
    import matplotlib.pyplot as plt
    cluster_counts = np.bincount(labels_pred, minlength=k)
    
    # Matriz para guardar cuántos elementos de cada True Class hay en cada Cluster
    composition_matrix = np.zeros((k, 10))
    for c in range(k):
        true_in_c = true_labels[labels_pred == c]
        if len(true_in_c) > 0:
            counts = np.bincount(true_in_c, minlength=10)
            composition_matrix[c, :] = counts

    fig, ax = plt.subplots(figsize=(14, 6))
    
    bottom = np.zeros(k)
    x = np.arange(k)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        bars = ax.bar(x, composition_matrix[:, i], bottom=bottom, label=class_names[i], color=colors[i])
        bottom += composition_matrix[:, i]
        
    ax.set_title(f'Homogeneidad y Tamaño de Clusters: {title}', fontsize=14)
    ax.set_xlabel('ID del Cluster (Algoritmo No Supervisado)')
    ax.set_ylabel('Cantidad de Muestras')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C_{c}\n(N={int(cluster_counts[c])})' for c in x])
    ax.legend(title='Clase Real', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
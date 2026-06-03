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

def GMM(X, n_clusters, n_inits=5, max_iters=100, tol=1e-4, random_seed=1973):
    """
    Implementación de GMM usando el algoritmo Expectation-Maximization
    """
    best_log_likelihood = -np.inf
    best_mu_k, best_sigma_k, best_pi_k, best_labels = None, None, None, None
    n_samples, n_features = X.shape

    for init in range(n_inits):
        np.random.seed(random_seed + init)  # Cambiar la semilla en cada inicializacion para obtener resultados diferentes
        mu_k = X[np.random.choice(n_samples, n_clusters, replace=False)]
        emp_cov = np.cov(X, rowvar=False) + np.eye(n_features) * 1e-6
        sigma_k = np.array([emp_cov for _ in range(n_clusters)])
        pi_k = np.ones(n_clusters) / n_clusters
        prev_log_likelihood = -np.inf

        for i in range(max_iters):
            # Expectation step
            log_resp = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                # Usamos nuestra propia función matemática
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
    Calcula el Silhouette Score global utilizando únicamente NumPy.
    Implementación optimizada con matriz de distancias pairwise.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    
    # Truco algebraico para calcular la matriz de distancias súper rápido
    X_sq = np.sum(X**2, axis=1)
    # clip(..., 0) evita raíces cuadradas de números microscópicamente negativos por error de flotante
    dist_matrix = np.sqrt(np.clip(X_sq[:, np.newaxis] + X_sq - 2 * np.dot(X, X.T), 0, None))
    
    s_i = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Distancias desde el punto 'i' a todos los demás
        dists_i = dist_matrix[i]
        
        # --- Calcular a(i): Distancia intra-cluster ---
        same_cluster_mask = (labels == cluster_i)
        same_cluster_mask[i] = False # Nos excluimos a nosotros mismos
        
        # Si el cluster tiene un solo punto, por convención el silhouette es 0
        if np.sum(same_cluster_mask) == 0:
            s_i[i] = 0.0
            continue
            
        a_i = np.mean(dists_i[same_cluster_mask])
        
        # --- Calcular b(i): Distancia al cluster vecino más cercano ---
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == cluster_i:
                continue
                
            other_cluster_mask = (labels == other_cluster)
            avg_dist_to_other = np.mean(dists_i[other_cluster_mask])
            
            if avg_dist_to_other < b_i:
                b_i = avg_dist_to_other
                
        # --- Calcular Silhouette para el punto 'i' ---
        s_i[i] = (b_i - a_i) / max(a_i, b_i)
        
    # El score global es el promedio de todos los puntos
    return np.mean(s_i)

import numpy as np

def calculate_H_P(D, beta):
    """Función auxiliar para calcular Entropía y Probabilidades en alta dimensionalidad (Estabilizada)."""
    P = np.exp(-D * beta)
    
    # 1. Agregamos un número pequeñísimo para evitar que P sea todo ceros
    P = np.maximum(P, 1e-12) 
    
    sumP = np.sum(P)
    
    # 2. Agregamos el mismo número a la suma para que el logaritmo no explote
    sumP = np.maximum(sumP, 1e-12) 
    
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    
    return H, P / sumP

def compute_pairwise_distances(X):
    """Calcula la matriz de distancias cuadradas euclidianas."""
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return np.maximum(D, 0) # Evitar negativos microscópicos por error de flotante

def get_perplexity_and_p(D, perplexity=30.0, tol=1e-5):
    """Búsqueda binaria para encontrar la varianza (sigma) de cada punto que alcance la perplejidad deseada."""
    n = D.shape[0]
    P = np.zeros((n, n))
    beta = np.ones((n, 1)) # beta = 1 / (2 * sigma^2)
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        
        # Calcular P y Entropía (H) inicial
        H, thisP = calculate_H_P(Di, beta[i])
        
        # Búsqueda binaria
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
    Reducimos iteraciones a 300 para que termine en un tiempo razonable para el TP.
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
    
    # 1. Probabilidades en alta dimensión (Distribución Normal)
    print("t-SNE: Calculando afinidades P (esto puede tardar...)")
    D = compute_pairwise_distances(X)
    P = get_perplexity_and_p(D, perplexity)
    
    # Simetrizar y exagerar (Early Exaggeration para mejorar la separación)
    P = P + P.T
    P = P / np.sum(P)
    P = P * 4.0 
    P = np.maximum(P, 1e-12)
    
    print(f"t-SNE: Iniciando Descenso de Gradiente ({max_iter} iteraciones)...")
    for iter in range(max_iter):
        # 2. Probabilidades en baja dimensión (Distribución t-Student)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # 3. Gradiente
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            
        # 4. Actualización con Momento (Momentum trick)
        momentum = initial_momentum if iter < 20 else final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, 0) # Centrar
        
        # Detener la exageración temprana
        if iter == 100:
            P = P / 4.0
            
        if (iter + 1) % 50 == 0:
            # Calcular la divergencia KL para mostrar progreso
            C = np.sum(P * np.log(P / Q))
            print(f"Iteración {iter+1}: Error (KL Divergence) = {C:.4f}")
            
    return Y
import numpy as np
import matplotlib.pyplot as plt

def delta_training(X, Z, eta=0.01, K=100, w_range=(-1, 1)):
    M, N = X.shape
    w = np.random.uniform(w_range[0], w_range[1], N)
    errors = []

    for k in range(K):
        total_error = 0
        for mu in range(M):
            y = np.dot(w, X[mu])           
            e = Z[mu] - y                  
            w = w + eta * e * X[mu]        
            total_error += e**2
        errors.append(total_error / M) 
    return w, errors

def run_experiment(N, M, eta=0.01, K=100, repeats=3):
    print(f"\n=== PRZYPADEK: N={N}, M={M} ===")
    
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (M, N))
    Z = np.random.uniform(-1, 1, M)

    all_weights = []
    plt.figure(figsize=(7,4))
    
    for i in range(repeats):
        np.random.seed(i)
        w, errors = delta_training(X, Z, eta=eta, K=K)
        all_weights.append(w)
        plt.plot(errors, label=f'Powtórzenie {i+1}')
    
    plt.title(f'Ewolucja błędu - przypadek N={N}, M={M}')
    plt.xlabel('Epoka')
    plt.ylabel('Średni błąd (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"wykres_N{N}_M{M}.png")
    plt.close()

    print("Wagi końcowe z różnych uruchomień:")
    for i, w in enumerate(all_weights):
        print(f"  W{i+1}: {np.round(w, 4)}")
    
    y_pred = X @ all_weights[-1]
    mse = np.mean((Z - y_pred)**2)
    print(f"Średni błąd końcowy (MSE): {mse:.6f}")


if __name__ == "__main__":
    run_experiment(N=2, M=6, eta=0.01, K=10000, repeats=5)

    run_experiment(N=4, M=4, eta=0.01, K=10000, repeats=5)

    run_experiment(N=6, M=3, eta=0.01, K=10000, repeats=5)

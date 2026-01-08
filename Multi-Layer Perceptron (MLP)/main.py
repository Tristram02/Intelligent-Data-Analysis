import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def sigmoid_prime(s):
    y = sigmoid(s)
    return y * (1 - y)

X = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]], dtype=float)
Z = X.copy()

class SimpleMLP:
    def __init__(self, eta=0.5, with_bias=True, w_init_range=(-0.5,0.5)):
        self.eta = eta
        self.with_bias = with_bias
        
        self.W1 = np.random.uniform(w_init_range[0], w_init_range[1], size=(2,4))
        
        out_cols = 3 if with_bias else 2
        self.W2 = np.random.uniform(w_init_range[0], w_init_range[1], size=(4,out_cols))

    def forward(self, x):
        h = self.W1.dot(x)
        if self.with_bias:
            h_with_bias = np.concatenate([h, [1.0]])
        else:
            h_with_bias = h.copy()
        s = self.W2.dot(h_with_bias)
        y = sigmoid(s)
        return h, h_with_bias, s, y

    def train_epoch(self, X, Z):
        epoch_error = 0.0
        for x, z in zip(X, Z):
            h, h_with_bias, s, y = self.forward(x)
            delta_out = sigmoid_prime(s) * (z - y)

            if self.with_bias:
                W2_no_bias = self.W2[:, :2] 
            else:
                W2_no_bias = self.W2 
           
            delta_hidden = (W2_no_bias.T.dot(delta_out))
            
            self.W2 += self.eta * np.outer(delta_out, h_with_bias)
            self.W1 += self.eta * np.outer(delta_hidden, x)
            epoch_error += 0.5 * np.sum((z - y)**2)
        return epoch_error

    def train(self, X, Z, max_epochs=10000, tol=1e-9, verbose=False):
        errors = []
        for epoch in range(1, max_epochs+1):
            E = self.train_epoch(X, Z)
            errors.append(E)
            if verbose and (epoch % 500 == 0 or E < tol):
                print(f"Epoch {epoch}, error={E:.8f}")
            if E < tol:
                break
        return errors

eta = 0.5
max_epochs = 10000

mlp_with_bias = SimpleMLP(eta=eta, with_bias=True, w_init_range=(-0.5,0.5))
errors_with = mlp_with_bias.train(X, Z, max_epochs=max_epochs, tol=1e-9, verbose=True)

mlp_without_bias = SimpleMLP(eta=eta, with_bias=False, w_init_range=(-0.5,0.5))
errors_without = mlp_without_bias.train(X, Z, max_epochs=max_epochs, tol=1e-9, verbose=True)

def evaluate_model(mlp):
    outputs = []
    for x in X:
        _, _, s, y = mlp.forward(x)
        outputs.append(y)
    return np.array(outputs)

out_with = evaluate_model(mlp_with_bias)
out_without = evaluate_model(mlp_without_bias)

print("Final results (with bias):")
for i, (x, y) in enumerate(zip(X, out_with), start=1):
    print(f"Pattern {i} input {x} -> output (rounded) {np.round(y,4)}  actual {y}")

print("\nFinal results (without bias):")
for i, (x, y) in enumerate(zip(X, out_without), start=1):
    print(f"Pattern {i} input {x} -> output (rounded) {np.round(y,4)}  actual {y}")

plt.figure(figsize=(10,5))
plt.plot(errors_with, label="with bias")
plt.plot(errors_without, label="without bias")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Sum squared error (log scale)")
plt.title("Training error vs epoch (log scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_errors.png')
print("Saved training error plot to mlp_errors.png")

df = pd.DataFrame({
    'pattern': ['p1','p2','p3','p4'],
    'input': [str(list(x.astype(int))) for x in X],
    'out_with_bias': [np.round(y,4).tolist() for y in out_with],
    'out_without_bias': [np.round(y,4).tolist() for y in out_without]
})
df.to_csv('mlp_final_outputs.csv', index=False)


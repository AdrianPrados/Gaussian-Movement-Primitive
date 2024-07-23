import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

N = 1001

np.random.seed(0)
tf.random.set_seed(0)

# Build inputs X
X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

# Deterministic functions in place of latent ones
f1 = np.sin
f2 = np.cos

# Use transform = exp to ensure positive-only scale values
transform = np.exp

# Compute loc and scale as functions of input X
loc = f2(X)
scale = transform(f2(2*X))

# Sample outputs Y from Gaussian Likelihood
Y = np.random.normal(loc, scale)

def plot_distribution(X, Y, loc, scale):
    plt.figure(figsize=(15, 5))
    x = X.squeeze()

    # Compute variance matrix
    variance_matrix = np.diag(scale.squeeze()**2)
    print(variance_matrix)

    # Define lower and upper bounds based on variance
    lb = loc.squeeze() - 2*np.sqrt(variance_matrix.diagonal())
    ub = loc.squeeze() + 2*np.sqrt(variance_matrix.diagonal())
    print("Valor de lb: ",lb)

    # Calcular la media y la desviación estándar de los datos dentro del área de interés
    mean_within_limits = np.mean(Y[(X.squeeze() >= lb) & (X.squeeze() <= ub)])
    std_within_limits = np.std(Y[(X.squeeze() >= lb) & (X.squeeze() <= ub)])
    # Print mean and standard deviation
    print("Mean within limits:", mean_within_limits)
    print("Standard deviation within limits:", std_within_limits)

    # Crear la distribución normal utilizando la media y la desviación estándar calculadas
    normal_distribution = tfp.distributions.Normal(loc=mean_within_limits, scale=std_within_limits)

    # Generar muestras de la distribución normal para visualizar la zona probabilística
    samples = normal_distribution.sample(1000)




    # Plot shaded region representing variance
    plt.fill_between(x, lb, ub, color="silver", alpha=0.3)

    # Plot mean line
    plt.plot(X, loc, color="black")

    # Plot data points
    plt.scatter(X, Y, color="blue", alpha=0.8)

    plt.show()
    plt.close()

    # Visualizar la zona probabilística
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
    plt.xlabel('Valores')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Zona Probabilística')
    plt.show()

    # Calculate covariance matrix within the area of interest
    indices_within_limits = np.where((X.squeeze() >= lb) & (X.squeeze() <= ub))
    Y_within_limits = Y[indices_within_limits]
    print(len(Y))
    covariance_matrix_within_limits = np.cov(Y_within_limits, rowvar=False)

    # Print covariance matrix
    print("\nCovariance matrix within limits:\n", covariance_matrix_within_limits)

    

plot_distribution(X, Y, loc, scale)

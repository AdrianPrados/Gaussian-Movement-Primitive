'''
Probabilitic Gaussian Process Movement Primitives
devoleped by Adrian Prados
'''
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import time
import matplotlib.pyplot as plt


np.random.seed(3)


class ProGpMp:
    def __init__(self, X, y, X_, y_, dim, demos, size, observation_noise=0.1):
        '''
        :param X: Original input set
        :param y: Original output set
        :param X_: Via-points input set
        :param y_: Via-points output set
        :param observation_noise: Observation noise for y
        '''
        if dim == 1:
            print("1D Case")
            self.X_total = np.vstack((X, X_)) #np.vstack((X.reshape(-1, 1), X_.reshape(-1, 1))).reshape(-1)#np.vstack((X, X_))
            self.y_total = np.vstack((y.reshape(-1, 1), y_.reshape(-1, 1))).reshape(-1)
            self.X = X
            self.y = y
            self.X_ = X_
            self.y_ = y_
            self.input_dim = np.shape(self.X_total)[1]
            self.input_num = np.shape(self.X_total)[0]
            self.via_points_num = np.shape(X_)[0]
            self.observation_noise = observation_noise
            self.num_gpmp = None
            self.gpmp_list=[]
            self.demos=demos
            self.size=size
            # Initialize the parameters
            self.param = self.init_random_param() #? Initialize with random values
            
        else:
            print(str(dim)+"D case")
            self.ProGP=[0]*dim
            for i in range(dim):
                print("Valor de i: ",i)
                print(y[:,i])
                self.ProGP[i] = ProGpMp(X,y[:,i],X_,y_[:,i],dim=1,demos=demos,size=size,observation_noise=observation_noise)
                self.ProGP[i].train()
            #self.BlendGP=BlendedGpMp(self.ProGP)
    
    def init_random_param(self):
        '''
        Initialize the hyper-parameters of GP-MP
        :return: Initial hyper-parameters
        '''
        kern_length_scale = 0.1 * np.random.normal(size=self.input_dim) + 1
        kern_noise = 1 * np.random.normal(size=1)
        return np.hstack((kern_noise, kern_length_scale))
    
    def variance_matrix_definition(self,visualize=False):
        x=self.X.reshape(self.demos,self.size)
        y=self.y.reshape(self.demos,self.size)

        upper_envelope = np.max(y, axis=0)
        lower_envelope = np.min(y, axis=0)
        # Añadimos un margen
        margin = 0.5
        upper_envelope += margin
        lower_envelope -= margin

        # Points inside the enveloped (filtering outliers)
        inside_envelope_indices = np.all((y >= lower_envelope) & (y <= upper_envelope), axis=0)
        inside_envelope_data = y[:, inside_envelope_indices]

        # mean and standar deviation
        mean_inside_envelope = np.mean(inside_envelope_data, axis=0)
        std_inside_envelope = np.std(inside_envelope_data, axis=0)
        print("Estandar desviation: ",std_inside_envelope)
        diference = abs(upper_envelope - lower_envelope)*std_inside_envelope

        if visualize:
            upper_bound = np.minimum(np.unique(mean_inside_envelope) + 5*np.unique(std_inside_envelope), upper_envelope)
            lower_bound = np.maximum(np.unique(mean_inside_envelope) - 5*np.unique(std_inside_envelope), lower_envelope)
            plt.figure(figsize=(10, 6))
            for i in range(y.shape[0]):
                plt.plot(x[0], y[i], label='Demostration '+str(i))
            plt.plot(x[0],mean_inside_envelope,'r-*', label='Mean')
            plt.plot(x[0], upper_envelope, 'r--', label='Upper envelope')
            plt.plot(x[0], lower_envelope, 'g--', label='Lower envelope')
            plt.fill_between(x[0], lower_envelope, upper_envelope, color='gray', alpha=0.3, label='Std Dev')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('Data')
            plt.title('Human demonstration')
            plt.grid(True)
            plt.show()


        # Create matrix with size (n, n)
        matrix_init= np.zeros((self.y.shape[0], self.y.shape[0]))
        print("Difference: ", diference)
        for i in range(self.y.shape[0]):
            matrix_init[i, i] = diference[i % y.shape[1]]
        
        # Create matrix with size using via-points
        matrix_vias = np.zeros((self.y_total.shape[0], self.y_total.shape[0]))
        matrix_vias[:self.y.shape[0], :self.y.shape[0]] = matrix_init

        # Rellenar los valores faltantes de la diagonal principal con 1
        for i in range(self.y.shape[0], self.y_total.shape[0]):
            matrix_vias[i, i] = 1


        return matrix_vias


    def build_objective(self, param):
        '''
        Compute the objective function (log pdf)
        :param param: Hyper-parameters of GP-MP
        :return: Value of the obj function
        '''
        cov_y_y_total = self.rbf(self.X_total, self.X_total, param)
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0 #! esta es la pinche matriz que hace que sea homocedastico :3
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise**2 * np.eye(self.input_num - self.via_points_num)
        #weigth = self.variance_matrix_definition()
        #print("Matriz de varianza: ",variance_matrix)
        cov_y_y_total = cov_y_y_total + variance_matrix
        out = - mvn.logpdf(self.y_total, np.zeros(self.input_num), cov_y_y_total)
        # Convertir los datos a tipo float
        """ dataPrueba = cov_y_y_total._value

        # Visualización de la matriz de covarianza
        plt.figure(figsize=(8, 6))
        plt.title('Matriz de Covarianza 2')
        plt.imshow(dataPrueba, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Covarianza')
        plt.xlabel('Índice de Observaciones')
        plt.ylabel('Índice de Observaciones')
        plt.show() """

        return out

    def train(self):
        def cons_f(param):
            '''
            Constrained function to ensure the positive semi-definite of the covariance matrix 
            :param param: Hyper-parameters of GP-MP
            :return: Value of the constrained function
            '''
            delta = 1e-10
            cov_y_y_ = self.rbf(self.X_, self.X_, param)
            min_eigen = np.min(np.linalg.eigvals(cov_y_y_))
            return min_eigen - delta

        # Using "trust-constr" approach to minimize the obj
        nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, np.inf, jac='2-point', hess=BFGS())
        result = minimize(value_and_grad(self.build_objective), self.param, method='trust-constr', jac=True,
                        options={'disp': True, 'maxiter': 50000, 'xtol': 1e-50, 'gtol': 1e-20},
                        constraints=[nonlinear_constraint], callback=self.callback)

        # Pre-computation for prediction
        self.param = result.x
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise ** 2 * np.eye(self.input_num - self.via_points_num)
        if self.demos!=1:
            weigths = self.variance_matrix_definition(visualize=False)
            self.cov_y_y_total = self.rbf(self.X_total, self.X_total, self.param) + weigths*variance_matrix
        else:
            self.cov_y_y_total = self.rbf(self.X_total, self.X_total, self.param) + variance_matrix
        self.beta = solve(self.cov_y_y_total, self.y_total)
        self.inv_cov_y_y_total = solve(self.cov_y_y_total, np.eye(self.input_num))

        dataPrueba = self.cov_y_y_total
        # Visualización de la matriz de covarianza
        plt.figure(figsize=(8, 6))
        plt.title('Prior covariance K(t*,t*)')
        plt.imshow(dataPrueba, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Covariance value')
        """ plt.xlabel('Values')
        plt.ylabel('Values') """
        plt.show()


    def rbf(self, x, x_, param):
        '''
        Interface to compute the Variance matrix (vector) of GP,
        :param x: Input 1
        :param x_: Input 2
        :param param: Hyper-parameters of GP-MP
        :return: Variance matrix (vector)
        '''
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input_1D(self, x):
        '''
        Compute the mean and variance functions of the posterior estimation of 1D ProGpMp
        :param x: Query inputs
        :return: Mean and variance functions
        '''
        """ print("----------------------")
        print(self.param)
        print("----------------------") """
        cov_y_f = self.rbf(self.X_total, x, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta.reshape((-1, 1)))
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y_total), cov_y_f))).reshape(-1, 1)
        return mean_outputs, var

    def callback(self, param, state):
        # ToDo: add something you want to know about the training process
        if state.nit % 100 == 0 or state.nit == 1:
            print('---------------------------------- Iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', state.fun)
            print('maximum constr_violation: ', state.constr_violation)

    def BlendedGpMp(self, gpmp_list):
        '''
        Blended of a list of GP
        :param gpmp_list: List with the trained GP
        '''
        self.num_gpmp = len(gpmp_list)
        self.gpmp_list = gpmp_list

    def predict_determined_input_Blended(self, inputs, alpha_list):
        '''
        Prediction for a list of inputs
        :param inputs: a (input_num, d_input) matrix
        :param alpha_list: a (num_gpmp, input_num) matrix
        :return: Mean and variance functions
        '''
        num_input = np.shape(inputs)[0]
        var = np.empty(num_input)
        mu = np.empty(num_input)
        print(inputs)
        print(self.num_gpmp)
        minimum_var = np.ones(num_input) * 1e-100
        for i in range(self.num_gpmp):
            print("Estado de i: ",i)
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input_1D(inputs)
            mu_i = mu_i.reshape(-1)
            var_i = var_i.reshape(-1)
            var_i = np.max([minimum_var, var_i], 0)
            var = var + alpha_list[i, :] / var_i
            mu = mu + alpha_list[i, :] / var_i * mu_i
        var = 1 / var
        mu = var * mu
        return mu, var

    def predict_single_determined_input(self, input, alpha_pair):
        '''
        Prediction for just one point as input
        :param input: Single input, a (d_input,) array
        :param alpha_pair: Values of alphas of GP-MPs, a (num_gpmp,) array
        :return: Mean and variance functions
        '''
        mu_list = np.empty(self.num_gpmp)
        var_list = np.empty(self.num_gpmp)
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input_1D(input.reshape(-1, 1))
            mu_list[i] = mu_i[0, 0]
            var_list[i] = var_i[0, 0]
        Matrix = np.empty((self.num_gpmp, self.num_gpmp - 1))
        for i in range(self.num_gpmp):
                Matrix[i, :] = np.delete(var_list, i, 0)
        temp = np.cumprod(Matrix, axis=1)[:, -1]
        den = np.sum(alpha_pair * temp)
        num_var = np.cumprod(var_list)[-1]
        var = num_var / den
        num_mu = np.sum(alpha_pair * temp * mu_list)
        mu = num_mu / den
        return mu, var
    
    def predict_BlendedPos(self, inputs):
        '''
        Prediction for a list of inputs in position (treated as individual GP in time)
        :param inputs: a (input_num, d_input) matrix
        :return: Mean and variance functions for n dimensions
        '''
        mu=[]
        var=[]
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input_1D(inputs)
            mu_i=mu_i.reshape(-1)
            var_i=var_i.reshape(-1)
            mu.append(mu_i)
            var.append(var_i)
        return mu, var
    

class BlendDifferentGaussians:
    def __init__(self, gpmp_list):
        '''
        :param gpmp_list: trained gpmp_list
        '''
        self.num_gpmp = len(gpmp_list)
        self.gpmp_list = gpmp_list

    def predict_blended_determined_input(self, inputs, alpha_list):
        '''
        :param inputs: a (input_num, d_input) matrix
        :param alpha_list: a (num_gpmp, input_num) matrix
        :return: Mean and variance functions
        '''
        num_input = np.shape(inputs)[0]
        var = np.empty(num_input)
        mu = np.empty(num_input)
        minimum_var = np.ones(num_input) * 1e-100
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input_1D(inputs)
            mu_i = mu_i.reshape(-1)
            var_i = var_i.reshape(-1)
            var_i = np.max([minimum_var, var_i], 0)
            var = var + alpha_list[i, :] / var_i
            mu = mu + alpha_list[i, :] / var_i * mu_i
        var = 1 / var
        mu = var * mu
        return mu, var

    def predict_single_blended_determined_input(self, input, alpha_pair):
        '''
        :param input: Single input, a (d_input,) array
        :param alpha_pair: Values of alphas of GP-MPs, a (num_gpmp,) array
        :return: Mean and variance functions
        '''
        mu_list = np.empty(self.num_gpmp)
        var_list = np.empty(self.num_gpmp)
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input_1D(input.reshape(-1, 1))
            mu_list[i] = mu_i[0, 0]
            var_list[i] = var_i[0, 0]
        Matrix = np.empty((self.num_gpmp, self.num_gpmp - 1))
        for i in range(self.num_gpmp):
                Matrix[i, :] = np.delete(var_list, i, 0)
        temp = np.cumprod(Matrix, axis=1)[:, -1]
        den = np.sum(alpha_pair * temp)
        num_var = np.cumprod(var_list)[-1]
        var = num_var / den
        num_mu = np.sum(alpha_pair * temp * mu_list)
        mu = num_mu / den
        return mu, var
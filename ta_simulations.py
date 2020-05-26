#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-------------------------------------------------
# Module containing the simulation functions for 
# the Homeworks solutions.
#-------------------------------------------------
## Author: Ezequias Júnior
## Version: 0.4.1 -> Homeworks: 09
## Email: ezequiasjunio@gmail.com
## Status: in development

# Imports
import numpy as np
import tensoralg # Module created for the hw.
from numpy.linalg import norm
from numpy.random import rand, randn
from scipy.io import loadmat
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import row
from bokeh.palettes import colorblind
from tqdm.notebook import tqdm, tnrange


# Utility functions:
def noise_std(snr, x, noise):
    # Calculates the noise standard dev. using equation 1
    if x.ndim > 2:
        return tensoralg.tensor_norm(x) / (np.sqrt(snr) *\
                                           tensoralg.tensor_norm(noise))
    else:
        return norm(x, 'fro') / (np.sqrt(snr) * norm(noise, 'fro'))


def apply_noise(snr_db, x):
    # Checking if x is complex and generating noise matrix:
    if np.iscomplexobj(x):
        noise = (randn(*x.shape) + 1j*randn(*x.shape))/np.sqrt(2)
    else: # real case:
        noise = randn(*x.shape)
    
    # SNR conversion:
    snr = 10**(.1*snr_db)
    std = noise_std(snr, x, noise)
    # Applying noise:
    return x + std*noise

def norm_mse(x, x_hat):
    # Calculates de Normalized Mean Square Error NMSE:
    if len(x.shape) > 2: #Tensor case
        return (tensoralg.tensor_norm(x-x_hat)/tensoralg.tensor_norm(x))**2
    else: # Matix case
        return (norm(x_hat - x, 'fro') / norm(x, 'fro'))**2

def plot_results(x, y, y2, label1, label2, method):
    # Plotting results for the homeworks by generating a graph 
    # of SNR versus NMSE.
    # Figure properties:
    plot_log = figure(tools="hover, pan, wheel_zoom, box_zoom, reset", 
                 plot_width=500, plot_height=400, 
                 background_fill_color="#fafafa",
                 x_axis_label='SNR [dB]',
                 y_axis_label='NMSE',
                 y_axis_type="log",
                 title=f'Normalized Error Curve - {method} - Log')
    # Curves: Log sacle
    plot_log.line(x, y, line_width=2, color='red', 
                                      legend=f'{method} - {label1}')
    plot_log.square(x, y, size=8, color='red', fill_color=None, 
                                           legend=f'{method} - {label1}')
    plot_log.line(x, y2, line_width=2, color='green', 
                                   legend=f'{method} - {label2}')
    plot_log.square(x, y2, size=8, color='green', fill_color=None, 
                                              legend=f'{method} - {label2}')
    plot_log.legend.location = "top_right"
    plot_log.legend.click_policy = "hide"
    # Linear scale:
    plot_lin = figure(tools="hover, pan, wheel_zoom, box_zoom, reset", 
                 plot_width=500, plot_height=400, 
                 background_fill_color="#fafafa",
                 x_axis_label='SNR [dB]',
                 y_axis_label='NMSE',
                 title=f'Normalized Error Curve - {method} - Linear')
    # Curves: Log sacle
    plot_lin.line(x, y, line_width=2, color='red', 
                                      legend=f'{method} - {label1}')
    plot_lin.square(x, y, size=8, color='red', fill_color=None, 
                                           legend=f'{method} - {label1}')
    plot_lin.line(x, y2, line_width=2, color='green', 
                                   legend=f'{method} - {label2}')
    plot_lin.square(x, y2, size=8, color='green', fill_color=None, 
                                              legend=f'{method} - {label2}')
    plot_lin.legend.location = "top_right"
    plot_lin.legend.click_policy = "hide"

    show(row(plot_lin, plot_log))
    pass

# Simulation functions: HOMEWORKS 03, 04
def run_simulation_lskr(snr_db, num_mc, param1, param2, ncol):
    # Storing the results:
    norm_square_error = np.zeros((num_mc, snr_db.size))
    # Monte Carlo Simulation:
    for realization in tnrange(num_mc):
        # Generating matrices:
        A = rand(param1, ncol).view(np.complex_)
        B = rand(param2, ncol).view(np.complex_)
        mt_x = tensoralg.kr(A, B)
        for ids, snr in enumerate(snr_db):
            # Applying noise to the matrix X_0:
            x_noise = apply_noise(snr, mt_x)
            # Estimating factor matrices:
            a_hat, b_hat = tensoralg.lskrf(x_noise, param1, param2)
            # Calculating the estimative of X_0:
            x_hat = tensoralg.kr(a_hat, b_hat)
            # Calculating the normalized error:
            norm_square_error[realization, ids] = norm_mse(mt_x, x_hat)
    # Returning the NMSE:
    return norm_square_error.mean(axis=0)

def run_simulation_lskron(snr_db, num_mc, param1, param2):
    # Storing the results:
    norm_square_error = np.zeros((num_mc, snr_db.size))
    # Monte Carlo Simulation:
    for realization in tnrange(num_mc):
        # Generating matrices:
        A = rand(param1[0], 2*param1[1]).view(np.complex_)
        B = rand(param2[0], 2*param2[1]).view(np.complex_)
        mt_x = tensoralg.kron(A, B)
        for ids, snr in enumerate(snr_db):
            # Applying noise to the matrix X_0:
            x_noise = apply_noise(snr, mt_x)
            # Estimating factor matrices:
            a_hat, b_hat = tensoralg.lskronf(x_noise, param1, param2)
            # Calculating the estimative of X_0:
            x_hat = tensoralg.kron(a_hat, b_hat)
            # Calculating the normalized error:
            norm_square_error[realization, ids] = norm_mse(mt_x, x_hat)
    # Returning the NMSE:
    return norm_square_error.mean(axis=0)
    
# Simulation functions: HOMEWORKS 06, 07 and 08
def all_orth(core):
    # All orthogonality check for a 3rd-Order core tensor.
    k, i, j = core.shape
    orth_k = []
    for a in range(k):
        for b in range(k):
            if a != b:
                orth_k.append(np.abs(np.trace(core[a, :, :].conj().T @\
                                              core[b, :, :])))
    orth_i = []
    for a in range(i):
        for b in range(i):
            if a != b:
                orth_i.append(np.abs(np.trace(core[:, a, :].conj().T @\
                                              core[:, b, :])))                     
    orth_j = []
    for a in range(j):
        for b in range(j):
            if a != b:
                orth_j.append(np.abs(np.trace(core[:, :, a].conj().T @\
                                              core[:, :, b])))

    print(f'Dimension I={i}: {np.round(orth_i)}')
    print(f'Dimension J={j}: {np.round(orth_j)}')
    print(f'Dimension K={k}: {np.round(orth_k)}')
    pass

def nth_singular_val(core):
    # Check the n-th singular values ordering property 
    # for a 3rd-Order core tensor.
    n_sing_values = []
    for n in range(len(core.shape)):
        sigma = np.zeros(core.shape[n])
        for i in range(core.shape[n]):
            if n == 0:
                sigma[i] = norm(core[i, :, :], 'fro')
            elif n == 1:
                sigma[i] = norm(core[:, i, :], 'fro')
            else:
                sigma[i] = norm(core[:, :, i], 'fro')
        n_sing_values.append(sigma)
    print(f'\nN-th singular values (I, J, K):\n{n_sing_values[1]}'+\
          f'\n{n_sing_values[2]}\n{n_sing_values[0]}')

def run_simulation_mlskr(snr_db, num_mc, nrows, ncol, flag=False):
    # Storing the results:
    norm_square_error = np.zeros((num_mc, snr_db.size))
    # Monte Carlo Simulation:
    for realization in tnrange(num_mc):
        # Generating matrices:
        mt_list = [rand(rows, ncol).view(np.complex_) for rows in nrows]
        # Matrix X_0
        mt_x = tensoralg.kr(*mt_list)
        for ids, snr in enumerate(snr_db):
            # Applying noise to the matrix X_0:
            x_noise = apply_noise(snr, mt_x)
            # Estimating factor matrices:
            if not flag: # Using HOSVD
                a_hat = tensoralg.mlskrf(x_noise, nrows)
            else: # Using HOOI
                a_hat = tensoralg.mlskrf(x_noise, nrows, flag)
            # Calculating the estimative of X_0:
            x_hat = tensoralg.kr(*a_hat)
            # Calculating the normalized error:
            norm_square_error[realization, ids] = norm_mse(mt_x, x_hat)
    # Returning the NMSE:
    return norm_square_error.mean(axis=0)

def run_simulation_lskron3d(snr_db, num_mc, shapes, flag=False):
    # Storing the results:
    norm_square_error = np.zeros((num_mc, snr_db.size))
    # Monte Carlo Simulation:
    for realization in tnrange(num_mc):
        # Generating matrices:
        mt_list = [rand(shape[0], shape[1]*2).view(np.complex_) 
                   for shape in shapes]
        # Matrix X_0
        mt_x = tensoralg.kron(*mt_list)
        for ids, snr in enumerate(snr_db):
            # Applying noise to the matrix X_0:
            x_noise = apply_noise(snr, mt_x)
            # Estimating factor matrices:
            if not flag: # Using HOSVD
                a_hat = tensoralg.lskronf_3d(x_noise, shapes)
            else: # Using HOOI
                a_hat = tensoralg.lskronf_3d(x_noise, shapes, flag)
            # Calculating the estimative of X_0:
            x_hat = tensoralg.kron(*a_hat)
            # Calculating the normalized error:
            norm_square_error[realization, ids] = norm_mse(mt_x, x_hat)
    # Returning the NMSE:
    return norm_square_error.mean(axis=0)

# Simulation functions: HOMEWORK 09
def run_simulation_als(snr_db, num_mc, rows, rank, tol, it):
    # Storing the results:
    norm_square_error_a = np.zeros((num_mc, snr_db.size))
    norm_square_error_b = np.zeros((num_mc, snr_db.size))
    norm_square_error_c = np.zeros((num_mc, snr_db.size))
    norm_square_error_x = np.zeros((num_mc, snr_db.size))
    # Monte Carlo Simulation:
    for realization in tnrange(num_mc):
        # Generating matrices:
        # mtx = [randn(i, rank*2).view(complex) for i in rows]
        a = randn(rows[0], rank*2).view(complex)
        b = randn(rows[1], rank*2).view(complex)
        c = randn(rows[2], rank*2).view(complex)
        mtx = [a, b, c]
        # Tensor X_0:
        tx = tensoralg.cpd_tensor(mtx)
        for ids, snr in enumerate(snr_db):
            # Applying noise to the tensor X_0:
            tx_noise = apply_noise(snr, tx)
            # Estimating factor matrices:            
            mtx_hat = tensoralg.cp_decomp(tx_noise, rank, 
                                          eps=tol, num_iter=it)
            # Constructing tensor X_hat:
            tx_hat = tensoralg.cpd_tensor(mtx_hat)
            # Calculating the normalized error:
            norm_square_error_x[realization, ids] = norm_mse(tx,tx_hat)
            norm_square_error_a[realization, ids] = norm_mse(mtx[0],mtx_hat[0])
            norm_square_error_b[realization, ids] = norm_mse(mtx[1],mtx_hat[1])
            norm_square_error_c[realization, ids] = norm_mse(mtx[2],mtx_hat[2])
    # Retruning results:
    results = [norm_square_error_a.mean(axis=0), 
               norm_square_error_b.mean(axis=0), 
               norm_square_error_c.mean(axis=0), 
               norm_square_error_x.mean(axis=0)]
    
    return results 

def plot_error(als_error):
    # Plotting results for the homework 9 by generating a graph 
    # of number of iteractions versus the ALS error.    
    colors = colorblind['Colorblind'][7]
    plot_log = figure(tools="hover, pan, wheel_zoom, box_zoom, reset", 
                 plot_width=500, plot_height=400,
                 background_fill_color="#fafafa",
                 x_axis_label='Iteractions',
                 y_axis_label='Error',
                 y_axis_type='log',
                 title='Error Curve - ALS')
    x = np.arange(als_error.size) + 1
    y = als_error.ravel()
    plot_log.line(x, y, line_width=2, color=colors[0], 
                                      legend='Error - ALS')

    plot_log.legend.location = "top_right"
    show(plot_log)
    pass

def plot_results_als(snr, results, method, labels):
    # Plotting results for the homework 9 by generating a graph 
    # of SNR versus NMSE.
    # Figure properties:
    colors = colorblind['Colorblind'][len(results)]
    plot_log = figure(tools="hover, pan, wheel_zoom, box_zoom, reset", 
                 plot_width=500, plot_height=400,
                 background_fill_color="#fafafa",
                 x_axis_label='SNR [dB]',
                 y_axis_label='NMSE',
                 y_axis_type='log',
                 title=f'Normalized Error Curves - {method} - Log')
    # Curves: Log sacle
    for i, result in enumerate(results):
        plot_log.line(snr, result, line_width=2, color=colors[i], 
                                      legend=f'{method} - {labels[i]}')
        plot_log.square(snr, result, size=8, color=colors[i], fill_color=None, 
                                           legend=f'{method} - {labels[i]}')
    plot_log.legend.location = "top_right"
    plot_log.legend.click_policy = "hide"
    # Linear scale:
    plot_lin = figure(tools="hover, pan, wheel_zoom, box_zoom, reset", 
                 plot_width=500, plot_height=400, 
                 background_fill_color="#fafafa",
                 x_axis_label='SNR [dB]',
                 y_axis_label='NMSE',
                 title=f'Normalized Error Curves - {method} - Linear')
    # Curves: Log sacle
    for i, result in enumerate(results):
        plot_lin.line(snr, result, line_width=2, color=colors[i], 
                                      legend=f'{method} - {labels[i]}')
        plot_lin.square(snr, result, size=8, color=colors[i], fill_color=None, 
                                           legend=f'{method} - {labels[i]}')
    plot_lin.legend.location = "top_right"
    plot_lin.legend.click_policy = "hide"

    show(row(plot_lin, plot_log))
    pass

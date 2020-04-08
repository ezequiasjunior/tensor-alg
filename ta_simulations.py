#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-------------------------------------------------
# Module containing the simulation functions for 
# the Homeworks solutions.
#-------------------------------------------------
## Author: Ezequias Júnior
## Version: 0.1.1 -> Homeworks 03 and 04 complex case
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
from tqdm.notebook import tqdm, tnrange


# Utility functions:
def noise_std(snr, x, noise):
    # Calculates the noise standard dev. using equation 1
    return norm(x, 'fro') / (np.sqrt(snr) * norm(noise, 'fro'))

def apply_noise(snr_db, x):
    # Checking if x is complex and generating noise matrix:
    if np.iscomplexobj(x):
        noise = randn(x.shape[0], x.shape[1]*2).view(np.complex_)/np.sqrt(2)
    else: # real case:
        noise = randn(*x.shape)
    
    # SNR conversion:
    snr = 10**(.1*snr_db)
    std = noise_std(snr, x, noise)
    # Applying noise:
    return x + std*noise

def norm_mse(x, x_hat):
    # Calculates de Normalized Mean Square Error NMSE:
    return (norm(x_hat - x, 'fro') / norm(x, 'fro'))**2

def plot_results(x, y, y2, label1, label2, method):
    # Plotting results for homework 03 and 04 by generating a graph 
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

# Simulation functions:
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
    

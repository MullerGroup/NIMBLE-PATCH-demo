#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import gc
import scipy.stats as st

def process_results_files():
    
    plot_options = {
            'excess_refresh_rate': [1, 2, 4, 8, 16, 64],
            'per_frame_target_counts' : [1, 4, 16, 64, 256],
            'slm_formats' : [32,64,128,256,512], # pixels, SLM sizes. Each entry represents a square SLM with NxN pixels
            'cgh_gs_pix_samplings': [1,3],
            'single_shot_analysis_en': False,
            'refresh_rate_analysis_en': False,
            'plot_accuracy_en': True,
            'plot_efficiency_en': True,
            'plot_contrast_en': True,
            'plot_speckle_contrast_en': True,
            'plot_computation_time_en': True,
            'single_algorithm_plots_en': True,
            'ratio_plots_en': True,
            'save_en': True,
            'individual_run_datapoints_en': True,
            'rra_constant_target_size_plots_en': False,
            'rra_constant_target_count_plots_en': True,
            'Optica_Figure_4_en': True,
            'Optica_Figure_S2_en': True,
            'Optica_Figure_7_en': True,
            'random_target_depth_range': 0.75,
            'random_target_lateral_range': 0.9,
            }
    
    resultdir = 'results'
    
    slm_colors = sns.color_palette("colorblind",as_cmap=True)
    slm_colors[0], slm_colors[1] = slm_colors[1], slm_colors[0]

    As_index = 0
    Ad_index = 1
    Gs_index = 2
    Gd_index = 3
    Go_index = 4
    
    average_accuracy_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_accuracy_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_accuracy_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_accuracy_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_accuracy_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    average_contrast_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_contrast_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_contrast_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_contrast_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_contrast_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    average_efficiency_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_efficiency_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_efficiency_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_efficiency_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_efficiency_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    average_speckle_contrast_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_speckle_contrast_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_speckle_contrast_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_speckle_contrast_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_speckle_contrast_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    average_computation_time_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])

    individual_accuracy_analytical_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_accuracy_analytical_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_accuracy_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_accuracy_GS_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_accuracy_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    
    individual_contrast_analytical_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_contrast_analytical_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_contrast_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_contrast_GS_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_contrast_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    
    individual_efficiency_analytical_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_efficiency_analytical_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_efficiency_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_efficiency_GS_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_efficiency_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    
    individual_speckle_contrast_analytical_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_speckle_contrast_analytical_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_speckle_contrast_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_speckle_contrast_GS_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_speckle_contrast_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    
    individual_computation_time_analytical_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_analytical_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_GS_decomposed = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
        
    filecount_analytical_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_analytical_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_GS_decomposed = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_contrast_ratio_analytical_singleshot_vs_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_ratio_analytical_singleshot_vs_GS_singleshot = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    average_accuracy_ratio_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_contrast_ratio_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_efficiency_ratio_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_ratio_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_analytical_singleshot_vs_GS_singleshot = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    
    individual_contrast_ratio_analytical_decomposed_vs_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    individual_computation_time_ratio_analytical_decomposed_vs_GS_oversampled = np.empty((np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])) + (0, )).tolist()
    average_contrast_ratio_analytical_decomposed_vs_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    average_computation_time_ratio_analytical_decomposed_vs_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
    filecount_analytical_decomposed_vs_GS_oversampled = np.zeros([np.size(plot_options['slm_formats']),np.size(plot_options['per_frame_target_counts']),np.size(plot_options['excess_refresh_rate']),np.size(plot_options['cgh_gs_pix_samplings'])])
        
    for filename in os.listdir(resultdir):
        
        if filename == ".DS_Store":
            continue
        
        filepath = os.path.join(resultdir, filename)
        
        if os.path.isdir(filepath):
            continue
        results = np.load(filepath,allow_pickle=True).item()
        
        results_accuracy_list = results['accuracy_list']
        results_contrast_list = results['contrast_list']
        results_efficiency_list = results['efficiency_list']
        results_speckle_contrast_list = results['speckle_contrast_list']
        results_computation_time_list = results['computation_time_list']
        
        try:
            results_system = results['system']
            results_targets = results['targets']
        except:
            continue
        
        if results_system['cgh_GS_offset'] != 0:
            continue
        
        # if results_system['cgh_gs_pix_sampling'] != plot_options['cgh_gs_pix_sampling']:
        #     continue
        
        if results_targets['random_target_depth_range'] != plot_options['random_target_depth_range']:
            continue
        
        if results_targets['random_target_lateral_range'] != plot_options['random_target_lateral_range']:
            continue
        
        cgh_gs_pix_sampling = results_system['cgh_gs_pix_sampling']
        try:
            glb_cgh_gs_pix_sampling_index = plot_options['cgh_gs_pix_samplings'].index(cgh_gs_pix_sampling)
        except:
            continue
        
        for results_slm_format_index, slm_format in enumerate(results['slm_formats']):
            
            try:
                glb_slm_format_index = plot_options['slm_formats'].index(slm_format)
            except:
                continue
            
            for results_target_count_index, target_count in enumerate(results['target_counts']):

                for results_excess_refresh_rate_index, excess_refresh_rate in enumerate(results['excess_refresh_rate']):
                    
                    try:
                        glb_excess_refresh_rate_index = plot_options['excess_refresh_rate'].index(excess_refresh_rate)
                    except:
                        continue
                    
                    try:
                        glb_per_frame_target_count_index = plot_options['per_frame_target_counts'].index(int(target_count/excess_refresh_rate))
                    except:
                        continue
                    
                    if 'As' in results['algorithm_list']:
                        average_accuracy_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_accuracy_list[As_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_efficiency_list[As_index][results_slm_format_index,results_target_count_index]
                        average_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_contrast_list[As_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[As_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_computation_time_list[As_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_analytical_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[As_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_analytical_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[As_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_analytical_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[As_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_analytical_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[As_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_analytical_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[As_index][results_slm_format_index,results_target_count_index])
                        filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1

                    if 'Gs' in results['algorithm_list']:
                        average_accuracy_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_accuracy_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_efficiency_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_contrast_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_computation_time_list[Gs_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[Gs_index][results_slm_format_index,results_target_count_index])
                        filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1
                    
                    if 'Ad' in results['algorithm_list']:
                        average_accuracy_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_accuracy_list[Ad_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_efficiency_list[Ad_index][results_slm_format_index,results_target_count_index]
                        average_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_contrast_list[Ad_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[Ad_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_computation_time_list[Ad_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_analytical_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[Ad_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_analytical_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[Ad_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_analytical_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[Ad_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_analytical_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[Ad_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_analytical_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[Ad_index][results_slm_format_index,results_target_count_index])
                        filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1

                    if 'Gd' in results['algorithm_list']:
                        average_accuracy_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_accuracy_list[Gd_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_efficiency_list[Gd_index][results_slm_format_index,results_target_count_index]
                        average_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_contrast_list[Gd_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[Gd_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_computation_time_list[Gd_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_GS_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[Gd_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_GS_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[Gd_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_GS_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[Gd_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_GS_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[Gd_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_GS_decomposed[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[Gd_index][results_slm_format_index,results_target_count_index])
                        filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1

                    if 'Go' in results['algorithm_list']:
                        average_accuracy_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_accuracy_list[Go_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_efficiency_list[Go_index][results_slm_format_index,results_target_count_index]
                        average_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_contrast_list[Go_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[Go_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += results_computation_time_list[Go_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[Go_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[Go_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[Go_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[Go_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[Go_index][results_slm_format_index,results_target_count_index])
                        filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1

                    if 'As' in results['algorithm_list'] and 'Gs' in results['algorithm_list']:
                        average_accuracy_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_accuracy_list[As_index][results_slm_format_index,results_target_count_index]/results_accuracy_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_contrast_list[As_index][results_slm_format_index,results_target_count_index]/results_contrast_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_efficiency_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_efficiency_list[As_index][results_slm_format_index,results_target_count_index]/results_efficiency_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_speckle_contrast_list[As_index][results_slm_format_index,results_target_count_index]/results_speckle_contrast_list[Gs_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_computation_time_list[As_index][results_slm_format_index,results_target_count_index]/results_computation_time_list[Gs_index][results_slm_format_index,results_target_count_index]
                        individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_accuracy_list[As_index][results_slm_format_index,results_target_count_index]/results_accuracy_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[As_index][results_slm_format_index,results_target_count_index]/results_contrast_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_efficiency_list[As_index][results_slm_format_index,results_target_count_index]/results_efficiency_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_speckle_contrast_list[As_index][results_slm_format_index,results_target_count_index]/results_speckle_contrast_list[Gs_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[As_index][results_slm_format_index,results_target_count_index]/results_computation_time_list[Gs_index][results_slm_format_index,results_target_count_index])
                        filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1
                        
                    if 'Ad' in results['algorithm_list'] and 'Go' in results['algorithm_list']:
                        average_contrast_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_contrast_list[Ad_index][results_slm_format_index,results_target_count_index]/results_contrast_list[Go_index][results_slm_format_index,results_target_count_index]
                        average_computation_time_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index] += results_computation_time_list[Ad_index][results_slm_format_index,results_target_count_index]/results_computation_time_list[Go_index][results_slm_format_index,results_target_count_index]
                        individual_contrast_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_contrast_list[Ad_index][results_slm_format_index,results_target_count_index]/results_contrast_list[Go_index][results_slm_format_index,results_target_count_index])
                        individual_computation_time_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index][glb_per_frame_target_count_index][glb_excess_refresh_rate_index][glb_cgh_gs_pix_sampling_index].append(results_computation_time_list[Ad_index][results_slm_format_index,results_target_count_index]/results_computation_time_list[Go_index][results_slm_format_index,results_target_count_index])
                        filecount_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] += 1
                        
    for glb_cgh_gs_pix_sampling_index, cgh_gs_pix_sampling in enumerate(plot_options['cgh_gs_pix_samplings']):
        for glb_slm_format_index, slm_format in enumerate(plot_options['slm_formats']):
            for glb_per_frame_target_count_index, per_frame_target_count in enumerate(plot_options['per_frame_target_counts']):
                for glb_excess_refresh_rate_index, excess_refresh_rate in enumerate(plot_options['excess_refresh_rate']):
                    
                    if filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:    
                        average_accuracy_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                    
                    if filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:
                        average_accuracy_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
    
                    if filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:    
                        average_accuracy_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                    
                    if filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:
                        average_accuracy_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_decomposed[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
    
                    if filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:
                        average_accuracy_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
    
                    if filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:
                        average_accuracy_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_accuracy_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_efficiency_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_efficiency_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_speckle_contrast_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_ratio_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_singleshot_vs_GS_singleshot[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
    
                    if filecount_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] != 0:
                        average_contrast_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_contrast_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]
                        average_computation_time_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index] = average_computation_time_ratio_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]/filecount_analytical_decomposed_vs_GS_oversampled[glb_slm_format_index,glb_per_frame_target_count_index,glb_excess_refresh_rate_index,glb_cgh_gs_pix_sampling_index]

    slm_formats = plot_options['slm_formats']
    per_frame_target_counts = plot_options['per_frame_target_counts']
    excess_refresh_rates = plot_options['excess_refresh_rate']
    result_root_dir = "processed_results/"
    if not os.path.isdir(result_root_dir):
        os.makedirs(result_root_dir)

    def optica_figure4_add_plot_vs_targets(data, plt, ax, plt_linestyle, ylabel, ratio_plot = False,fillstyle = 'full', x_offset_mult = 1, excess_refresh_rate_index=0, cgh_gs_pix_sampling_index=0, mew=1, markersize=10, marker='o', hline=False, loglog=True, hline_value=1, errorbars_en=False, confidence_en=False):
        
        for slm_format_index, slm_format in enumerate(slm_formats):
            
            data_avg_plot = []
            data_std_plot = []
            per_frame_target_count_plot = []
            ci_lower = []
            ci_upper = []
            
            for per_frame_target_count_index, per_frame_target_count in enumerate(per_frame_target_counts):
            
                data_sliced = np.squeeze(np.array(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                
                if np.any(data_sliced) is False:
                    continue
                
                else:
                    data_avg_plot.append(np.mean(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                    if errorbars_en: 
                        data_std_plot.append(np.std(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                    else: 
                        data_std_plot.append(0)
                    
                    if confidence_en:
                        data_t = data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]
                        ci_temp = st.t.interval(alpha=0.95, df=len(data_t)-1, loc=np.mean(data_t), scale=st.sem(data_t)) 
                        ci_lower.append(ci_temp[0].item())
                        ci_upper.append(ci_temp[1].item())
                    per_frame_target_count_plot.append(per_frame_target_count*x_offset_mult)
            
            if errorbars_en:
                plt.errorbar(per_frame_target_count_plot,np.squeeze(np.array(data_avg_plot)),yerr=np.squeeze(np.array(data_std_plot)), fmt=marker, linestyle = plt_linestyle, capsize=6)
            else:
                plt.plot(per_frame_target_count_plot,np.squeeze(np.array(data_avg_plot)), fillstyle=fillstyle, marker=marker, mew=mew, markersize=markersize, linestyle = plt_linestyle)
            plt.xticks(per_frame_target_count_plot)
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
            if confidence_en:
                data_avg_plot = np.array(data_avg_plot)
                ci_lower = np.array(ci_lower)
                ci_upper = np.array(ci_upper)
                per_frame_target_count_plot = np.array(per_frame_target_count_plot)
                ax.fill_between(per_frame_target_count_plot, (ci_lower), (ci_upper), alpha=.1, color=slm_colors[slm_format_index])
                
    if plot_options['Optica_Figure_4_en']:
        
        single_fig_width = 7
        single_fig_height = 10
        
        ratio_fig_width = 7
        ratio_fig_height = 6
        
        default_cycler = cycler(color=slm_colors)
        
        NP_linestyle = '-'
        GSx1_linestyle = 'dotted'
        GSx3_linestyle = (0, (3, 1, 1, 1, 1, 1))
        
        NP_marker = 'o'
        GSx1_marker = '^'
        GSx3_marker = 's'
        
        NP_offset = 1
        GSx1_offset = 1
        GSx3_offset = 1
        
        NP_markersize = 8
        GSx1_markersize = 8
        GSx3_markersize = 8
        
        NP_mew = 2.5
        GSx1_mew = 2.5
        GSx3_mew = 2.5
        
        NP_fillstyle = 'none'
        GSx1_fillstyle = 'none'
        GSx3_fillstyle = 'none'
          
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(single_fig_width,single_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt_linestyle = NP_linestyle
        mpl.rc('lines', linewidth=2)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_contrast_analytical_singleshot, plt, ax, plt_linestyle,mew=NP_mew,fillstyle=NP_fillstyle,x_offset_mult = NP_offset, marker = NP_marker, markersize=NP_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx1_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_contrast_GS_singleshot, plt, ax, plt_linestyle,mew=GSx1_mew,fillstyle=GSx1_fillstyle, x_offset_mult = GSx1_offset, marker = GSx1_marker, markersize=GSx1_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx3_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        mpl.rc('lines', linewidth=3)
        optica_figure4_add_plot_vs_targets(individual_contrast_GS_singleshot, plt, ax, plt_linestyle,mew=GSx3_mew,fillstyle=GSx3_fillstyle, x_offset_mult = GSx3_offset, cgh_gs_pix_sampling_index=1, marker = GSx3_marker, markersize=GSx3_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt.xlabel("Target Count")
        plt.ylabel("Contrast")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_RawContrast.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(single_fig_width,single_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt_linestyle = '-'
        mpl.rc('lines', linewidth=2)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_computation_time_analytical_singleshot, plt, ax, plt_linestyle,mew=NP_mew,fillstyle=NP_fillstyle,x_offset_mult = NP_offset, marker = NP_marker, markersize=NP_markersize, ylabel = "Computation Time")
        plt_linestyle = 'dotted'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_computation_time_GS_singleshot, plt, ax, plt_linestyle,mew=GSx1_mew,fillstyle=GSx1_fillstyle,x_offset_mult = GSx1_offset, marker = GSx1_marker, markersize=GSx1_markersize , ylabel = "Computation Time")
        plt_linestyle = (0, (3, 1, 1, 1, 1, 1))
        mpl.rc('lines', linewidth=3)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_computation_time_GS_singleshot, plt, ax, plt_linestyle, cgh_gs_pix_sampling_index=1,mew=GSx3_mew,fillstyle=GSx3_fillstyle,x_offset_mult = GSx3_offset, marker = GSx3_marker, markersize=GSx3_markersize, ylabel = "Computation Time")
        plt.xlabel("Target Count")
        plt.ylabel("Computation Time (s)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_Computation.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_contrast_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Contrast Ratio (NP / GSx1)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_ContrastRatio_X1.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_contrast_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3, ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Contrast Ratio (NP / GSx3)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_ContrastRatio_X3.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
            
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(np.array(individual_computation_time_ratio_analytical_singleshot_vs_GS_singleshot), plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time Ratio",ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Computation Time Ratio (NP / GSx1)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_ComputationRatio_X1.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
           
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(np.array(individual_computation_time_ratio_analytical_singleshot_vs_GS_singleshot), plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3,ylabel = "Computation Time Ratio",ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Computation Time Ratio (NP / GSx3)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_4_ComputationRatio_X3.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')

    def optica_figure7_add_plot_vs_refresh_rate_constant_target_count(data, plt, ax, plt_linestyle, ylabel, total_target_count, fillstyle = 'full', mew=1, markersize=10, marker='o',cgh_gs_pix_sampling_index=0, hline=False, loglog=True, hline_value=1, errorbars_en=False, confidence_en=False):
        
        for slm_format_index, slm_format in enumerate(slm_formats):
            
            if (slm_format == 64 or slm_format == 128 or slm_format == 256) is False:
                continue
            
            data_avg_plot = []
            data_std_plot = []
            excess_refresh_rate_plot = []
            ci_lower = []
            ci_upper = []
            
            for per_frame_target_count_index, per_frame_target_count in enumerate(per_frame_target_counts):
                            
                for excess_refresh_rate_index, excess_refresh_rate in enumerate(excess_refresh_rates):
                    if per_frame_target_count*excess_refresh_rate != total_target_count:
                        continue
                    
                    data_sliced = np.squeeze(np.array(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                    
                    if np.any(data_sliced) is False:
                        continue
                
                    else:
                        data_avg_plot.append(np.mean(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                        if errorbars_en:
                            data_std_plot.append(np.std(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                        else:
                            data_std_plot.append(0)
                            
                        if confidence_en:
                            data_t = data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]
                            ci_temp = st.t.interval(alpha=0.95, df=len(data_t)-1, loc=np.mean(data_t), scale=st.sem(data_t)) 
                            ci_lower.append(ci_temp[0].item())
                            ci_upper.append(ci_temp[1].item())

                        excess_refresh_rate_plot.append(excess_refresh_rate)
            if errorbars_en:
                plt.errorbar(excess_refresh_rate_plot,np.squeeze(np.array(data_avg_plot)),yerr=np.squeeze(np.array(data_std_plot)), fmt='o', linestyle = plt_linestyle, capsize=6)
            else:
                plt.plot(excess_refresh_rate_plot,np.squeeze(np.array(data_avg_plot)),fillstyle=fillstyle, marker=marker, mew=mew, markersize=markersize, linestyle = plt_linestyle, color=slm_colors[slm_format_index])
            
            plt.xticks(excess_refresh_rate_plot)
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
            if confidence_en:
                data_avg_plot = np.array(data_avg_plot)
                ci_lower = np.array(ci_lower)
                ci_upper = np.array(ci_upper)
                excess_refresh_rate_plot = np.array(excess_refresh_rate_plot)
                ax.fill_between(excess_refresh_rate_plot, (ci_lower), (ci_upper), alpha=.1, color=slm_colors[slm_format_index])
            
    def optica_figure7_add_plot_vs_refresh_rate_constant_target_size(data, plt, ax, plt_linestyle, ylabel, per_frame_target_count_index,fillstyle = 'full', mew=1, markersize=10, marker='o', cgh_gs_pix_sampling_index=0, hline=False, loglog=True, hline_value=1, errorbars_en=False, confidence_en=False):
        
        for slm_format_index, slm_format in enumerate(slm_formats):
            
            if (slm_format == 64 or slm_format == 128 or slm_format == 256) is False:
                continue
            
            data_avg_plot = []
            data_std_plot = []
            excess_refresh_rate_plot = []
            ci_lower = []
            ci_upper = []
                            
            for excess_refresh_rate_index, excess_refresh_rate in enumerate(excess_refresh_rates):
                
                data_sliced = np.squeeze(np.array(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                
                if np.any(data_sliced) is False:
                    continue
            
                else:
                    data_avg_plot.append(np.mean(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                    if errorbars_en:
                        data_std_plot.append(np.std(data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]))
                    else:
                        data_std_plot.append(0)
                        
                    if confidence_en:
                        data_t = data[slm_format_index][per_frame_target_count_index][excess_refresh_rate_index][cgh_gs_pix_sampling_index]
                        ci_temp = st.t.interval(alpha=0.95, df=len(data_t)-1, loc=np.mean(data_t), scale=st.sem(data_t)) 
                        ci_lower.append(ci_temp[0].item())
                        ci_upper.append(ci_temp[1].item())

                    excess_refresh_rate_plot.append(excess_refresh_rate)
            if errorbars_en:
                plt.errorbar(excess_refresh_rate_plot,np.squeeze(np.array(data_avg_plot)),yerr=np.squeeze(np.array(data_std_plot)), fmt='o', linestyle = plt_linestyle, capsize=6)
            else:
                plt.plot(excess_refresh_rate_plot,np.squeeze(np.array(data_avg_plot)),fillstyle=fillstyle, marker=marker, mew=mew, markersize=markersize, linestyle = plt_linestyle, color=slm_colors[slm_format_index])
            
            plt.xticks(excess_refresh_rate_plot)
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
            if confidence_en:
                data_avg_plot = np.array(data_avg_plot)
                ci_lower = np.array(ci_lower)
                ci_upper = np.array(ci_upper)
                excess_refresh_rate_plot = np.array(excess_refresh_rate_plot)
                ax.fill_between(excess_refresh_rate_plot, (ci_lower), (ci_upper), alpha=.1, color=slm_colors[slm_format_index])
    
    
    
    if plot_options['Optica_Figure_S2_en']:
        
        single_fig_width = 7
        single_fig_height = 10
        
        ratio_fig_width = 7
        ratio_fig_height = 6
        
        default_cycler = cycler(color=slm_colors)
        
        NP_linestyle = '-'
        GSx1_linestyle = 'dotted'
        GSx3_linestyle = (0, (3, 1, 1, 1, 1, 1))
        
        NP_marker = 'o'
        GSx1_marker = '^'
        GSx3_marker = 's'
        
        NP_offset = 1
        GSx1_offset = 1
        GSx3_offset = 1
        
        NP_markersize = 8
        GSx1_markersize = 8
        GSx3_markersize = 8
        
        NP_mew = 2.5
        GSx1_mew = 2.5
        GSx3_mew = 2.5
        
        NP_fillstyle = 'none'
        GSx1_fillstyle = 'none'
        GSx3_fillstyle = 'none'
        
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(single_fig_width,single_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = NP_linestyle
        mpl.rc('lines', linewidth=2)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_accuracy_analytical_singleshot, plt, ax, plt_linestyle,mew=NP_mew,fillstyle=NP_fillstyle,x_offset_mult = NP_offset, marker = NP_marker, markersize=NP_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx1_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_accuracy_GS_singleshot, plt, ax, plt_linestyle,mew=GSx1_mew,fillstyle=GSx1_fillstyle, x_offset_mult = GSx1_offset, marker = GSx1_marker, markersize=GSx1_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx3_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        mpl.rc('lines', linewidth=3)
        optica_figure4_add_plot_vs_targets(individual_accuracy_GS_singleshot, plt, ax, plt_linestyle,mew=GSx3_mew,fillstyle=GSx3_fillstyle, x_offset_mult = GSx3_offset, cgh_gs_pix_sampling_index=1, marker = GSx3_marker, markersize=GSx3_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt.xlabel("Target Count")
        plt.ylabel("Accuracy")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_RawAccuracy.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        # optica_figure4_add_plot_vs_targets(individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3, ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Accuracy Ratio (NP / GSx1)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_AccuracyRatio_X1.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
            
        #slm_colors = sns.color_palette("colorblind",as_cmap=True)
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        # optica_figure4_add_plot_vs_targets(individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_accuracy_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3, ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Accuracy Ratio (NP / GSx3)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_AccuracyRatio_X3.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
            
        #slm_colors = sns.color_palette("colorblind",as_cmap=True)
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(single_fig_width,single_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt_linestyle = NP_linestyle
        mpl.rc('lines', linewidth=2)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_efficiency_analytical_singleshot, plt, ax, plt_linestyle,mew=NP_mew,fillstyle=NP_fillstyle,x_offset_mult = NP_offset, marker = NP_marker, markersize=NP_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx1_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_efficiency_GS_singleshot, plt, ax, plt_linestyle,mew=GSx1_mew,fillstyle=GSx1_fillstyle, x_offset_mult = GSx1_offset, marker = GSx1_marker, markersize=GSx1_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt_linestyle = GSx3_linestyle
        plt.gca().set_prop_cycle(default_cycler)
        mpl.rc('lines', linewidth=3)
        optica_figure4_add_plot_vs_targets(individual_efficiency_GS_singleshot, plt, ax, plt_linestyle,mew=GSx3_mew,fillstyle=GSx3_fillstyle, x_offset_mult = GSx3_offset, cgh_gs_pix_sampling_index=1, marker = GSx3_marker, markersize=GSx3_markersize, ylabel = "Raw Contrast", confidence_en=True)
        plt.xlabel("Target Count")
        plt.ylabel("Efficiency")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_RawEfficiency.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
        #slm_colors = sns.color_palette("colorblind",as_cmap=True)
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        # optica_figure4_add_plot_vs_targets(individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3, ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Efficiency Ratio (NP / GSx1)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_EfficiencyRatio_X1.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
            
        #slm_colors = sns.color_palette("colorblind",as_cmap=True)
        default_cycler = cycler(color=slm_colors)
        excess_refresh_rate = 1
        plt.figure(figsize=(ratio_fig_width,ratio_fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        # optica_figure4_add_plot_vs_targets(individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D', ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure4_add_plot_vs_targets(individual_efficiency_ratio_analytical_singleshot_vs_GS_singleshot, plt, ax, plt_linestyle, markersize=8, marker = 'D',cgh_gs_pix_sampling_index=1, mew=3, ylabel = "Computation Time", ratio_plot = True, confidence_en=True)
        plt.axhline(y = 1, color = 'black', linestyle = '-')
        plt.xlabel("Target Count")
        plt.ylabel("Efficiency Ratio (NP / GSx3)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_S2_EfficiencyRatio_X3.png")
            plt.tight_layout()
            plt.savefig(result_path,bbox_inches='tight')
        
    if plot_options['Optica_Figure_7_en']:
        
        fig_width = 7
        fig_height = 6
        
        default_cycler = cycler(color=slm_colors)
        
        NP_linestyle = '-'
        Gs_linestyle = 'dashed'
        Gd_linestyle = 'dotted'
        Go_linestyle = '-.'
        
        NP_marker = 'o'
        Gd_marker = 'o'
        Go_marker = 'o'
        
        NP_markersize = 6
        Gd_markersize = 6
        Go_markersize = 6
        
        NP_mew = 2.5
        Gd_mew = 2.5
        Go_mew = 2.5
        
        NP_fillstyle = 'full'
        Gd_fillstyle = 'full'
        Go_fillstyle = 'full'
        
        total_target_count = 64
        
        default_cycler = cycler(color=slm_colors)
        
        excess_refresh_rate = 1
        plt.figure(figsize=(fig_width,fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        mpl.rc('lines', linewidth=2)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_contrast_analytical_decomposed, plt, ax, ylabel = "Raw Contrast", plt_linestyle=NP_linestyle, marker=NP_marker, fillstyle=NP_fillstyle, markersize=NP_markersize, mew=NP_mew, total_target_count=total_target_count, errorbars_en=False, confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_contrast_GS_singleshot, plt, ax, ylabel = "Raw Contrast", plt_linestyle=Gs_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew,total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_contrast_GS_decomposed, plt, ax, ylabel = "Raw Contrast", plt_linestyle=Gd_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew,total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.xlabel("Excess Refresh Rate")
        plt.ylabel("Contrast")
        ax.set_ylim([4e0,2e4])
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_7_RawContrast_NTarg_"+str(total_target_count)+".png")
            plt.savefig(result_path,bbox_inches='tight')
            
        excess_refresh_rate = 1
        plt.figure(figsize=(fig_width,fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        mpl.rc('lines', linewidth=2)
        plt_linestyle = '-'
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_accuracy_analytical_decomposed, plt, ax, ylabel = "Raw Contrast", plt_linestyle=NP_linestyle, marker=NP_marker, fillstyle=NP_fillstyle, markersize=NP_markersize, mew=NP_mew, total_target_count=total_target_count, errorbars_en=False, confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_accuracy_GS_singleshot, plt, ax, ylabel = "Raw Contrast", plt_linestyle=Gs_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew,total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_accuracy_GS_decomposed, plt, ax, ylabel = "Raw Contrast", plt_linestyle=Gd_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew,total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_accuracy_GS_oversampled, plt, ax, ylabel = "Raw Contrast", plt_linestyle=Go_linestyle, marker=Go_marker, fillstyle=Go_fillstyle, markersize=Go_markersize, mew=Go_mew,total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.xlabel("Excess Refresh Rate")
        plt.ylabel("Raw Accuracy")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_7_RawAccuracy_NTarg_"+str(total_target_count)+".png")
            plt.savefig(result_path,bbox_inches='tight')
            
        default_cycler = cycler(color=slm_colors)

        excess_refresh_rate = 1
        plt.figure(figsize=(fig_width,fig_height))
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_computation_time_analytical_decomposed, plt, ax, plt_linestyle=NP_linestyle, marker=NP_marker, fillstyle=NP_fillstyle, markersize=NP_markersize, mew=NP_mew, ylabel = "Computation Time", total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_computation_time_GS_singleshot, plt, ax, plt_linestyle=Gs_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew, ylabel = "Computation Time", total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.gca().set_prop_cycle(default_cycler)
        optica_figure7_add_plot_vs_refresh_rate_constant_target_count(individual_computation_time_GS_decomposed, plt, ax, plt_linestyle=Gd_linestyle, marker=Gd_marker, fillstyle=Gd_fillstyle, markersize=Gd_markersize, mew=Gd_mew, ylabel = "Computation Time", total_target_count=total_target_count, errorbars_en=False ,confidence_en=True)
        plt.xlabel("Excess Refresh Rate")
        plt.ylabel("Computation Time (s)")
        if plot_options['save_en']:
            result_path = os.path.join(result_root_dir,"Optica_Figure_7_Computation_NTarg_"+str(total_target_count)+".png")
            plt.savefig(result_path,bbox_inches='tight')
            
if __name__ == '__main__':

    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

    mpl.rc('font', **font)
    mpl.rc('lines', linewidth=3)
    
    process_results_files()

    gc.collect()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
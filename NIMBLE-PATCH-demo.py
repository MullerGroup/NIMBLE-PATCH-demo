#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from collections import deque
import cv2
import numpy as np
from skimage.draw import disk
from matplotlib import pyplot as plt
import metrics
import time
import gc
from scipy.optimize import linear_sum_assignment

def compute_target_size(system,targets,slm_format,size_target_set):
    num_blocks = int(np.size(targets,0))
    num_blocks_1D = int(np.sqrt(num_blocks))
    # NOTE: changed to scale with number of targets per frame
    size_diff_limit = system['cgh_nx'] / slm_format /2 * num_blocks_1D
    
    if size_target_set > size_diff_limit:
        size_target = size_target_set
    else:
        size_target = int(np.ceil(size_diff_limit))
        if system['verbose']:
            print('INFO: Target radius replaced from %i to %i for SLM format of %i to meet diffraction limit.' % (size_target_set, size_target,slm_format))
            
    return size_target

def generate_random_target_locations(system,targets):
    
    Nx = system['cgh_nx']
    Ny = system['cgh_ny']
    
    tic_tgen = time.perf_counter()
    
    print("Generating Randomized Targets...")
    
    # if system['decompose_en']:
    #     target_size = compute_target_size(system,targets, min(system['slm_formats'],targets['size']))
    # else:
    #     target_size = compute_target_size(system,targets, min(system['slm_formats'],targets['size'])) * system['excess_refresh_rate']
    
    if targets['random_locations']:
        if system['decompose_en']:
            num_blocks=int(targets['random_target_count']/system['excess_refresh_rate'])
        else:
            num_blocks = targets['random_target_count']
    else:
        if system['decompose_en']:
            num_blocks=int(np.size(targets['locations'],0)/system['excess_refresh_rate'])
        else:
            num_blocks = np.size(targets['locations'],axis=0)
    
    # if system['decompose_en']:
    #     num_blocks = int(np.size(targets['locations'],0)/system['excess_refresh_rate'])
    # else:
    #     num_blocks = np.size(targets['locations'],0)
    
    num_blocks_1D = int(np.sqrt(num_blocks))
    # NOTE: changed to scale with number of targets per frame
    size_diff_limit = system['cgh_nx'] / min(system['slm_formats']) /2 * num_blocks_1D
    # size_diff_limit = system['cgh_nx']/ min(system['slm_formats']) /2
    if targets['size'] > size_diff_limit:
        target_size = targets['size']
    else:
        target_size = int(np.ceil(size_diff_limit))
        
    target_depth_count = targets['random_target_depth_count']
    range_axial = system['range_axial']
    target_depth_max = targets['random_target_depth_range'] * range_axial
    target_depth_min = -target_depth_max
    
    target_lateral_range = targets['random_target_lateral_range'] * system['cgh_nx'] / 2
    
    target_depths = np.linspace(target_depth_min,target_depth_max,target_depth_count)
    
    existing_target_locations = deque()
    depth_tally = np.zeros(np.size(target_depths))
    
    def _add_new_target(depth_tally,existing_target_locations):
        
        def _target_loc_valid_check(new_target_depth_index,new_target_x,new_target_y):
            
            toc_tgen = time.perf_counter()
            tgen_compute_time = toc_tgen - tic_tgen
            if tgen_compute_time > 15:
                raise Exception("Targets Not Generatable")
            
            #if depth_tally[new_target_depth_index] != min(depth_tally):
                #return False
            
            if targets['experimental_setup_conditionals_en']:
                
                if system['cgh_nx'] - new_target_x <= (target_size*2+1) or new_target_x <= (target_size*2+1):
                    return False
                
                if system['cgh_ny'] - new_target_y <= (target_size*2+1) or new_target_y <= (target_size*2+1):
                    return False
                
                for existing_target in existing_target_locations:
                    
                    if existing_target[0] != target_depths[new_target_depth_index]:
                        continue
                    
                    delta_x = new_target_x - existing_target[1]
                    delta_y = new_target_y - existing_target[2]
                    
                    delta = np.sqrt( (delta_x**2) + (delta_y**2) )
                    
                    if delta < (target_size*10+1):
                        return False
                    
                delta_x_from_origin = abs(new_target_x - (system['cgh_nx']/2))
                delta_y_from_origin = abs(new_target_x - (system['cgh_nx']/2))
                
                if (delta_x_from_origin < targets['experimental_setup_keepout_x_range']*system['cgh_nx']) and (delta_y_from_origin < targets['experimental_setup_keepout_y_range']*system['cgh_nx']):
                    return False
                
            else:
            
                if system['cgh_nx'] - new_target_x <= (target_size*2+1) or new_target_x <= (target_size*2+1):
                    return False
                
                if system['cgh_ny'] - new_target_y <= (target_size*2+1) or new_target_y <= (target_size*2+1):
                    return False
                
                for existing_target in existing_target_locations:
                    
                    if existing_target[0] != target_depths[new_target_depth_index]:
                        continue
                    
                    delta_x = new_target_x - existing_target[1]
                    delta_y = new_target_y - existing_target[2]
                    
                    delta = np.sqrt( (delta_x**2) + (delta_y**2) )
                    
                    if delta < (target_size+1):
                        return False
                
            return True
        
        if np.any(existing_target_locations):
            new_target_location_valid = False
            while new_target_location_valid is False:
                new_target_depth_index = np.random.randint(0,np.size(target_depths))
                new_target_x = int(np.ceil((np.random.random() - 0.5)*2*target_lateral_range + (Nx/2))) 
                new_target_y = int(np.ceil((np.random.random() - 0.5)*2*target_lateral_range + (Ny/2)))
                new_target_location_valid = _target_loc_valid_check(new_target_depth_index,new_target_x,new_target_y)
            
        else:
            new_target_location_valid = False
            while new_target_location_valid is False:
                new_target_depth_index = np.random.randint(0,np.size(target_depths))
                new_target_x = int(np.ceil((np.random.random() - 0.5)*2*target_lateral_range + (Nx/2))) 
                new_target_y = int(np.ceil((np.random.random() - 0.5)*2*target_lateral_range + (Ny/2)))
                new_target_location_valid = _target_loc_valid_check(new_target_depth_index,new_target_x,new_target_y)

        new_target_depth = target_depths[new_target_depth_index]
        new_target_location = [new_target_depth,new_target_x,new_target_y]
        depth_tally[new_target_depth_index] += 1
            
        return new_target_location
    
    # Set the random seed for pseudo-random target generation to be (mostly) consistent across X1 and X3 cgh_gs_pix_sampling
    if(targets['use_seed']):
        np.random.seed(targets['random_target_seed'])
    else:   
        np.random.seed(None)
    
    for target_index in range(targets['random_target_count']):
        
        this_target_location = _add_new_target(depth_tally,existing_target_locations)
        existing_target_locations.append(this_target_location)
        
    print("Random target generation done.")
    return np.array(existing_target_locations)

def generate_target_plane_intensities(system,decomposed_targets,slm_format,size_target):
    
    Nx = system['cgh_nx']
    Ny = system['cgh_ny']
    
    depths_unique = np.unique(decomposed_targets[:,0])
    Nz = depths_unique.size
    
    target_plane_intensities = np.zeros([Nz,Nx,Ny])
    target_plane_depths = depths_unique

    
    #num_blocks = np.size(decomposed_targets,0)
    #num_blocks_1D = int(np.sqrt(num_blocks))
    #size_diff_limit = Nx / slm_format /2 * num_blocks_1D
    
    # if size_targets > size_diff_limit:
    #     size_target = size_targets
    # else:
    #     size_target = int(np.ceil(size_diff_limit))
    #     if system['verbose']:
    #         print('NOTE: Target radius replaced from %i to %i for SLM format of %i to meet diffraction limit.' % (size_targets, size_target,slm_format))

    
    for decomposed_target in decomposed_targets:
        target_loc_x = decomposed_target[1]
        target_loc_y = decomposed_target[2]
        target_depth = decomposed_target[0]
        target_depth_index = np.where(depths_unique == target_depth)[0][0]
        
        rr,cc = disk([target_loc_y,target_loc_x],size_target)
        target_plane_intensities[target_depth_index][rr,cc] = 1
    
    return target_plane_intensities, target_plane_depths


def decompose_and_assign_targets(system,targets,current_slm_format,num_frames,num_blocks):
    
    num_targets = num_frames * num_blocks
    num_blocks_1D = int(np.sqrt(num_blocks))
    size_block = int(current_slm_format/num_blocks_1D)
    
    # Generate target coordinate matrix
    coordinates_targets = np.ones((3,num_targets))
    i = 0
    #for target in targets['locations']:
    for target in targets:
        coordinates_targets[0,i] =  target[0]
        coordinates_targets[1,i] =  target[2]
        coordinates_targets[2,i] =  target[1]
        i = i+1
            
    # Calculate shift centers for each partitition block
    centers_1D = np.array(range(int(size_block/2), int(current_slm_format), int(size_block))) - current_slm_format/2
    centers_shift = np.ones((2,num_blocks))
    centers_shift[0,:] = np.tile(centers_1D, num_blocks_1D)
    centers_shift[1,:] = np.repeat(centers_1D, num_blocks_1D, axis=0)
        
    # Get locations for all targets
    centers_1D_xtile = np.zeros((num_blocks_1D, 1))
    centers_1D_ytile = np.zeros((1, num_blocks_1D))
    centers_1D_xtile[:,0] = centers_1D[:]
    centers_1D_ytile[0,:] = centers_1D[:]
    centers_1D_xtile = np.tile(centers_1D_xtile,(1,num_blocks_1D))
    centers_1D_ytile = np.tile(centers_1D_ytile,(num_blocks_1D,1))

    
    targets_zxy = np.transpose(coordinates_targets)
    phasestep_x = np.zeros((num_targets,num_blocks_1D,1))
    phasestep_y = np.zeros((num_targets,1,num_blocks_1D))
    # phasestep_xy = np.zeros((num_targets,num_blocks_1D, num_blocks_1D))
    
    correction_scale=  -2*targets_zxy[:,0] * system['slm_pix_size']/ system['optics_focal_length']/2/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size'])
    
    # Calculate phase steps from steering and correction along both X and Y for every target at each block
    for i in range(num_blocks_1D):
        
        correction_step = centers_1D[i] * correction_scale
        #phasestep_x[:,i,0] = 2 * np.pi * (-targets_zxy[:,0] * centers_1D[i] * system['slm_pix_size']/ system['optics_focal_length']/2/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size'])  +  (targets_zxy[:,2] - system['cgh_nx']/2) / system['cgh_nx'])
        #phasestep_y[:,0,i] = 2 * np.pi * (-targets_zxy[:,0] * centers_1D[i] * system['slm_pix_size']/ system['optics_focal_length']/2/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size'])  +  (targets_zxy[:,1] - system['cgh_nx']/2) / system['cgh_nx'])
        phasestep_x[:,i,0] = 2 * np.pi * (correction_step  +  (targets_zxy[:,2] - system['cgh_nx']/2) / system['cgh_nx'])
        phasestep_y[:,0,i] = 2 * np.pi * (correction_step  +  (targets_zxy[:,1] - system['cgh_nx']/2) / system['cgh_nx'])
    
    phasestep_x_tiled = np.tile(phasestep_x,(1,1,num_blocks_1D))
    phasestep_y_tiled = np.tile(phasestep_y,(1,num_blocks_1D,1))
    assignments = np.zeros((num_targets,6))
        
    # Modify zeros to allows for sinc^2 calculation
    zeros_x = phasestep_x_tiled ==0
    zeros_y = phasestep_y_tiled ==0
    phasestep_x_tiled[zeros_x]=0.001
    phasestep_y_tiled[zeros_y]=0.001
    
    # Sinc^2 calculation
    efficiency_x = np.square(np.divide(np.sin(phasestep_x_tiled/2),phasestep_x_tiled/2))
    efficiency_y = np.square(np.divide(np.sin(phasestep_y_tiled/2),phasestep_y_tiled/2))
    loss_xy = 1 - np.multiply(efficiency_x,efficiency_y)
    
    # Reformat array to solve linear sum assignment problem
    loss_xy_2D = np.reshape(loss_xy,(num_targets,num_blocks))
    loss_xy_2D = np.tile(loss_xy_2D,(1,num_frames))
    row_ind, col_ind = linear_sum_assignment(loss_xy_2D)
    row_ind_sorting = np.argsort(col_ind,axis=0)
    #row_ind_sorting = np.arange(num_targets)
    #np.random.shuffle(row_ind_sorting)
    col_ind_sorted = col_ind[row_ind_sorting]
    
    for i in range(num_targets):
        current_target_index = row_ind[row_ind_sorting[i]]
        #current_target_index = row_ind[i]
        block_indices = np.unravel_index(col_ind_sorted[i]%num_blocks, centers_1D_xtile.shape)
        block_center_x = centers_1D_xtile[block_indices]
        block_center_y = centers_1D_ytile[block_indices]
        # Each row is formatted as [X location, Y location, Z location, X shift, Y shift, loss]
        assignments[i,:] = [targets_zxy[current_target_index,1],targets_zxy[current_target_index,2],targets_zxy[current_target_index,0],block_center_x,block_center_y, loss_xy_2D[current_target_index, col_ind_sorted[i]]]            
        
    return assignments

def compute_hologram_globalGS(system,targets,slm_format,size_target,num_iter=50,initial_phase_mask='superposition',ovrsamp_idx=0):
    
    def _globalGS_compute_hologram_superposition(system,target_plane_intensities,Hs,slm_format):
        Nx = system['cgh_nx']
        Ny = system['cgh_ny']
        Nz = target_plane_intensities[:,1,1].size
        
        hologram = np.zeros([Nx,Ny])
        for depth_index in range(Nz):
            hologram_this_depth = np.fft.ifft2(np.fft.ifftshift(np.exp(1j * 2 * np.pi * np.random.rand(Nx,Ny)))) / Hs[depth_index]
            hologram = hologram + hologram_this_depth
            
        return np.angle(hologram)
    
    def _globalGS_iteration(system,target_plane_intensities,Hs,slm_format,hologram_phase,source):
        Nz = target_plane_intensities[:,1,1].size
        
        hologram_phase_resized = np.repeat(hologram_phase,system['cgh_gs_pix_sampling'],axis=1)
        hologram_phase_resized = np.repeat(hologram_phase_resized,system['cgh_gs_pix_sampling'],axis=0)
        hologram_slm = source * np.exp(1j * hologram_phase_resized)
        tempim = hologram_slm * 0
        for depth_index in range(Nz):
            imagez = np.fft.fftshift(np.fft.fft2(hologram_slm * Hs[depth_index]))
            target = target_plane_intensities[depth_index] + system['cgh_GS_offset']
            imagez = np.sqrt(target) * np.exp(1j * np.angle(imagez))
            tempim = tempim + np.fft.ifft2(np.fft.ifftshift(imagez))/Hs[depth_index]
            tempim_resized = tempim[::system['cgh_gs_pix_sampling'],::system['cgh_gs_pix_sampling']]
            
        return np.angle(tempim_resized)
    
    _system_GS = dict(system)
    _system_GS['simul_nx'] = system['cgh_gs_nx']
    _system_GS['simul_ny'] = system['cgh_gs_ny']
    _system_GS['simul_array_fold_reduction'] = 1
    _system_GS['simul_pix_sampling'] = system['cgh_gs_pix_sampling']
    
    # Generate the target planes from just the target center locations
    target_plane_intensities, target_plane_depths = generate_target_plane_intensities(_system_GS,targets,slm_format,size_target)
    target_plane_intensities_resized = np.zeros([np.size(target_plane_depths),system['cgh_gs_nx'],system['cgh_gs_ny']])
    for current_depth_index in range(np.size(target_plane_depths)):
        target_plane_intensities_resized[current_depth_index,:,:] = resize_and_zeropad_targets(_system_GS,target_plane_intensities[current_depth_index,:,:])
    
    if system['plots_en'] and system['debug_en']:
        for target_plane_intensity in target_plane_intensities:
            plt.figure()
            plt.imshow(target_plane_intensity,interpolation='none')
    
    # Separate the depths and the target plane intensities
    
    # Compute the H stacks that correspond to the target depths, index matched
    psX_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size']) / system['cgh_nx']
    psY_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size']) / system['cgh_ny']
    psX_hologram_gs = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['cgh_gs_pix_sampling']) / system['cgh_gs_nx']
    psY_hologram_gs = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['cgh_gs_pix_sampling']) / system['cgh_gs_ny']
    Hs_cgh = generate_H_stacks(system['cgh_nx'], system['cgh_ny'], target_plane_depths, system['optics_wavelength'], psX_hologram, psY_hologram)
    Hs_cgh_gs = generate_H_stacks(system['cgh_gs_nx'], system['cgh_gs_ny'], target_plane_depths, system['optics_wavelength'], psX_hologram_gs, psY_hologram_gs)
    
    
    
    # Initial phase mask computation for GS
    if initial_phase_mask == 'superposition':
        hologram_phase = _globalGS_compute_hologram_superposition(_system_GS,target_plane_intensities,Hs_cgh,slm_format)
    elif initial_phase_mask == 'nimblepatch':
        hologram_phase = generate_partitioned_hologram(system, targets, slm_format,size_target)[:,:,ovrsamp_idx]
        hologram_phase = resize_and_zeropad(_system_GS, slm_format, hologram_phase)
    else:
        raise Exception("GlobalGS: No such initial mask option")
    
    source_simul_full = (1/ system['cgh_gs_nx']) * (1/ system['cgh_gs_ny']) * np.ones([int(system['cgh_gs_nx']),int(system['cgh_gs_ny'])])
    slm_mask = np.ones([slm_format,slm_format])
    maskarray = resize_and_zeropad(_system_GS,slm_format,slm_mask)
    source_masked = np.multiply(source_simul_full, maskarray)
    
    for iter_index in range(num_iter):
        hologram_phase = _globalGS_iteration(_system_GS,target_plane_intensities_resized,Hs_cgh_gs,slm_format,hologram_phase,source_masked)
        # plt.figure()
        # plt.imshow(np.angle(hologram),interpolation='none')
    
    phase_slm = hologram_phase[int(system['cgh_nx']/2 - slm_format/2 ):int(system['cgh_nx']/2 + slm_format/2 ),int(system['cgh_ny']/2 - slm_format/2 ):int(system['cgh_ny']/2 + slm_format/2 )]
    
    return phase_slm

def compute_hologram_nimblepatch(system,slm_format,target_x_index,target_y_index,target_z,shift=[0, 0]):

    ux = system['slm_pix_size'] * np.array(range(1,slm_format+1))
    ux = ux - np.mean(ux)
    uy = system['slm_pix_size'] * np.array(range(1,slm_format+1))
    uy = uy - np.mean(uy)
    xx,yy = np.meshgrid(ux,uy,indexing='ij')
    
    phasex_calc = np.array(range(slm_format),ndmin=2) * 2 * np.pi * (target_x_index - system['cgh_nx']/2) / system['cgh_nx']
    phasex_calc = np.tile(phasex_calc,(slm_format,1))
    
    phasey_calc = np.array(range(slm_format),ndmin=2) * 2 * np.pi * (target_y_index - system['cgh_ny']/2) / system['cgh_ny']
    phasey_calc = np.transpose(phasey_calc)
    phasey_calc = np.tile(phasey_calc,(1,slm_format))
           
    phase_xcorrection_calc = np.array(range(slm_format),ndmin=2) * 2 * np.pi * (-target_z * shift[1] * system['slm_pix_size']/ system['optics_focal_length']/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size']))
    phase_xcorrection_calc = np.tile(phase_xcorrection_calc,(slm_format,1))
        
    phase_ycorrection_calc = np.array(range(slm_format),ndmin=2) * 2 * np.pi * (-target_z * shift[0] * system['slm_pix_size']/ system['optics_focal_length']/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size']))
    phase_ycorrection_calc = np.transpose(phase_ycorrection_calc)
    phase_ycorrection_calc = np.tile(phase_ycorrection_calc,(1,slm_format))
    
    if target_z != 0:
        R = -1 * (system['optics_focal_length'] ** 2) / target_z # Approximation
        #phasez_calc = 2 * np.pi * (R - R * np.sqrt(1 - (((xx**2) + (yy**2))/(R**2))))/system['optics_wavelength']
        phasez_calc = np.pi * ((xx**2) + (yy**2)) / system['optics_wavelength'] / R
        
    else:
        phasez_calc = xx - xx + 1
        
    phase_calc = (phasex_calc + phasey_calc + phasez_calc + phase_xcorrection_calc + phase_ycorrection_calc)
    phase_slm = phase_calc % (2*np.pi)
    
    return phase_slm

def crop_to_0th_order(system, image):
    size_0th = int(system['cgh_nx']/system['simul_array_fold_reduction'])
    corner_min = int((system['cgh_nx']/system['simul_array_fold_reduction']) * ((system['simul_pix_sampling']-1)/ 2))
    corner_max = int((system['cgh_nx']/system['simul_array_fold_reduction']) * ((system['simul_pix_sampling']-1)/ 2) + size_0th)
    return image[:,corner_min:corner_max,corner_min:corner_max]

def resize_and_zeropad(system,slm_format,hologram):
    pad_width = (system['cgh_nx']/system['simul_array_fold_reduction'] - slm_format)/2
    phase_padded = np.pad(hologram, int(pad_width), 'constant', constant_values=(0))
    phase_resized = np.repeat(phase_padded,system['simul_pix_sampling'],axis=1)
    phase_resized = np.repeat(phase_resized,system['simul_pix_sampling'],axis=0)
    return phase_resized

def resize_and_zeropad_targets(system,target_depth_intensity):
    new_size = int(system['cgh_nx']/system['simul_array_fold_reduction'])
    target_depth_intensity_shrunk = cv2.resize(target_depth_intensity,(new_size,new_size),interpolation=cv2.INTER_NEAREST)
    pad_width = (system['cgh_nx']/system['simul_array_fold_reduction']) * ((system['simul_pix_sampling']-1)/ 2)
    target_depth_intensity_shrunk_and_padded = np.pad(target_depth_intensity_shrunk, int(pad_width), 'constant', constant_values=(0))
    return target_depth_intensity_shrunk_and_padded

def discretize_phase_mask(system,phase_mask):
    
    Nx, Ny = np.shape(phase_mask)
    if system['verbose']:
        print('INFO: Discretizing phase mask')
    phase_step = 2*np.pi/(2**(system['slm_bit_depth']))
    phase_mask_discretized = phase_step * np.ceil(phase_mask/phase_step)
    if system['slm_noise_enabled']:
        if system['verbose']:
            print('INFO: Adding noise to phase mask')
            noise_mask = np.random.normal(0,system['slm_phase_noise_rms'],size=[Nx,Ny])
            phase_mask_discretized = phase_mask_discretized + noise_mask
        
    return phase_mask_discretized % (2*np.pi)
    
def generate_partitioned_hologram(system,targets,current_slm_format,size_target):
    
    # Check for target size
    num_blocks = int(np.size(targets,0))
    num_blocks_1D = int(np.sqrt(num_blocks))
    
    # Calculate number of targets (should be 1, 4, 9, 16...)
    num_blocks = np.size(targets,0)
    num_blocks_1D = int(np.sqrt(num_blocks))
    size_block = int(current_slm_format/num_blocks_1D)
    
    # Generate target coordinate matrix
    coordinates_targets = np.ones((3,num_blocks))
    i = 0
    for target in targets:
        coordinates_targets[0,i] =  target[0]
        coordinates_targets[1,i] =  target[2]
        coordinates_targets[2,i] =  target[1]
        i = i+1
    
    # Case of no partition
    if num_blocks == 1:
        partitioned_holograms = np.zeros((current_slm_format,current_slm_format,1))
        partitioned_holograms[:,:,0] = compute_hologram_nimblepatch(system,size_block, coordinates_targets[2,0], coordinates_targets[1,0], coordinates_targets[0,0])
        return partitioned_holograms
        
    
    # Calculate shift centers for each partitition block
    centers_1D = np.array(range(int(size_block/2), int(current_slm_format), int(size_block))) - current_slm_format/2
    centers_shift = np.ones((2,num_blocks))
    centers_shift[0,:] = np.tile(centers_1D, num_blocks_1D)
    centers_shift[1,:] = np.repeat(centers_1D, num_blocks_1D, axis=0)
    
        
    assignments = decompose_and_assign_targets(system,targets,current_slm_format,1,num_blocks)
        
    current_hologram_frame = np.zeros((current_slm_format,current_slm_format))
    partitioned_holograms = np.zeros((current_slm_format,current_slm_format,1))
    
    # Call compute_hologram_nimblepatch for each target-block pair
    for k in range(num_blocks):
        
        current_center = np.array(([assignments[k,4]],[assignments[k,3]]))
        current_hologram_block = compute_hologram_nimblepatch(system,size_block,assignments[k,1],assignments[k,0],assignments[k,2],(assignments[k,4],assignments[k,3]))
        
        #plt.figure()
        #plt.imshow(current_hologram_block,interpolation="none")
        
        # Add block to frame
        current_hologram_frame[int(current_slm_format/2+current_center[0]-size_block/2):int(current_slm_format/2+current_center[0]+size_block/2), int(current_slm_format/2+current_center[1]-size_block/2):int(current_slm_format/2+current_center[1]+size_block/2)] = current_hologram_block
        
    # Add frame to hologram permutation stack      
    partitioned_holograms[:,:,0] =  current_hologram_frame    
        
       
    return partitioned_holograms

def generate_H_stacks(Nx, Ny, zs, wavelength, psX_hologram, psY_hologram):
    Hs = []
    for z in zs:
        # Create the XY grid for H stack computation, centered around zero
        x, y = np.meshgrid(np.linspace(-Nx//2+1, Nx//2, int(Nx)),
                           np.linspace(-Ny//2+1, Ny//2, int(Ny)))
        
        # Normalize the grid to -1/2 to 1/2 first, then scale it to max spatial frequency
        fx = x/psX_hologram/Nx
        fy = y/psY_hologram/Ny
        
        # Fresnel prop exponential
        exp = np.exp(1j * np.pi * wavelength * z * (fx**2 + fy**2))
        
        # Add it to the stack
        Hs.append(exp.astype(np.complex64))
        #Hs.append(np.fft.fftshift(exp.astype(np.complex64)))
    return Hs

def fresnel_forward_propagate(source, hologram, H):
    # Compute source X SLM phase mask
    object_field = source * np.exp(1j * hologram)
    
    # Compute and return the image at the target Z plane as defined by the H matrix
    image = np.abs(np.fft.fftshift(np.fft.fft2(np.multiply(object_field, H)))**2)
    return image

def compute_volume_intensity(source, hologram, Hs, targets =[], plot_enabled=False, size_target=25):
    # Extract the dimensions from H matrices
    Nz,Nx,Ny = np.shape(Hs);
    
    # Initialize the Intensity matrix
    volume_intensity = np.zeros([Nz, Nx, Ny])
    
    # Forward propagate the SLM phase mask to each depth plane as defined by H matrices
    for z in range(Nz):
        if plot_enabled:
            imagez = fresnel_forward_propagate(source, hologram, Hs[z])
            if z == int(np.floor((Nz+1)/2))-1:
                target_plane_intensity = np.array(imagez)
            volume_intensity[z,:,:] = imagez
            
            circles = []
            for target_index in range(len(targets)):
                target = targets[target_index]
                circles.append(plt.Circle((target[0],target[1]), size_target, color = 'r', fill=False))
            plt.figure()
            plt.imshow(imagez,interpolation='none')
            ax = plt.gca()
            for target_index in range(len(targets)):
                ax.add_patch(circles[target_index])
                
        else:
            if z == int(np.floor((Nz+1)/2))-1:
                imagez = fresnel_forward_propagate(source, hologram, Hs[z])
                target_plane_intensity = np.array(imagez)
            else:
                continue
            
    return target_plane_intensity

def compute_volume_intensity_oversampled(system, source, holograms, Hs, targets =[], plot_enabled=False, size_target=25):
    
    Nz,Nx,Ny = np.shape(Hs);
    target_plane_intensity = np.zeros([ Nx, Ny])
    
    for hologram in holograms:
        target_plane_intensity += (compute_volume_intensity(source, hologram, Hs, targets, plot_enabled=False, size_target=size_target))

    target_plane_intensity = target_plane_intensity / system['excess_refresh_rate']
    
    if plot_enabled:
        circles = []
        for target_index in range(len(targets)):
            target = targets[target_index]
            circles.append(plt.Circle((target[0],target[1]), size_target, color = 'r', fill=False))
        plt.figure()
        plt.imshow(target_plane_intensity,interpolation='none')
        ax = plt.gca()
        for target_index in range(len(targets)):
            ax.add_patch(circles[target_index])
    
    return target_plane_intensity
    
def calculate_axial_range_step(system, targets, slm_format):
    
    if targets['random_locations']:
        if system['decompose_en']:
            num_blocks=int(targets['random_target_count']/system['excess_refresh_rate'])
        else:
            num_blocks = targets['random_target_count']
    else:
        if system['decompose_en']:
            num_blocks=int(np.size(targets['locations'],0)/system['excess_refresh_rate'])
        else:
            num_blocks = np.size(targets['locations'],axis=0)
    
    block_format = int(slm_format/np.ceil(np.sqrt(num_blocks)))
    f = system['optics_focal_length']
    lam = system['optics_wavelength']
    p = system['slm_pix_size']
    

    #range_axial_both[0] = 8*f*f*lam/((block_format-1)/2)/(lam*lam+4*p*p)
    #range_axial_both[1] = 1/ (2 * np.amax(abs(centers_1D)) * system['slm_pix_size']/ system['optics_focal_length']/2/(system['optics_wavelength']*system['optics_focal_length']/system['slm_pix_size']))
    #range_axial[i] = np.amin(range_axial_both)
    
    range_axial = 4*f*f*lam/((block_format-1)/2)/(lam*lam+4*p*p)
    step_axial = 4*lam/((np.sin(np.arctan((block_format/2)*p/f)))**2)
    if system['verbose']:
        print('NOTE: Recommended axial range for SLM format of %i is +/- %4.3e m.' % (slm_format, range_axial))
        print('NOTE: Recommended min depth step for SLM format of %i is +/- %4.3e m.' % (slm_format, step_axial))
        
    return range_axial, step_axial
  
def generate_volume_intensity_GS(system,current_targets,targets,depths_unique,slm_format,source,size_target):
    
    psX_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_nx']
    psY_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_ny']
    
    # Define the array that will hold the generated intensity at target depths
    generated_depth_intensities_GS = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
    
    # Compute the Global GS Phase Mask for the given targets
    tic_gs = time.perf_counter()
    hologram_phase_GS = compute_hologram_globalGS(system,current_targets,slm_format,size_target,num_iter=system['cgh_gs_iter_count'],initial_phase_mask="superposition") 
    toc_gs = time.perf_counter()
    computation_time_GS = toc_gs - tic_gs
    if system['verbose']:
        print('GS completed in %0.4f seconds.' % (computation_time_GS))
    
    # Discretize, add noise, and resize the GS-generated Phase Mask
    hologram_phase_GS = discretize_phase_mask(system, hologram_phase_GS)
    hologram_phase_GS_resized = resize_and_zeropad(system,slm_format,hologram_phase_GS)
    if system['plots_en']:
        plt.figure()
        plt.imshow(hologram_phase_GS_resized,interpolation='none')
    
    # Compute and record generated depth intensities
    for current_depth_index in range(np.size(depths_unique)):
        current_depth = depths_unique[current_depth_index]
        
        # Define the z-stack depths for the given depth
        depths = np.linspace((current_depth-targets['zstack_range']/2),(current_depth+targets['zstack_range']/2),num=(int(targets['zstack_range']/targets['zstack_step'])+1))
        
        # Compute the H matrices for each depth in the z-stack
        Hs_simul = generate_H_stacks(system['simul_nx'], system['simul_ny'], depths, system['optics_wavelength'], psX_hologram, psY_hologram)

        # This is the scaling factor and offset to compute the resulting target location in simul_nx from its location in cgh_nx (and ny)
        scaling_factor = (1/system['simul_array_fold_reduction'])
        offset = (system['simul_pix_sampling']-1)/2*(system['cgh_nx']/system['simul_array_fold_reduction'])
        
        # Compute the location of the targets at this depth, in simul_nx (and ny)
        targets_at_this_depth = []
        for target in current_targets:
            if target[0] == current_depth :
                targets_at_this_depth.append([(offset + (target[1]*scaling_factor)),(offset + (target[2]*scaling_factor))])
                
        # Compute the generated intensity for this depth. All of the z-stack is computed inside the function, but only the target plane is returned.
        generated_depth_intensities_GS[current_depth_index,:,:] = (compute_volume_intensity(source, hologram_phase_GS_resized, Hs_simul, targets_at_this_depth, plot_enabled=system['plots_en'], size_target=size_target))
        
        if system['verbose']:
            print("Peak intensity at depth %0.5e (GS) = %.5e" % (current_depth, np.amax(generated_depth_intensities_GS[current_depth_index,:,:])))
            print("Total intensity at depth %.5e (GS) = %.5e" % (current_depth, np.sum(generated_depth_intensities_GS[current_depth_index,:,:])))
    
    return generated_depth_intensities_GS, computation_time_GS, hologram_phase_GS_resized


def generate_volume_intensity_nimblepatch_optimum(system,current_targets,targets,depths_unique,slm_format,source):
    
    psX_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_nx']
    psY_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_ny']
    size_target = targets['size']
    
    # Define the array that will hold the generated intensity at target depths
    if system['verbose']:
        print("Initializing generated depth intensities array (nimblepatch)")
    generated_depth_intensities_nimblepatch = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
    if system['verbose']:
        print("Depth intensity generation done (nimblepatch)")
    
    # Compute the Optimized nimblepatch Phase Mask for the given targets
    if system['verbose']:
        print("Starting partitioned hologram generation (nimblepatch)")
    tic_ana = time.perf_counter()
    hologram_phase_nimblepatch_optimum = generate_partitioned_hologram(system,current_targets,slm_format,size_target)[:,:,0]
    toc_ana = time.perf_counter()
    if system['verbose']:
        print("Partitioned hologram generation done (nimblepatch)")
    computation_time_ana = toc_ana - tic_ana
    if system['verbose']:
        print('nimblepatch computation for %i permutations completed in %0.4f seconds.' % (system['simul_nimblepatch_permutations_to_plot'],computation_time_ana))
    
    hologram_phase_nimblepatch_optimum = discretize_phase_mask(system, hologram_phase_nimblepatch_optimum)
    
    # Discretize, add noise, and resize the GS-generated Phase Mask
    hologram_phase_nimblepatch_optimum_resized = resize_and_zeropad(system,slm_format,hologram_phase_nimblepatch_optimum)
    if system['plots_en']:
        plt.figure()
        plt.imshow(hologram_phase_nimblepatch_optimum_resized,interpolation='none')
        #plt.savefig(f'{img_index}.png')
    
    # Compute and record generated depth intensities
    for current_depth_index in range(np.size(depths_unique)):
        current_depth = depths_unique[current_depth_index]
        
        # Define the z-stack depths for the given depth
        depths = np.linspace((current_depth-targets['zstack_range']/2),(current_depth+targets['zstack_range']/2),num=(int(targets['zstack_range']/targets['zstack_step'])+1))
        
        # Compute the H matrices for each depth in the z-stack
        Hs_simul = generate_H_stacks(system['simul_nx'], system['simul_ny'], depths, system['optics_wavelength'], psX_hologram, psY_hologram)

        # This is the scaling factor and offset to compute the resulting target location in simul_nx from its location in cgh_nx (and ny)
        scaling_factor = (1/system['simul_array_fold_reduction'])
        offset = (system['simul_pix_sampling']-1)/2*(system['cgh_nx']/system['simul_array_fold_reduction'])
        
        # Compute the location of the targets at this depth, in simul_nx (and ny)
        targets_at_this_depth = []
        for target in current_targets:
            if target[0] == current_depth :
                targets_at_this_depth.append([(offset + (target[1]*scaling_factor)),(offset + (target[2]*scaling_factor))])
                
        # Compute the generated intensity for this depth. All of the z-stack is computed inside the function, but only the target plane is returned.
        generated_depth_intensities_nimblepatch[current_depth_index,:,:] = (compute_volume_intensity(source, hologram_phase_nimblepatch_optimum_resized, Hs_simul, targets_at_this_depth, plot_enabled=system['plots_en'], size_target=size_target))
        
        if system['verbose']:
            print("Peak intensity at depth %0.5e (GS) = %.5e" % (current_depth, np.amax(generated_depth_intensities_nimblepatch[current_depth_index,:,:])))
            print("Total intensity at depth %.5e (GS) = %.5e" % (current_depth, np.sum(generated_depth_intensities_nimblepatch[current_depth_index,:,:])))
            
    return generated_depth_intensities_nimblepatch, computation_time_ana, hologram_phase_nimblepatch_optimum_resized

def generate_full_volume_intensity(holograms,system,current_targets,targets,decomposed_targets,img_max,slm_format,source,result_dir,mode='single_shot'):
    
    axial_spot_size = system['range_axial'] / ( slm_format / np.sqrt(targets['random_target_count']))    
    target_depth_max = targets['random_target_depth_range'] * system['range_axial'] + axial_spot_size/2
    target_depth_min =  -target_depth_max
    distinct_plane_count = int(targets['random_target_depth_range'] * ( slm_format / np.sqrt(targets['random_target_count'])) + 1)
    num = int(distinct_plane_count * 6 + 1)
    depths = np.linspace(target_depth_min , target_depth_max, num)
    
    #depths = np.linspace(target_depth_min,target_depth_max,num=(int(2*target_depth_max/targets['zstack_step'])+1))
    psX_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_nx']
    psY_hologram = system['optics_wavelength'] * system['optics_focal_length'] / (system['slm_pix_size'] / system['simul_pix_sampling']) / system['simul_ny']
    size_target = targets['size']
    
    tif_dir = os.path.join(result_dir,"Tifs")
    if not os.path.isdir(tif_dir):
        os.makedirs(tif_dir)
        
    npy_dir = os.path.join(result_dir,"Npys")
    if not os.path.isdir(npy_dir):
        os.makedirs(npy_dir)
        
    if mode == 'single_shot':
        hologram = holograms
        plot_path = os.path.join(result_dir,"PM")
        plt.figure()
        plt.imshow(hologram,interpolation='none')
        plt.axis('off')
        plt.savefig(plot_path,bbox_inches='tight')
        np.save(plot_path+".npy",hologram)
        
        crop_x_i = int(system['simul_nx'] / 3)
        crop_x_f = int(2*crop_x_i-1)
        hologram_cropped_PM = hologram[crop_x_i:crop_x_f,crop_x_i:crop_x_f]
        hologram_cropped_PM_resized = hologram_cropped_PM[::5,::5]
        plot_path = os.path.join(result_dir,"PM_resized")
        plt.figure()
        plt.imshow(hologram_cropped_PM_resized,interpolation='none')
        plt.axis('off')
        plt.savefig(plot_path,bbox_inches='tight')
        np.save(plot_path+".npy",hologram_cropped_PM_resized)
    elif mode == 'multi_shot':
        for hologram_index, hologram in enumerate(holograms):
            subframe_dir = os.path.join(result_dir,"F",str(hologram_index))
            if not os.path.isdir(subframe_dir):
                os.makedirs(subframe_dir)
            # subframe_path = os.path.join(subframe_dir,"PM")
            # plt.figure()
            # plt.imshow(hologram,interpolation='none')
            # plt.axis('off')
            # plt.savefig(subframe_path+".npy",bbox_inches='tight')
            # np.save(subframe_path,hologram)
            
            # crop_x_i = int(system['simul_nx'] / 3)
            # crop_x_f = int(2*crop_x_i-1)
            # hologram_cropped_PM = hologram[crop_x_i:crop_x_f,crop_x_i:crop_x_f]
            # hologram_cropped_PM_resized = hologram_cropped_PM[::5,::5]
            # plot_path = os.path.join(result_dir,"PM_resized")
            # plt.figure()
            # plt.imshow(hologram,interpolation='none')
            # plt.axis('off')
            # plt.savefig(plot_path,bbox_inches='tight')
            # np.save(plot_path+".npy",hologram_cropped_PM_resized)

    for current_depth_index, current_depth in enumerate(depths):
        
        if system['verbose']:
            print("Full Volume Capture depth index %d out of %d" % (current_depth_index+1,np.size(depths)))

        # Compute the H matrices for each depth in the z-stack
        current_depth_list = [current_depth]
        Hs_simul = generate_H_stacks(system['simul_nx'], system['simul_ny'], current_depth_list, system['optics_wavelength'], psX_hologram, psY_hologram)

        # This is the scaling factor and offset to compute the resulting target location in simul_nx from its location in cgh_nx (and ny)
        scaling_factor = (1/system['simul_array_fold_reduction'])
        offset = (system['simul_pix_sampling']-1)/2*(system['cgh_nx']/system['simul_array_fold_reduction'])
        
        # Compute the location of the targets at this depth, in simul_nx (and ny)
        targets_at_this_depth = []
        for target in current_targets:
            if abs(target[0] - current_depth) < (5 * targets['zstack_step']) and current_depth <= target[0] + targets['zstack_step'] :
                targets_at_this_depth.append([(offset + (target[1]*scaling_factor)),(offset + (target[2]*scaling_factor))])
                
        # Compute the generated intensity for this depth. All of the z-stack is computed inside the function, but only the target plane is returned.
        if mode == 'single_shot':
            depth_intensity = (compute_volume_intensity(source, hologram, Hs_simul, targets_at_this_depth, plot_enabled=system['plots_en'], size_target=size_target))
        elif mode == 'multi_shot':
            depth_intensities = np.zeros([len(holograms),system['simul_nx'],system['simul_ny']])
            depth_intensity = np.zeros([system['simul_nx'],system['simul_ny']])
            depth_intensity_acc = np.zeros([system['simul_nx'],system['simul_ny']])
            for hologram_index, hologram in enumerate(holograms):
                depth_intensity = (compute_volume_intensity(source, hologram, Hs_simul, targets_at_this_depth, plot_enabled=system['plots_en'], size_target=size_target))
                depth_intensities[hologram_index,:,:] = depth_intensity
                depth_intensity_acc += depth_intensity
            depth_intensity = depth_intensity_acc / len(holograms)
        plot_path = os.path.join(result_dir,str(current_depth_index).zfill(3))
        npy_filename = os.path.join(npy_dir,str(current_depth_index).zfill(3))
        if system['plot_target_circles_en']:
            circles = []
            for target_index, target in enumerate(targets_at_this_depth):
                target = targets_at_this_depth[target_index]
                circles.append(plt.Circle((target[0],target[1]), size_target, color = 'r', fill=False))
        
        if(current_depth_index == 0):
            myplot, myfig = plt.subplots()
            myfig = plt.imshow(depth_intensity,interpolation='none')
        else:
            myfig.set_data(depth_intensity)
            myplot.canvas.flush_events()
            plt.draw()
        
        ax = plt.gca()
        if system['plot_target_circles_en']:
            for target_index in range(len(targets_at_this_depth)):
                ax.add_patch(circles[target_index])
        plt.axis('off')
        plt.savefig(plot_path,bbox_inches='tight')
        
        np.save(npy_filename,depth_intensity)
        #imageio.imwrite(tif_filename+".tif",depth_intensity)
                
        # TODO: Fix this imshow too
        # if mode == 'multi_shot':
        #     for idx, individual_depth_intensity in enumerate(depth_intensities):
        #         subframe_targets = decomposed_targets[idx]
        #         subframe_targets_at_this_depth = []
        #         for subframe_target in subframe_targets:
        #             if abs(subframe_target[0] - current_depth) < (5 * targets['zstack_step']) and current_depth <= subframe_target[0] + targets['zstack_step'] :
        #                 subframe_targets_at_this_depth.append([(offset + (subframe_target[1]*scaling_factor)),(offset + (subframe_target[2]*scaling_factor))])
        #         subframe_dir = os.path.join(result_dir,"F",str(idx))
        #         subframe_path = os.path.join(subframe_dir,str(current_depth_index))
        #         plt.figure()
        #         if system['plot_target_circles_en']:
        #             circles = []
        #             for target_index, target in enumerate(subframe_targets_at_this_depth):
        #                 target = subframe_targets_at_this_depth[target_index]
        #                 circles.append(plt.Circle((target[0],target[1]), size_target, color = 'r', fill=False))
        #         plt.imshow(individual_depth_intensity,interpolation='none')
        #         ax = plt.gca()
        #         if system['plot_target_circles_en']:
        #             for target_index in range(len(subframe_targets_at_this_depth)):
        #                 ax.add_patch(circles[target_index])
        #         plt.axis('off')
        #         plt.savefig(subframe_path,bbox_inches='tight')
        #         plt.close()
    
    plt.close()
    return 3

def run_cgh_sweeps(system_main,targets_main,unique_id="000000"):
    
    accuracy_nimblepatch_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    accuracy_nimblepatch_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    accuracy_GS_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    accuracy_GS_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    accuracy_GS_oversampled = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    
    contrast_nimblepatch_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    contrast_nimblepatch_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    contrast_GS_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    contrast_GS_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    contrast_GS_oversampled = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    
    efficiency_nimblepatch_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    efficiency_nimblepatch_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    efficiency_GS_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    efficiency_GS_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    efficiency_GS_oversampled = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    
    speckle_contrast_nimblepatch_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    speckle_contrast_nimblepatch_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    speckle_contrast_GS_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    speckle_contrast_GS_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    speckle_contrast_GS_oversampled = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    
    computation_time_nimblepatch_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    computation_time_nimblepatch_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    computation_time_GS_singleshot = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    computation_time_GS_decomposed = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    computation_time_GS_oversampled = np.zeros([np.size(system_main['slm_formats']),np.size(targets_main['random_target_count']),np.size(system_main['excess_refresh_rate'])])
    
    algorithm_list = []
    if system_main['nimblepatch_enabled'] :
        if system_main['decompose_en']:
            algorithm_list.append("Ad")
        else:
            algorithm_list.append("As")
    if system_main['GS_enabled'] :
        algorithm_list.append("Gs")
        if system_main['decompose_en']:
            algorithm_list.append("Gd")
        if system_main['oversample_en']:
            algorithm_list.append("Go")

    for current_slm_format_index in range(np.size(system_main['slm_formats'])) :
        
        current_slm_format = system_main['slm_formats'][current_slm_format_index]
        
        for current_target_count_index in range(np.size(targets_main['random_target_count'])):
            
            current_target_count = targets_main['random_target_count'][current_target_count_index]
            
            for current_excess_refresh_rate_index in range(np.size(system_main['excess_refresh_rate'])):
                
                system = dict(system_main)
                targets = dict(targets_main)
                
                system['simul_nx'] = int(system['cgh_nx'] * system['simul_pix_sampling'] / system['simul_array_fold_reduction'])
                system['simul_ny'] = int(system['cgh_ny'] * system['simul_pix_sampling'] / system['simul_array_fold_reduction'])
                system['cgh_gs_nx'] = int(system['cgh_nx'] * system['cgh_gs_pix_sampling'])
                system['cgh_gs_ny'] = int(system['cgh_ny'] * system['cgh_gs_pix_sampling'])
                
                if system['depth_plane_count_scaling_en']:
                    targets['random_target_depth_count'] = int(targets['random_target_depth_range'] * current_slm_format / np.sqrt(current_target_count)) + 1  
                else:
                    targets['random_target_depth_count'] = int((1/4) * current_slm_format) + 1
                
                source_simul_full = (1/ system['simul_nx']) * (1/ system['simul_ny']) * np.ones([int(system['simul_nx']),int(system['simul_ny'])])
                
                system['excess_refresh_rate'] = system_main['excess_refresh_rate'][current_excess_refresh_rate_index]
            
                print("Starting calculations for SLM Size: %d, Target Count: %d" % (current_slm_format,current_target_count))
            
                targets['random_target_count'] = current_target_count
                range_axial, step_axial = calculate_axial_range_step(system, targets,current_slm_format)
                system['range_axial'] = range_axial
                system['step_axial'] = step_axial
                
                if targets['random_locations'] is True:
                    try:
                        targets['locations'] = generate_random_target_locations(system,targets)
                    except:
                        print("Targets not generatable, moving on")
                        return
                
                # Reset the seed for GS
                np.random.seed(None)
                
                if system['verbose']:
                    print("Compute unique depth planes and number of depth planes (universal)")
                depths_unique = np.unique(targets['locations'][:,0])
                system['simul_nz'] = int(np.size(depths_unique) * (targets['zstack_range']/targets['zstack_step']+1))
                
                if system['verbose']:
                    print("Generate masked source (universal)")
                slm_mask = np.ones([current_slm_format,current_slm_format])
                maskarray = resize_and_zeropad(system,current_slm_format,slm_mask)
                source_masked = np.multiply(source_simul_full, maskarray)
                source_masked = 1
                
                if system['decompose_en']:
                    num_blocks=int(np.size(targets['locations'],0)/system['excess_refresh_rate'])
                    num_frames=system['excess_refresh_rate']
                    
                    if system['verbose']:
                        print("Decompose and assign targets across frames")
                    try:
                        assignments = decompose_and_assign_targets(system,targets['locations'],current_slm_format,num_frames,num_blocks)
                    except:
                        breakpoint()
                        assignments = decompose_and_assign_targets(system,targets['locations'],current_slm_format,num_frames,num_blocks)
                    assignments = np.reshape(assignments,(num_frames,num_blocks,6))
                    decomposed_x = assignments[:,:,0]
                    decomposed_x = np.expand_dims(decomposed_x,axis=2)
                    decomposed_y = assignments[:,:,1]
                    decomposed_y = np.expand_dims(decomposed_y,axis=2)
                    decomposed_z = assignments[:,:,2]
                    decomposed_z = np.expand_dims(decomposed_z,axis=2)
                    decomposed_targets = np.concatenate((decomposed_z,decomposed_y,decomposed_x),axis=2)
                                        
                    size_target = compute_target_size(system,decomposed_targets[0],current_slm_format,targets['size'])
                    target_plane_intensities, target_plane_depths = generate_target_plane_intensities(system,targets['locations'],current_slm_format,size_target)
                    target_plane_intensities_resized = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
                    for current_depth_index in range(np.size(depths_unique)):
                        target_plane_intensities_resized[current_depth_index,:,:] = resize_and_zeropad_targets(system,target_plane_intensities[current_depth_index,:,:])
                
                else:
                    size_target = compute_target_size(system,targets['locations'],current_slm_format,targets['size'])
                    target_plane_intensities, target_plane_depths = generate_target_plane_intensities(system,targets['locations'],current_slm_format,size_target)
                    target_plane_intensities_resized = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
                    for current_depth_index in range(np.size(depths_unique)):
                        target_plane_intensities_resized[current_depth_index,:,:] = resize_and_zeropad_targets(system,target_plane_intensities[current_depth_index,:,:])
                
                if system['verbose']:
                    print("Compute target size (universal)")
                targets['size'] = size_target
                result_root_dir = "results/FVC/"+"SLMFormat_"+str(current_slm_format)+"_Ntarg_"+str(current_target_count)+"_ExRefR_"+str(system_main['excess_refresh_rate'])+"_"+time.strftime("%Y%m%d-%H%M%S")+"/"
                
                    # nimblepatch Computation + Partitioning Approach
                if system['nimblepatch_enabled'] :
                    
                    if system['decompose_en'] == False or system['nimblepatch_singleshot_en']:
                    
                        if system['nimblepatch_optimized']:
                            system['simul_nimblepatch_permutations_to_plot'] = 1
                            
                        print("Starting nimblepatch single-shot computation")

                    
                        system['simul_nz'] = int(np.size(depths_unique) * (targets['zstack_range']/targets['zstack_step']+1))
                        generated_depth_intensities_nimblepatch_singleshot, computation_time_nimblepatch_singleshot_acc, hologram_nimblepatch_singleshot = generate_volume_intensity_nimblepatch_optimum(system,targets['locations'],targets,depths_unique,current_slm_format,source_masked)
            
                        target_plane_intensities_resized_0th = crop_to_0th_order(system,target_plane_intensities_resized)
                        generated_depth_intensities_nimblepatch_singleshot_0th = crop_to_0th_order(system,generated_depth_intensities_nimblepatch_singleshot)
                
                        accuracy_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_accuracy(target_plane_intensities_resized_0th, generated_depth_intensities_nimblepatch_singleshot_0th)
                        contrast_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_contrast(target_plane_intensities_resized, generated_depth_intensities_nimblepatch_singleshot)
                        efficiency_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_efficiency(target_plane_intensities_resized, generated_depth_intensities_nimblepatch_singleshot)
                        speckle_contrast_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_speckle_contrast(target_plane_intensities_resized_0th, generated_depth_intensities_nimblepatch_singleshot_0th)
                        
                        computation_time_nimblepatch_singleshot[current_slm_format_index,current_target_count_index] = computation_time_nimblepatch_singleshot_acc

                        if system['lite_plots_en'] or system['plots_en']:
                            for depth_intensity in generated_depth_intensities_nimblepatch_singleshot:
                                plt.figure()
                                plt.imshow(depth_intensity,interpolation='none')
                             
                        print("Accuracy in 0th order (nimblepatch single-shot) = %f" % accuracy_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Contrast (nimblepatch single-shot) = %f" % contrast_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Efficiency (nimblepatch single-shot) = %f" % efficiency_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Speckle Contrast in 0th order (nimblepatch single-shot) = %f" % speckle_contrast_nimblepatch_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Compute Time (nimblepatch single-shot) = %f" % computation_time_nimblepatch_singleshot_acc)
                        
                        if system['capture_full_volume_en']:
                            result_dir = os.path.join(result_root_dir, "As")
                            if not os.path.isdir(result_dir):
                                os.makedirs(result_dir)
                            generate_full_volume_intensity(hologram_nimblepatch_singleshot,system,targets['locations'],targets,0,np.amax(generated_depth_intensities_nimblepatch_singleshot),current_slm_format,source_masked,result_dir,mode='single_shot')
                    
                    if system['decompose_en']:
                        computation_time_nimblepatch_decomposed_acc = 0
                        generated_depth_intensities_nimblepatch_decomposed = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
                        holograms_nimblepatch_decomposed = []
                        for current_frame_number,current_decomposed_targets in enumerate(decomposed_targets):
                            current_depths_unique = np.unique(current_decomposed_targets[:,0])
                            system['simul_nz'] = int(np.size(current_depths_unique) * (targets['zstack_range']/targets['zstack_step']+1))
                            generated_depth_intensities_nimblepatch_decomposed_this_frame, computation_time_nimblepatch_decomposed_this_frame, hologram_nimblepatch_decomposed_this_frame = generate_volume_intensity_nimblepatch_optimum(system,current_decomposed_targets,targets,depths_unique,current_slm_format,source_masked)
                            computation_time_nimblepatch_decomposed_acc += computation_time_nimblepatch_decomposed_this_frame
                            generated_depth_intensities_nimblepatch_decomposed += generated_depth_intensities_nimblepatch_decomposed_this_frame
                            holograms_nimblepatch_decomposed.append(hologram_nimblepatch_decomposed_this_frame)
                            
                        generated_depth_intensities_nimblepatch_decomposed = generated_depth_intensities_nimblepatch_decomposed/system['excess_refresh_rate']
            
                        target_plane_intensities_resized_0th = crop_to_0th_order(system,target_plane_intensities_resized)
                        generated_depth_intensities_nimblepatch_decomposed_0th = crop_to_0th_order(system,generated_depth_intensities_nimblepatch_decomposed)
                
                        accuracy_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_accuracy(target_plane_intensities_resized_0th, generated_depth_intensities_nimblepatch_decomposed_0th)
                        contrast_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_contrast(target_plane_intensities_resized, generated_depth_intensities_nimblepatch_decomposed)
                        efficiency_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_efficiency(target_plane_intensities_resized, generated_depth_intensities_nimblepatch_decomposed)
                        speckle_contrast_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_speckle_contrast(target_plane_intensities_resized_0th, generated_depth_intensities_nimblepatch_decomposed_0th)
                        
                        computation_time_nimblepatch_decomposed[current_slm_format_index,current_target_count_index] = computation_time_nimblepatch_decomposed_acc

                        if system['lite_plots_en'] or system['plots_en']:
                            for depth_intensity in generated_depth_intensities_nimblepatch_decomposed:
                                plt.figure()
                                plt.imshow(depth_intensity,interpolation='none')
                             
                        print("Accuracy in 0th order (nimblepatch decomposed optimum) = %f" % accuracy_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Contrast (nimblepatch decomposed optimum) = %f" % contrast_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Efficiency (nimblepatch decomposed optimum) = %f" % efficiency_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Speckle Contrast in 0th order (nimblepatch decomposed optimum) = %f" % speckle_contrast_nimblepatch_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Compute Time (nimblepatch decomposed optimum) = %f" % computation_time_nimblepatch_decomposed_acc)
                        
                        if system['capture_full_volume_en']:
                            result_dir = os.path.join(result_root_dir, "Ad")
                            if not os.path.isdir(result_dir):
                                os.makedirs(result_dir)
                            generate_full_volume_intensity(holograms_nimblepatch_decomposed,system,targets['locations'],targets,decomposed_targets,np.amax(generated_depth_intensities_nimblepatch_decomposed),current_slm_format,source_masked,result_dir,mode='multi_shot')
                    
                
                if system['GS_enabled']:
                    
                    print('GS Single-shot calculation started.' )
                    
                    generated_depth_intensities_GS_singleshot, computation_time_GS_singleshot_acc, hologram_GS_singleshot = generate_volume_intensity_GS(system,targets['locations'],targets,depths_unique,current_slm_format,source_masked,size_target)
        
                    # Crop only to 0th order (center of the FoV) for accuracy and speckle contrast calculation
                    target_plane_intensities_resized_0th = crop_to_0th_order(system,target_plane_intensities_resized)
                    generated_depth_intensities_GS_singleshot_0th = crop_to_0th_order(system,generated_depth_intensities_GS_singleshot)
                    
                    # Calculate the metrics
                    accuracy_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_accuracy(target_plane_intensities_resized_0th, generated_depth_intensities_GS_singleshot_0th)
                    contrast_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_contrast(target_plane_intensities_resized, generated_depth_intensities_GS_singleshot)
                    efficiency_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_efficiency(target_plane_intensities_resized, generated_depth_intensities_GS_singleshot)
                    speckle_contrast_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_speckle_contrast(target_plane_intensities_resized_0th, generated_depth_intensities_GS_singleshot_0th)
                    
                    computation_time_GS_singleshot[current_slm_format_index,current_target_count_index] = computation_time_GS_singleshot_acc

                    print("Accuracy in 0th order (GS single-shot) = %f" % accuracy_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                    print("Contrast (GS single-shot) = %f" % contrast_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                    print("Efficiency (GS single-shot) = %f" % efficiency_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                    print("Speckle Contrast in 0th order (GS single-shot) = %f" % speckle_contrast_GS_singleshot[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                    print("Compute Time (GS single-shot) = %f" % computation_time_GS_singleshot_acc)
                    
                    if system['capture_full_volume_en']:
                        result_dir = os.path.join(result_root_dir, "Gs")
                        if not os.path.isdir(result_dir):
                            os.makedirs(result_dir)
                        generate_full_volume_intensity(hologram_GS_singleshot,system,targets['locations'],targets,0,np.amax(generated_depth_intensities_GS_singleshot),current_slm_format,source_masked,result_dir,mode='single_shot')
                    
                    if system['lite_plots_en'] or system['plots_en']:
                        for depth_intensity in generated_depth_intensities_GS_singleshot:
                            plt.figure()
                            plt.imshow(depth_intensity,interpolation='none')
                        
                    if system['decompose_en']:
                        
                        print('Decomposed GS calculation started.' )
                        
                        # Given targets and system parameters, compute the phase mask and the resulting intensities at target depth planes using GS Algorithm system['excess_refresh_rate'] times
                        # sum them, and average them
                        computation_time_GS_decomposed_acc = 0
                        generated_depth_intensities_GS_decomposed = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
                        holograms_GS_decomposed = []
                        for current_frame_number,current_decomposed_targets in enumerate(decomposed_targets):
                            system['simul_nz'] = int(np.size(depths_unique) * (targets['zstack_range']/targets['zstack_step']+1))
                            generated_depth_intensities_GS_decomposed_this_frame, computation_time_GS_decomposed_this_frame, hologram_GS_decomposed_this_frame = generate_volume_intensity_GS(system,current_decomposed_targets,targets,depths_unique,current_slm_format,source_masked,size_target)
                            generated_depth_intensities_GS_decomposed += generated_depth_intensities_GS_decomposed_this_frame
                            computation_time_GS_decomposed_acc += computation_time_GS_decomposed_this_frame
                            holograms_GS_decomposed.append(hologram_GS_decomposed_this_frame)
                            
                        generated_depth_intensities_GS_decomposed = generated_depth_intensities_GS_decomposed/system['excess_refresh_rate']
                
                        # Crop only to 0th order (center of the FoV) for accuracy and speckle contrast calculation
                        target_plane_intensities_resized_0th = crop_to_0th_order(system,target_plane_intensities_resized)
                        generated_depth_intensities_GS_decomposed_0th = crop_to_0th_order(system,generated_depth_intensities_GS_decomposed)
                        
                        # Calculate the metrics
                        accuracy_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_accuracy(target_plane_intensities_resized_0th, generated_depth_intensities_GS_decomposed_0th)
                        contrast_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_contrast(target_plane_intensities_resized, generated_depth_intensities_GS_decomposed)
                        efficiency_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_efficiency(target_plane_intensities_resized, generated_depth_intensities_GS_decomposed)
                        speckle_contrast_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index] = metrics.calculate_speckle_contrast(target_plane_intensities_resized_0th, generated_depth_intensities_GS_decomposed_0th)
                        
                        computation_time_GS_decomposed[current_slm_format_index,current_target_count_index] = computation_time_GS_decomposed_acc

                        if system['lite_plots_en'] or system['plots_en']:
                            for depth_intensity in generated_depth_intensities_GS_decomposed:
                                plt.figure()
                                plt.imshow(depth_intensity,interpolation='none')
                        
                        print("Accuracy in 0th order (GS decomposed) = %f" % accuracy_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Contrast (GS decomposed) = %f" % contrast_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Efficiency (GS decomposed) = %f" % efficiency_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Speckle Contrast in 0th order (GS decomposed) = %f" % speckle_contrast_GS_decomposed[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Compute Time (GS decomposed) = %f" % computation_time_GS_decomposed_acc)
                        
                        if system['capture_full_volume_en']:
                            result_dir = os.path.join(result_root_dir, "Gd")
                            if not os.path.isdir(result_dir):
                                os.makedirs(result_dir)
                            generate_full_volume_intensity(holograms_GS_decomposed,system,targets['locations'],targets,decomposed_targets,np.amax(generated_depth_intensities_GS_decomposed),current_slm_format,source_masked,result_dir,mode='multi_shot')
        
                    # Oversampled Gerchberg-Saxton
                    if system['oversample_en']:
                        print('Oversampled GS calculation started.' )
                        
                        # Given targets and system parameters, compute the phase mask and the resulting intensities at target depth planes using GS Algorithm system['excess_refresh_rate'] times
                        # sum them, and average them
                        computation_time_GS_oversampled_acc = 0
                        generated_depth_intensities_GS_oversampled = np.zeros([np.size(depths_unique),system['simul_nx'],system['simul_ny']])
                        holograms_GS_oversampled = []
                        for oversample_idx in range(system['excess_refresh_rate']):
                            generated_depth_intensities_GS_oversampled_this_frame, computation_time_GS_oversampled_this_frame, hologram_GS_oversampled_this_frame = generate_volume_intensity_GS(system,targets['locations'],targets,depths_unique,current_slm_format,source_masked,size_target)
                            generated_depth_intensities_GS_oversampled += generated_depth_intensities_GS_oversampled_this_frame
                            computation_time_GS_oversampled_acc += computation_time_GS_oversampled_this_frame
                            holograms_GS_oversampled.append(hologram_GS_oversampled_this_frame)
                        generated_depth_intensities_GS_oversampled = generated_depth_intensities_GS_oversampled/system['excess_refresh_rate']
            
                        # Crop only to 0th order (center of the FoV) for accuracy and speckle contrast calculation
                        target_plane_intensities_resized_0th = crop_to_0th_order(system,target_plane_intensities_resized)
                        generated_depth_intensities_GS_oversampled_0th = crop_to_0th_order(system,generated_depth_intensities_GS_oversampled)
                        
                        # Calculate the metrics
                        accuracy_GS_oversampled[current_slm_format_index,current_target_count_index] = metrics.calculate_accuracy(target_plane_intensities_resized_0th, generated_depth_intensities_GS_oversampled_0th)
                        contrast_GS_oversampled[current_slm_format_index,current_target_count_index] = metrics.calculate_contrast(target_plane_intensities_resized, generated_depth_intensities_GS_oversampled)
                        efficiency_GS_oversampled[current_slm_format_index,current_target_count_index] = metrics.calculate_efficiency(target_plane_intensities_resized, generated_depth_intensities_GS_oversampled)
                        speckle_contrast_GS_oversampled[current_slm_format_index,current_target_count_index] = metrics.calculate_speckle_contrast(target_plane_intensities_resized_0th, generated_depth_intensities_GS_oversampled_0th)
                        
                        computation_time_GS_oversampled[current_slm_format_index,current_target_count_index] = computation_time_GS_oversampled_acc
                        
                        print("Accuracy in 0th order (GS oversampled) = %f" % accuracy_GS_oversampled[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Contrast (GS oversampled) = %f" % contrast_GS_oversampled[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Efficiency (GS oversampled) = %f" % efficiency_GS_oversampled[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Speckle Contrast in 0th order (GS oversampled) = %f" % speckle_contrast_GS_oversampled[current_slm_format_index,current_target_count_index,current_excess_refresh_rate_index])
                        print("Compute Time (GS oversampled) = %f" % computation_time_GS_oversampled_acc)
                        
                        if system['lite_plots_en'] or system['plots_en']:
                            for depth_intensity in generated_depth_intensities_GS_oversampled:
                                plt.figure()
                                plt.imshow(depth_intensity,interpolation='none')
                                
                        if system['capture_full_volume_en']:
                            result_dir = os.path.join(result_root_dir, "Go")
                            if not os.path.isdir(result_dir):
                                os.makedirs(result_dir)
                            generate_full_volume_intensity(holograms_GS_oversampled,system,targets['locations'],targets,decomposed_targets,np.amax(generated_depth_intensities_GS_oversampled),current_slm_format,source_masked,result_dir,mode='multi_shot')
         
        
def run_main():
    system = {
            'verbose' : True, # Additional verbosity for debugging
            'plots_en' : True, # Plot depth plane intensities near target planes
            'lite_plots_en' : False, # Plot just the depth planes
            'debug_en' : False,
            'nimblepatch_enabled' : True, # Enable NIMBLE-PATCH
            'nimblepatch_singleshot_en' : False,
            'nimblepatch_optimized' : True,
            'oversample_en': False,
            'decompose_en': False,
            'simul_nimblepatch_permutations_to_plot' : 1,
            'GS_enabled' : True,
            'cgh_zero_padding_factor' : 3, # int, Number of pixels in X axis for holography computation
            'cgh_GS_offset' : 0, # float, Regularization constant to allow low background light for GS algorithm
            'optics_focal_length' : 0.009, # meters, focal length of the lens after the SLM. Here 9 mm corresponds with 20x objective with NA of 1
            'optics_wavelength' : 940e-9, # meters, Wavelength of operation
            'slm_pix_size' : 70e-6, # meters, Apparent size of the phase modulating elements
            'slm_bit_depth' : 8, # bits, SLM mirror control bit depth
            'slm_noise_enabled' : False,
            'slm_phase_noise_rms' : (2*np.pi)/10,
            'cgh_gs_pix_sampling' : 1, # int, How many simulation pixels are used to represent a single SLM phase modulating element during GS CGH computation
            'cgh_gs_iter_count' : 50, # int, How many simulation pixels are used to represent a single SLM phase modulating element during GS CGH computation
            'simul_pix_sampling' : 5, # int, How many simulation pixels are used to represent a single SLM phase modulating element during propagation
            'simul_array_fold_reduction' : 1, # int (must be < simul_pix_sampling), How much the computed hologram will be resized (shrunk) to save computation time
            'save_results' : False,
            'capture_full_volume_en' : False,
            'plot_target_circles_en' : False,
            'depth_plane_count_scaling_en' : True, # Should depth plane count scale with SLM block size?
            'multithread_en' : False,
            'multithread_count' : 4,
        }
    
    targets = {
            'type' : 'disk',
            'random_locations' : True,
            'size' : 1, # radius of the target in terms of cgh_nx/ny pixels
            'random_target_depth_range' : 0.5, # maximum axial half range for random target generation as ratio to mean pi phase step along radius
            'random_target_lateral_range' : 0.8, # maximum lateral range, in terms of maximum possible 0th order deflection (0-pi-0-pi)
            'zstack_range' : 100e-6, # total depth over which zstack image will be generated for a given spot
            'zstack_step' : 50e-6, # step size for the zstack image
            'random_target_seed' : 7, # seed for random target generation
            'experimental_setup_conditionals_en' : True, # more restricted target locations to help with experimental setup
            'experimental_setup_keepout_x_range' : 0.125,
            'experimental_setup_keepout_y_range' : 0.125,
            'use_seed' : True, # switch to choose whether to use random target seed
        }
    
    sweeps = {
        'run_count' : 1, # How many times should the sweeps be done
        'slm_formats' : [64,128],
        'excess_refresh_rates' : [1],
        'per_frame_target_counts' : np.array([
            [4],
            [16],
            ], dtype=object)
        }
    

    for run_index in range(sweeps['run_count']):
        for current_slm_format_index, current_slm_format in enumerate(sweeps['slm_formats']):
            for current_excess_refresh_rate_index, current_excess_refresh_rate in enumerate(sweeps['excess_refresh_rates']):
                print("Starting sweeps for SLM Size: %d, Refresh Rate: %d, sweep %d out of %d, run %d out of %d" % (current_slm_format,current_excess_refresh_rate,(current_slm_format_index*np.size(sweeps['excess_refresh_rates'])+current_excess_refresh_rate_index+1),np.size(sweeps['slm_formats']*np.size(sweeps['excess_refresh_rates'])),run_index+1,sweeps['run_count']))
                system['slm_formats'] = [current_slm_format]
                system['cgh_nx'] = system['cgh_zero_padding_factor']*current_slm_format
                system['cgh_ny'] = system['cgh_zero_padding_factor']*current_slm_format
                system['excess_refresh_rate'] = np.array([current_excess_refresh_rate])
                targets['random_target_count'] = np.array(sweeps['per_frame_target_counts'][current_slm_format_index])*current_excess_refresh_rate
                run_cgh_sweeps(system,targets)
                gc.collect()
                
    if system['capture_full_volume_en']:
        plt.close('all')

            
if __name__ == '__main__':
    run_main()
    gc.collect()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
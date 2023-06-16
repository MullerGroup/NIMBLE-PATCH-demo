import numpy as np

def calculate_accuracy(target_intensity,generated_intensity):
    
    inner_product = target_intensity * generated_intensity
    target_intensity_sq = target_intensity**2
    generated_intensity_sq = generated_intensity**2
    
    accuracy_nominator = np.sum(inner_product)
    accuracy_denominator = np.sqrt(np.sum(target_intensity_sq)*np.sum(generated_intensity_sq))
    
    accuracy = accuracy_nominator/accuracy_denominator
    
    return accuracy

def calculate_efficiency(target_intensity,generated_intensity):
    target_intensity_normalized = target_intensity#/np.max(target_intensity)
    masked_generated = generated_intensity * target_intensity_normalized
    
    efficiency_nominator = np.sum(masked_generated)
    efficiency_denominator = np.sum(generated_intensity)
    
    efficiency = efficiency_nominator / efficiency_denominator
    
    return efficiency
    
def calculate_contrast(target_intensity,generated_intensity):
    
    masked_only_targets = generated_intensity * (target_intensity)
    masked_only_non_targets = generated_intensity * (1 - target_intensity)
    
    total_power_targets = np.sum(masked_only_targets)
    total_power_non_targets = np.sum(masked_only_non_targets)
    
    power_density_targets = total_power_targets/np.sum(target_intensity)
    power_density_non_targets = total_power_non_targets/np.sum((1-target_intensity))
    
    contrast = power_density_targets/power_density_non_targets
    
    return contrast

def calculate_speckle_contrast(target_intensity,generated_intensity):
    
    masked_only_non_targets = generated_intensity * (1 - target_intensity)
    
    mean_non_targets = np.mean(masked_only_non_targets)
    stdev_non_targets = np.std(masked_only_non_targets)
    
    speckle_contrast = mean_non_targets/stdev_non_targets
    
    return speckle_contrast

def calculate_peak(generated_intensity):
    
    return np.max(generated_intensity)
























import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.constants import h, c, k

class MichelsonInterferometerSimulator:
    def __init__(self):
        pass
    
    def black_body_spectrum(self, wavenumber_range, resolution, temperature):
        """
        1. Generate synthetic black-body spectrum
        """
        v_min, v_max = wavenumber_range
        
        # Create wavenumber grid
        wavenumbers = np.arange(v_min, v_max, resolution)
        
        # Convert wavenumber from cm⁻¹ to m⁻¹ for calculations
        wavenumbers_m = wavenumbers * 100
        
        # Planck's law for spectral radiance as function of wavenumber
        numerator = 2 * h * c**2 * (wavenumbers_m**3)
        exponent = h * c * wavenumbers_m / (k * temperature)
        denominator = np.exp(exponent) - 1
        
        spectrum = numerator / denominator
        
        # Normalize to make plotting easier
        spectrum = spectrum / np.max(spectrum)
        
        return wavenumbers, spectrum
    
    def add_spectral_lines(self, wavenumbers, spectrum, line_centers, line_strengths, line_widths, line_type='absorption'):
        """
        2. Add absorption/emission lines to the spectrum
        """
        modified_spectrum = spectrum.copy()
        
        for center, strength, width in zip(line_centers, line_strengths, line_widths):
            # Gaussian line profile
            gaussian = strength * np.exp(-0.5 * ((wavenumbers - center) / width)**2)
            
            if line_type == 'absorption':
                # For absorption lines, subtract from continuum
                modified_spectrum -= gaussian
            else:
                # For emission lines, add to continuum
                modified_spectrum += gaussian
        
        # Ensure spectrum doesn't go negative
        modified_spectrum = np.maximum(modified_spectrum, 0)
        
        return modified_spectrum
    
    def generate_interferogram(self, wavenumbers, spectrum, delta_max, num_delta_points):
        """
        4. Construct interferogram using numerical integration
        """
        # Create delta grid (optical path difference)
        delta_values = np.linspace(0, delta_max, num_delta_points)
        
        # Convert wavenumbers to m⁻¹ for consistent units
        wavenumbers_m = wavenumbers * 100
        
        interferogram = np.zeros_like(delta_values)
        
        # Numerical integration using trapezoidal rule
        print("Computing interferogram...")
        for i, delta in enumerate(delta_values):
            # General case: I(δ) = ∫ S(v)(1 + cos(2πvδ))dv
            cos_term = np.cos(2 * np.pi * wavenumbers_m * delta)
            integrand = spectrum * (1 + cos_term)
            interferogram[i] = np.trapz(integrand, wavenumbers_m)
            
            if i % 500 == 0 and i > 0:
                print(f"Progress: {i}/{len(delta_values)}")
        
        return delta_values, interferogram

def improved_recover_spectrum(delta_values, interferogram):
    """
    5. Improved spectrum recovery using Inverse Fourier Transform
    """
    N = len(interferogram)
    
    # Remove DC component (the average value)
    interferogram_ac = interferogram - np.mean(interferogram)
    
    # Apply FFT and take magnitude (important for spectroscopy)
    spectrum_complex = fft(interferogram_ac)
    spectrum_magnitude = np.abs(spectrum_complex)
    
    # FFT shift to center the spectrum
    spectrum_shifted = fftshift(spectrum_magnitude)
    
    # Proper scaling
    recovered_spectrum = spectrum_shifted / N
    
    # Generate wavenumber axis for recovered spectrum
    delta_spacing = np.diff(delta_values)[0] * 0.01  # Convert cm to m
    wavenumbers_fft = np.fft.fftshift(np.fft.fftfreq(N, d=delta_spacing))
    
    # Convert back to cm⁻¹ for comparison
    recovered_wavenumbers = wavenumbers_fft / 100
    
    # Only keep positive wavenumbers (physical solution)
    positive_mask = recovered_wavenumbers > 0
    recovered_wavenumbers = recovered_wavenumbers[positive_mask]
    recovered_spectrum = recovered_spectrum[positive_mask]
    
    return recovered_wavenumbers, recovered_spectrum

def calculate_simple_error(original_wavenumbers, original_spectrum, 
                          recovered_wavenumbers, recovered_spectrum):
    """
    Simple error calculation focusing on spectral features
    """
    # Interpolate recovered spectrum to original wavenumber grid
    recovered_interp = np.interp(original_wavenumbers, recovered_wavenumbers, 
                               recovered_spectrum, left=0, right=0)
    
    # Normalize both to compare shape rather than absolute intensity
    orig_norm = original_spectrum / np.max(original_spectrum)
    rec_norm = recovered_interp / np.max(recovered_interp) if np.max(recovered_interp) > 0 else recovered_interp
    
    # Calculate RMS of normalized spectra
    rms_error = np.sqrt(np.mean((orig_norm - rec_norm)**2))
    
    return rms_error

# Main simulation
if __name__ == "__main__":
    simulator = MichelsonInterferometerSimulator()
    
    print("Michelson Interferometer Simulator")
    print("=" * 50)
    
    # 1. Generate synthetic black-body spectrum
    print("Step 1: Generating black-body spectrum...")
    wavenumbers, bb_spectrum = simulator.black_body_spectrum(
        wavenumber_range=(500, 4000),  # cm⁻¹
        resolution=2,                   # cm⁻¹
        temperature=3000                # K
    )
    
    # Plot black-body spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, bb_spectrum, 'b-', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.title('Black-body Spectrum (T=3000K)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. Add spectral lines
    print("Step 2: Adding spectral lines...")
    line_centers = [1000, 1500, 2000, 2500, 3000, 3500]  # cm⁻¹
    line_strengths = [0.2, 0.3, 0.25, 0.35, 0.15, 0.3]   # absorption depth
    line_widths = [30, 40, 35, 45, 25, 50]               # cm⁻¹
    
    spectrum_with_lines = simulator.add_spectral_lines(
        wavenumbers, bb_spectrum, line_centers, line_strengths, line_widths
    )
    
    # Plot spectrum with lines
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, bb_spectrum, 'b--', alpha=0.7, label='Black-body')
    plt.plot(wavenumbers, spectrum_with_lines, 'g-', linewidth=2, label='With absorption lines')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.title('Synthetic Spectrum with Absorption Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. Generate interferogram (OPTIMIZED PARAMETERS)
    print("Step 3: Generating interferogram with optimized parameters...")
    delta_values, interferogram = simulator.generate_interferogram(
        wavenumbers=wavenumbers,
        spectrum=spectrum_with_lines,
        delta_max=2.0,        # cm - increased for better resolution
        num_delta_points=4096 # more points for better FFT
    )
    
    # Plot interferogram
    plt.figure(figsize=(10, 6))
    plt.plot(delta_values, interferogram, 'purple', linewidth=1)
    plt.xlabel('Optical Path Difference δ (cm)')
    plt.ylabel('Intensity I(δ)')
    plt.title('Interferogram')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 4. Recover spectrum (IMPROVED METHOD)
    print("Step 4: Recovering spectrum with improved method...")
    recovered_wavenumbers, recovered_spectrum = improved_recover_spectrum(
        delta_values, interferogram
    )
    
    # Normalize both spectra for meaningful comparison
    spectrum_with_lines_norm = spectrum_with_lines / np.max(spectrum_with_lines)
    recovered_spectrum_norm = recovered_spectrum / np.max(recovered_spectrum)
    
    # 5. Calculate error and assess recovery
    print("Step 5: Assessing recovery quality...")
    rms_error = calculate_simple_error(wavenumbers, spectrum_with_lines, 
                                     recovered_wavenumbers, recovered_spectrum)
    
    # Final comparison plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(wavenumbers, spectrum_with_lines_norm, 'b-', label='Original', linewidth=2)
    plt.plot(recovered_wavenumbers, recovered_spectrum_norm, 'r--', label='Recovered', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.title('Spectrum Recovery Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Close-up of spectral features
    mask_orig = (wavenumbers >= 800) & (wavenumbers <= 3700)
    mask_rec = (recovered_wavenumbers >= 800) & (recovered_wavenumbers <= 3700)
    plt.plot(wavenumbers[mask_orig], spectrum_with_lines_norm[mask_orig], 'b-', label='Original', linewidth=2)
    plt.plot(recovered_wavenumbers[mask_rec], recovered_spectrum_norm[mask_rec], 'r--', label='Recovered', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.title('Close-up: Spectral Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Residuals
    recovered_interp = np.interp(wavenumbers[mask_orig], recovered_wavenumbers[mask_rec], recovered_spectrum_norm[mask_rec])
    residuals = spectrum_with_lines_norm[mask_orig] - recovered_interp
    plt.plot(wavenumbers[mask_orig], residuals, 'g-', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (RMS: {rms_error:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Interferogram zoom
    plt.plot(delta_values[:500], interferogram[:500], 'purple', linewidth=1)
    plt.xlabel('Optical Path Difference δ (cm)')
    plt.ylabel('Intensity I(δ)')
    plt.title('Interferogram (First 500 points)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    print(f"Normalized RMS Error: {rms_error:.4f}")
    
    if rms_error < 0.1:
        print("✅ EXCELLENT RECOVERY - All features well reproduced!")
    elif rms_error < 0.2:
        print("✅ GOOD RECOVERY - Most features matched")
    elif rms_error < 0.3:
        print("⚠️  FAIR RECOVERY - Some features visible")
    else:
        print("❌ POOR RECOVERY - Check parameters")
    
    print(f"Original range: {wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm⁻¹")
    print(f"Recovered range: {recovered_wavenumbers[0]:.0f}-{recovered_wavenumbers[-1]:.0f} cm⁻¹")
    print("="*60)
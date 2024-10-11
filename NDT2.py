import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from matplotlib.widgets import Slider, TextBox
from PIL import Image
from tqdm import tqdm
import os

def load_gif(filepath):
    gif = Image.open(filepath)
    frames = []
    try:
        with tqdm(desc="Loading GIF frames", unit="frame") as pbar:
            while True:
                frame = np.array(gif.convert('L'))  # Convert to grayscale
                frames.append(frame)
                gif.seek(gif.tell() + 1)
                pbar.update(1)
    except EOFError:
        pass
    return np.array(frames)  # Convert to 3D numpy array

def compute_temporal_fft(frames):
    with tqdm(desc="Computing temporal FFT", total=1) as pbar:
        fft_result = fftshift(fft(frames, axis=0), axes=0)
        pbar.update(1)
    return fft_result

def compute_unwrapped_phase(fft_data):
    with tqdm(desc="Computing unwrapped phase", total=1) as pbar:
        phase = np.angle(fft_data)
        unwrapped_phase = np.unwrap(phase, axis=0)
        pbar.update(1)
    return unwrapped_phase

def compute_wrapped_phase(fft_data):
    with tqdm(desc="Computing wrapped phase", total=1) as pbar:
        wrapped_phase = np.angle(fft_data)
        pbar.update(1)
    return wrapped_phase

def find_max_wrapped_phase_difference(wrapped_phases):
    num_frames = wrapped_phases.shape[0]
    max_diff = 0
    max_pair = (0, 1)
    
    total_comparisons = (num_frames * (num_frames - 1)) // 2
    with tqdm(desc="Finding max phase difference", total=total_comparisons) as pbar:
        for i in range(num_frames):
            for j in range(i+1, num_frames):
                diff = np.mean(np.abs(np.angle(np.exp(1j*(wrapped_phases[j] - wrapped_phases[i])))))
                if diff > max_diff:
                    max_diff = diff
                    max_pair = (i, j)
                pbar.update(1)
    
    return max_pair, max_diff

def main():
    # Use relative path, assuming the GIF is in the same directory as this script
    script_dir = os.path.dirname(__file__)  # Directory where the script is located
    filepath = os.path.join(script_dir, "0.5mL_4mm.gif")  # Adjust the filename as needed
    
    # Load the GIF frames
    gif_frames = load_gif(filepath)
    print(f"Number of frames in the GIF: {gif_frames.shape[0]}")
    
    # Compute temporal FFT for each pixel
    fft_data = compute_temporal_fft(gif_frames)
    
    # Compute unwrapped and wrapped phases
    unwrapped_phases = compute_unwrapped_phase(fft_data)
    wrapped_phases = compute_wrapped_phase(fft_data)
    
    # Find frames with maximum wrapped phase difference
    max_diff_pair, max_diff = find_max_wrapped_phase_difference(wrapped_phases)
    print(f"Frames with maximum wrapped phase difference: {max_diff_pair}")
    print(f"Maximum wrapped phase difference: {max_diff:.4f}")
    
    # Initial frame indices (using the frames with max difference)
    initial_prev, initial_curr = max_diff_pair
    
    # Calculate initial delta phi (unwrapped)
    delta_unwrapped = unwrapped_phases[initial_curr] - unwrapped_phases[initial_prev]
    
    # Calculate initial delta phi (wrapped)
    delta_wrapped = np.angle(np.exp(1j*(wrapped_phases[initial_curr] - wrapped_phases[initial_prev])))
    
    # Create figure and axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.9, wspace=0.4)
    
    # Determine common vmin and vmax for consistent scaling
    vmin, vmax = np.min(gif_frames), np.max(gif_frames)
    
    # Display the initial frames and delta phi
    img_prev = ax1.imshow(gif_frames[initial_prev], cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Previous Frame ({initial_prev})')
    plt.colorbar(img_prev, ax=ax1)
    
    img_curr = ax2.imshow(gif_frames[initial_curr], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Current Frame ({initial_curr})')
    plt.colorbar(img_curr, ax=ax2)
    
    img_delta_unwrapped = ax3.imshow(delta_unwrapped, cmap='viridis')
    ax3.set_title('Unwrapped Phase Difference')
    plt.colorbar(img_delta_unwrapped, ax=ax3)
    
    img_delta_wrapped = ax4.imshow(delta_wrapped, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Wrapped Phase Difference')
    plt.colorbar(img_delta_wrapped, ax=ax4)
    
    # Add sliders for selecting frames
    ax_prev = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_curr = plt.axes([0.1, 0.15, 0.8, 0.03])
    
    slider_prev = Slider(ax_prev, 'Prev Frame', 0, gif_frames.shape[0]-1, valinit=initial_prev, valstep=1)
    slider_curr = Slider(ax_curr, 'Curr Frame', 0, gif_frames.shape[0]-1, valinit=initial_curr, valstep=1)
    
    # Add text boxes for direct input
    ax_text_prev = plt.axes([0.1, 0.05, 0.1, 0.03])
    ax_text_curr = plt.axes([0.8, 0.05, 0.1, 0.03])
    
    text_box_prev = TextBox(ax_text_prev, 'Prev Frame', initial=str(initial_prev))
    text_box_curr = TextBox(ax_text_curr, 'Curr Frame', initial=str(initial_curr))
    
    # Update function for sliders and text boxes
    def update(val):
        prev_index = int(slider_prev.val)
        curr_index = int(slider_curr.val)
        
        img_prev.set_data(gif_frames[prev_index])
        ax1.set_title(f'Previous Frame ({prev_index})')
        img_curr.set_data(gif_frames[curr_index])
        ax2.set_title(f'Current Frame ({curr_index})')
        
        delta_unwrapped = unwrapped_phases[curr_index] - unwrapped_phases[prev_index]
        img_delta_unwrapped.set_data(delta_unwrapped)
        img_delta_unwrapped.set_clim(delta_unwrapped.min(), delta_unwrapped.max())
        
        delta_wrapped = np.angle(np.exp(1j*(wrapped_phases[curr_index] - wrapped_phases[prev_index])))
        img_delta_wrapped.set_data(delta_wrapped)
        
        fig.canvas.draw_idle()

    # Function to update slider from text box input
    def update_from_text(text, slider):
        try:
            val = int(text)
            if 0 <= val < gif_frames.shape[0]:
                slider.set_val(val)
        except ValueError:
            pass
    
    # Connect sliders and text boxes to update functions
    slider_prev.on_changed(update)
    slider_curr.on_changed(update)
    text_box_prev.on_submit(lambda text: update_from_text(text, slider_prev))
    text_box_curr.on_submit(lambda text: update_from_text(text, slider_curr))
    
    plt.show()

if __name__ == "__main__":
    main()

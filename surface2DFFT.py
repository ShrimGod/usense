import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from matplotlib.widgets import Slider, TextBox
from PIL import Image

# Function to load a GIF and return the frames as numpy arrays
def load_gif(filepath):
    gif = Image.open(filepath)
    frames = []
    try:
        while True:
            frame = np.array(gif.convert('L'))  # Convert to grayscale
            frames.append(frame)
            gif.seek(gif.tell() + 1)  # Move to the next frame
    except EOFError:
        pass  # End of GIF reached
    return frames

# Function to compute the FFT and shift it
def compute_fft(frames):
    return [fftshift(fft2(frame)) for frame in frames]

# Function to compute unwrapped phase
def compute_unwrapped_phase(fft_frame):
    real = np.real(fft_frame)
    imag = np.imag(fft_frame)
    phase = np.arctan2(imag, real)
    return np.unwrap(phase)

# Main function
def main():
    # File path to GIF
    filepath = r"C:\Users\Douglas Fleetwood\Desktop\uSense\0.5mL_4mm.gif"
    
    # Load the GIF frames
    gif_frames = load_gif(filepath)
    print(f"Number of frames in the GIF: {len(gif_frames)}")
    
    # Compute FFT for each frame in the GIF
    fft_frames = compute_fft(gif_frames)
    
    # Compute unwrapped phase for each FFT frame
    unwrapped_phases = [compute_unwrapped_phase(fft_frame) for fft_frame in fft_frames]
    
    # Initial frame indices
    initial_prev = 0
    initial_curr = 1
    
    # Calculate initial delta phi
    delta = np.abs(unwrapped_phases[initial_curr] - unwrapped_phases[initial_prev])
    
    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.4)
    
    # Determine common vmin and vmax for consistent scaling
    vmin, vmax = 0, np.max(gif_frames)
    
    # Display the initial frames and delta phi in grayscale with consistent scaling
    img_prev = ax1.imshow(gif_frames[initial_prev], cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Previous GIF Frame')
    plt.colorbar(img_prev, ax=ax1)
    
    img_curr = ax2.imshow(gif_frames[initial_curr], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title('Current GIF Frame')
    plt.colorbar(img_curr, ax=ax2)
    
    img_delta = ax3.imshow(delta, cmap='gray')
    ax3.set_title('Delta Phi (Change in Unwrapped Phase)')
    plt.colorbar(img_delta, ax=ax3)
    
    # Add sliders for selecting frames
    ax_prev = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_curr = plt.axes([0.1, 0.15, 0.8, 0.03])
    
    slider_prev = Slider(ax_prev, 'Prev Frame', 0, len(gif_frames)-1, valinit=initial_prev, valstep=1)
    slider_curr = Slider(ax_curr, 'Curr Frame', 0, len(gif_frames)-1, valinit=initial_curr, valstep=1)
    
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
        img_curr.set_data(gif_frames[curr_index])
        
        delta = np.abs(unwrapped_phases[curr_index] - unwrapped_phases[prev_index])
        img_delta.set_data(delta)
        
        fig.canvas.draw_idle()

    # Function to update slider from text box input
    def update_from_text(text, slider):
        try:
            val = int(text)
            if 0 <= val < len(gif_frames):
                slider.set_val(val)
        except ValueError:
            pass
    
    # Connect sliders and text boxes to update functions
    slider_prev.on_changed(update)
    slider_curr.on_changed(update)
    text_box_prev.on_submit(lambda text: update_from_text(text, slider_prev))
    text_box_curr.on_submit(lambda text: update_from_text(text, slider_curr))
    
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
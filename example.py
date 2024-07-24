from fat_llama.audio_fattener.feed import upscale_mp3_to_flac

# Example call to the method
upscale_mp3_to_flac(
    input_file_path='input_test.mp3',
    output_file_path_processed='output_test.flac',
    max_iterations=800,
    threshold_value=0.6,
    gain_factor=132.8,
    reduction_profile=[
        (1, 140, -72.5),
        (1000, 10000, -216),
    ],
    lowcut=1.0,
    highcut=80000.0,
    target_bitrate_kbps=1400,
    output_file_path_no_processing=None,
    toggle_wiener_filter=False,
    toggle_normalize=True,       # Toggle normalization
    toggle_equalize=False,        # Toggle equalization
    toggle_scale_amplitude=False, # Toggle amplitude scaling
    toggle_gain_reduction=False   # Toggle gain reduction
)
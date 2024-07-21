from fat_llama.audio_fattener.feed import upscale_mp3_to_flac

# Example call to the method
upscale_mp3_to_flac(
    input_file_path='input_test.mp3',
    output_file_path_processed='output_test.flac',
    max_iterations=1000,
    threshold_value=0.6,
    gain_factor=32.8,
    reduction_profile=[
        (5, 140, -38.4),
        (1000, 10000, 36.4),
    ],
    lowcut=5.0,
    highcut=150000.0,
    target_bitrate_kbps=1400,
    output_file_path_no_processing=None,
    use_wiener_filter=False
)

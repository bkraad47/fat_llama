from fat_llama.audio_fattener.feed import upscale

# Example call to the method
upscale(
    input_file_path='input_test.mp3',
    output_file_path='output_test.flac',
    source_format='mp3',
    target_format='flac',
    max_iterations=1000,
    threshold_value=0.6,
    target_bitrate_kbps=1400
)

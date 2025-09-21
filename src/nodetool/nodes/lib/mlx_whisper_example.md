# MLX Whisper Example

This example shows how to use the MLX Whisper node for speech-to-text transcription.

## Workflow

1. Load an audio file using a LoadAudioFile node
2. Connect the audio output to the MLXWhisper node
3. Configure the MLXWhisper node with the desired model and settings
4. Run the workflow to get the transcribed text

## Node Configuration

```python
# Example configuration for MLXWhisper node
mlx_whisper_node = MLXWhisper(
    model=MLXWhisper.Model.TINY_EN,  # or any other available model
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    word_timestamps=False
)
```

## Available Models

- TINY_EN: "mlx-community/whisper-tiny.en-mlx" (English only, fastest)
- TINY: "mlx-community/whisper-tiny-mlx" (Multilingual)
- BASE_EN: "mlx-community/whisper-base.en-mlx" (English only)
- BASE: "mlx-community/whisper-base-mlx" (Multilingual)
- SMALL_EN: "mlx-community/whisper-small.en-mlx" (English only)
- SMALL: "mlx-community/whisper-small-mlx" (Multilingual)
- MEDIUM_EN: "mlx-community/whisper-medium.en-mlx" (English only)
- MEDIUM: "mlx-community/whisper-medium-mlx" (Multilingual)
- LARGE_V3: "mlx-community/whisper-large-v3-mlx" (Multilingual, most accurate)

## Output

The node returns:
- `text`: The full transcribed text
- `segments`: List of segment-level details including timestamps and text

## Requirements

- Apple Silicon Mac (M1, M2, etc.) for optimal performance
- mlx-whisper package installed
# ComfyUI NVEncC FRUC

ComfyUI output nodes for NVIDIA Optical Flow frame-rate up-conversion through NVEncC and NvOFFRUC.

`Save Video with NVEncC FRUC` accepts ComfyUI's native `VIDEO` type directly. A video loaded with `Load Video` is passed to NVEncC as its original file instead of being decoded into one large `IMAGE` tensor first. Generated `IMAGE` batches remain supported through a streamed Y4M path.

## Nodes

### Save Video with NVEncC FRUC

- Accepts `VIDEO` or `IMAGE` through one backward-compatible `video / images` socket.
- Uses NVEncC hardware decoding by default, with software decoding available for unsupported source codecs.
- Produces `2N-1` frames at twice the input frame rate while preserving playback duration.
- Detects FPS from `VIDEO`; the FPS widget applies only to `IMAGE` batches.
- Copies embedded source audio when no replacement audio is connected.
- Passes native ComfyUI `Trim Video` ranges to NVEncC without extracting frames.
- Encodes AV1, HEVC, or H.264 into MP4 or MKV.

### Replace Video Tail with NVEncC FRUC

Replaces the final frames of an `IMAGE` batch with an NvOFFRUC-smoothed transition while preserving the batch length.

## Requirements

- Windows
- A supported NVIDIA GPU and current NVIDIA driver
- ComfyUI with the native `VIDEO` API
- [NVEncC](https://github.com/rigaya/NVEnc) built with NvOFFRUC support

NVEncC and its binaries are not included in this repository. The node searches for the executable in this order:

1. The advanced `nvencc_path` node setting
2. The `NVENCC_PATH` environment variable
3. `ComfyUI/tools/NVEncC/NVEncC64.exe`
4. `ComfyUI/tools/NVEncC64.exe`
5. The system `PATH`

The direct video path uses NVEncC's `avhw` or `avsw` reader. Available codecs depend on the installed NVEncC build; typical builds support H.264/AVC, HEVC, MPEG-1/2/4, VP8, VP9, VC-1, and AV1.

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/Molinist/ComfyUI-NVEncC-FRUC.git
```

Place NVEncC in `ComfyUI/tools/NVEncC`, set `NVENCC_PATH`, or select it with the advanced node input. Restart ComfyUI and reload the browser.

## Basic workflow

```text
Load Video (VIDEO) --> Save Video with NVEncC FRUC (video / images)
```

For generated frames, connect an `IMAGE` batch instead and set its source FPS on the node.

## Tests

Run the tests from a ComfyUI Python environment with the ComfyUI repository on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "C:\path\to\ComfyUI"
python -m pytest tests -q
python -m ruff check .
```

## License

The node code is available under the MIT License. NVEncC and NVIDIA components have their own licenses.

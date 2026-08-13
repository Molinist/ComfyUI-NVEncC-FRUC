# ComfyUI NVEncC Suite

Composable ComfyUI nodes for NVEncC encoding, NVIDIA Optical Flow frame-rate up-conversion, and GPU video sharpening.

The save node accepts ComfyUI's native `VIDEO` type directly. A video from `Load Video` goes to NVEncC as its original file instead of first becoming one large `IMAGE` tensor. Generated `IMAGE` batches remain supported through a streamed Y4M path.

## Suite nodes

- **Save Video with NVEncC** encodes a native `VIDEO` or `IMAGE` batch and accepts an optional filter chain.
- **NVEncC Frame Double (FRUC)** inserts one generated frame between each source-frame pair, doubling FPS without changing duration.
- **NVEncC Sharpen (CAS)** applies restrained contrast-adaptive sharpening.
- **NVEncC Sharpen (Unsharp)** provides direct, visibly aggressive edge enhancement.
- **NVEncC Sharpen (EdgeLevel)** strengthens detected edges with separate dark- and bright-side emphasis.
- **NVEncC Sharpen (MSharpen)** sharpens detected edges with dark-area and compression-block protection.
- **NVEncC Sharpen (DetailSharpen)** enhances fine texture with nonlinear damping controls.

Each processing feature is its own node. Leaving a filter node out disables that feature completely. Connect filter nodes through their `filters` sockets, then connect the final chain to `Save Video with NVEncC`.

Filter nodes build a deferred NVEncC recipe; they do not encode or materialize another video by themselves. The save node executes the complete chain once, avoiding an intermediate encode for every filter. Frame doubling is off unless `NVEncC Frame Double (FRUC)` is present in the connected chain. The save node's FPS widget only sets the frame rate for `IMAGE` batches and does not enable FRUC; native `VIDEO` input keeps its source FPS.

```text
Load Video (VIDEO) --------------------------------> Save Video with NVEncC (video / images)
NVEncC Sharpen (MSharpen) -> NVEncC Frame Double -> Save Video with NVEncC (filters)
```

Filters execute in NVEncC's supported order: Unsharp, EdgeLevel, MSharpen, CAS, DetailSharpen, then FRUC. Wiring the same filter type more than once makes the later node replace that stage rather than applying it twice.

The former **Save Video with NVEncC (Legacy All-in-One)** remains registered under `video/nvencc/legacy` so existing workflows, embedded media metadata, and long-lived browser tabs continue to load. Use the suite nodes for new workflows.

**Replace Video Tail with NVEncC FRUC** remains available for replacing the final frames of an `IMAGE` batch with an NvOFFRUC-smoothed transition while preserving batch length.

## Save behavior

- Hardware video decoding is the default; software decoding is available for source codecs unsupported by the GPU.
- `VIDEO` input keeps source FPS, trim ranges, and embedded audio. Connecting audio replaces the source audio.
- The FPS widget applies only to `IMAGE` batches.
- AV1, HEVC, and H.264 output are supported in MP4 or MKV.
- With no filter chain connected, the save node performs encoding only.

## Sharpening guide

- **CAS:** subtle and adaptive. Start around `0.2-0.4`; enable HDR for PQ/HLG input, and enable chroma only when colored detail needs sharpening.
- **Unsharp:** the clearest before/after demonstration. Start at radius `3`, weight `0.5`, threshold `10`, then raise weight carefully.
- **EdgeLevel:** direct outline enhancement with separate dark and bright edge controls. Start at strength `5`, threshold `20`, and leave black/white at `0` until needed.
- **MSharpen:** a strong general-purpose choice for compressed video. Its protection controls can reduce dark-noise and block enhancement.
- **DetailSharpen:** emphasizes fine texture. Increase damping if grain or compression texture becomes prominent.

Strong filters or stacked sharpen stages can create halos, ringing, block emphasis, grain, or temporal shimmer.

## Requirements

- Windows
- A supported NVIDIA GPU and current NVIDIA driver
- ComfyUI with the native `VIDEO` API
- [NVEncC](https://github.com/rigaya/NVEnc); NvOFFRUC support is needed only for FRUC nodes

NVEncC and its binaries are not included. The save node searches in this order:

1. The advanced `nvencc_path` setting
2. The `NVENCC_PATH` environment variable
3. `ComfyUI/tools/NVEncC/NVEncC64.exe`
4. `ComfyUI/tools/NVEncC64.exe`
5. The system `PATH`

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/Molinist/ComfyUI-NVEncC-FRUC.git "NVEncC Suite"
```

ComfyUI derives the source badge from the installed custom-node directory, not the package display name. The explicit destination above therefore gives these nodes the `NVEncC Suite` badge in search. For an existing install, stop ComfyUI and rename its folder to `NVEncC Suite` (or reinstall with the command above).

Place NVEncC in `ComfyUI/tools/NVEncC`, set `NVENCC_PATH`, or select it with the advanced save-node input. Restart ComfyUI and hard-refresh the browser once after installation.

The legacy all-in-one node stores widget values by field name and migrates earlier positional layouts, including the temporary layout that could display `NaN` for sharpening. Future suite features are added as separate nodes, so updating the suite does not shift settings in an open workflow tab.

## Tests

Run from a ComfyUI Python environment with the ComfyUI repository on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "C:\path\to\ComfyUI"
python -m pytest tests -q
node --test tests/widget_compat.test.mjs
python -m ruff check .
```

## License

The node code is available under the MIT License. NVEncC and NVIDIA components have their own licenses.

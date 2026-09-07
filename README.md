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

- **Windows x64** and a working, up-to-date ComfyUI installation with the native `VIDEO` and `comfy_api.latest` APIs. These nodes do not require patched ComfyUI core files or another custom-node suite.
- **NVIDIA GPU with NVENC** for encoding. FRUC additionally requires NVIDIA Optical Flow support (Turing / RTX 20-series or newer, subject to the GPU's capabilities). AMD, Intel and CPU-only systems cannot run this backend.
- **A current NVIDIA driver compatible with your GPU and NVEncC build.** Upstream lists 528.24 as the FRUC feature's historical minimum; newer NVEncC builds can require a newer driver, so that number is not a sufficient installation target.
- **NVEncC for Windows x64**, installed separately as described below. The Python node repository does not include or download the executable or its DLLs.

See [upstream FRUC requirements](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.ja.md#--vpp-fruc-param1value1param2value2).

### Compatibility status

The setup links below were checked on **2026-09-07**. NVEncC **9.33 x64** is the release linked here; this is a setup reference, not a claimed minimum version or a clean-machine certification. Current ComfyUI source exposes the APIs used by this suite, but a minimum ComfyUI version and a complete tested GPU/driver/version matrix have not been established. Older NVEncC builds may lack newer sharpening options.

## Installation for Windows

### 1. Install the nodes

Find the folder containing ComfyUI's `main.py` and `custom_nodes`. In the Windows portable distribution, this is the inner `ComfyUI` folder, not the folder containing `run_nvidia_gpu.bat`.

With [Git for Windows](https://git-scm.com/downloads/win) installed, open PowerShell and run these commands, replacing the example path:

```powershell
cd "C:\path\to\ComfyUI\custom_nodes"
git clone https://github.com/Molinist/ComfyUI-NVEncC-FRUC.git "NVEncC Suite"
```

Without Git, use this repository's **Code > Download ZIP**, extract it, and place its contents in `custom_nodes/NVEncC Suite`. The file `__init__.py` must be directly inside that folder, not inside a second nested repository folder.

Install only one copy of this suite. Another copy, including a local `comfyui_nvencc` package, can register the same node IDs. The folder name `NVEncC Suite` gives the nodes that source badge in search; renaming an existing installation is optional.

### 2. Install NVEncC and keep its DLLs together

Download **[NVEncC_9.33_x64.7z](https://github.com/rigaya/NVEnc/releases/download/9.33/NVEncC_9.33_x64.7z)** from the [official NVEncC releases](https://github.com/rigaya/NVEnc/releases). Choose the **NVEncC x64** archive, not Win32, an AviUtl installer, or the source-code archive.

Extract the **whole archive**, using [7-Zip](https://www.7-zip.org/) if needed, into `ComfyUI/tools/NVEncC`. Create those folders if they do not exist. Ensure the executable is directly at:

```text
ComfyUI/
  main.py
  custom_nodes/
    NVEncC Suite/
      __init__.py
      nodes_nvencc_fruc.py
      web/
  tools/
    NVEncC/
      NVEncC64.exe
      NVEncNVOFFRUC.dll
      NvOFFRUC.dll
      ...other files from the NVEncC archive...
```

Do not copy only `NVEncC64.exe`. Keep all supplied DLLs alongside it. Upstream's x64 release packaging includes the FRUC runtime DLLs; a separate Optical Flow SDK installation is not the normal setup path. If those files are missing, re-extract the complete official x64 package. See [upstream release packaging](https://github.com/rigaya/NVEnc/blob/master/.github/workflows/build_releases.yml).

If NVEncC is already installed elsewhere, keep it there and set the save node's advanced **`nvencc_path`** to the full executable path, such as `D:\VideoTools\NVEncC\NVEncC64.exe`. No quotes are needed inside the widget.

Executable discovery checks, in order:

1. The advanced `nvencc_path` setting
2. The `NVENCC_PATH` environment variable (full executable path)
3. `ComfyUI/tools/NVEncC/NVEncC64.exe`
4. `ComfyUI/tools/NVEncC64.exe`
5. The system `PATH`

### 3. Check NVEncC before opening a workflow

In PowerShell, from the ComfyUI folder:

```powershell
cd "C:\path\to\ComfyUI"
& ".\tools\NVEncC\NVEncC64.exe" --version
& ".\tools\NVEncC\NVEncC64.exe" --check-hw
& ".\tools\NVEncC\NVEncC64.exe" --check-features
```

Use your actual executable path if you installed it elsewhere. `--version` should run without a missing-DLL error and report `nvof fruc : yes` for a FRUC-enabled build. The hardware checks should identify your NVIDIA GPU and supported encoding features. These checks do not execute interpolation; the short workflow below verifies that separately. [NVEncC diagnostic options](https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md#display-options)

### 4. Restart ComfyUI

Restart ComfyUI, then hard-refresh the browser once to load the extension's JavaScript. Search for **Save Video with NVEncC**. Use the normal suite node for a new workflow; the legacy all-in-one node remains for existing workflows.

## First video: encode, then double FPS

Use a short **SDR** clip with even dimensions, such as 1280 x 720, for the first check.

1. Add ComfyUI's **Load Video** node and load the clip.
2. Add **Save Video with NVEncC** and connect `Load Video`'s `VIDEO` output to its `video / images` input.
3. Set `container` to `mp4`, `codec` to `h264`, `preset` to `p4`, and `quality` to `20`. Leave `filters` and `audio` disconnected. Leave `nvencc_path` blank if you used the standard folder.
4. Run the workflow. The preview should appear and the file should be saved under ComfyUI's `output` folder, using the node's filename prefix. This checks encoding and keeps the source FPS and embedded audio.
5. Add **NVEncC Frame Double (FRUC)**. Connect its `filters` output to the save node's `filters` input, then run again. A 24 FPS source should become 48 FPS with approximately the same duration and its source audio.

```text
Load Video ── VIDEO ─────────────────> Save Video with NVEncC: video / images
NVEncC Frame Double (FRUC) ─ filters ─> Save Video with NVEncC: filters
```

The save node's **FPS widget does not change native VIDEO timing or enable interpolation**. It specifies the source FPS only when you connect an `IMAGE` batch. FRUC requires its filter node. For `IMAGE` batches, connect `AUDIO` separately if needed; FRUC needs at least two frames.

Once that works, optionally connect a sharpening node before FRUC through the `filters` sockets. Filters carry a processing recipe, not video frames.

**Output limits:** the current save implementation requests **8-bit BT.709** output. It is not an HDR-preserving export or an HDR-to-SDR tone-mapping workflow. CAS's `HDR source` control only changes that filter's response. The suite does not upscale resolution, and FRUC may produce artifacts around occlusion or complex motion.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Nodes do not appear / import error | Check the folder layout and ComfyUI startup log. Update ComfyUI and its dependencies using your distribution's normal updater, then restart. Errors mentioning `comfy_api.latest`, `MultiType`, or video APIs can indicate an older core. |
| Missing `av`, `torch`, or `numpy` | Repair ComfyUI's dependencies in the Python environment that runs ComfyUI, not an unrelated system Python. The suite uses ComfyUI's existing packages. |
| `NVEncC was not found` | Check that `NVEncC64.exe` is directly in `tools/NVEncC`, or provide its full path in `nvencc_path`. Avoid an extra nested extraction folder. |
| Missing DLL / FRUC initialization failure | Re-extract the complete official x64 NVEncC archive, retaining all DLLs. Check `NVEncNVOFFRUC.dll` and `NvOFFRUC.dll`, the NVIDIA driver, and GPU support. If ordinary encoding works but FRUC fails, focus on the Optical Flow runtime and hardware. |
| Unsupported codec / encoder initialization error | Start with `h264`; AV1 encoding is not available on every NVIDIA GPU. Run `--check-features` and check the driver requirement of your NVEncC release. |
| Hardware decoding fails for the input | Change `input_decoder` to `software`. Encoding and FRUC still require the NVIDIA GPU. |
| Unknown sharpening option | Update NVEncC; loading the Python nodes does not prove an older executable supports every filter option. |
| Width or height must be even | Resize or crop to even dimensions before this node; its YUV420 path rejects odd dimensions. |
| FPS stays unchanged | Connect the FRUC filter. Changing the FPS widget has no effect on native `VIDEO` input. |
| Audio fails to mux into MP4 | Try `mkv` for source-audio passthrough, or connect a replacement `AUDIO` input, which is encoded as AAC. |
| Duplicate or unexpected nodes | Keep one installed copy; remove the duplicate installation rather than installing over it. |

If a run fails, include the **full NVEncC error from the ComfyUI log**, GPU model, driver version, ComfyUI version, NVEncC `--version` output, and the failing node's settings in an issue. A successful import alone does not verify encoding or FRUC.

## Updating

For a Git installation, run this from the installed suite folder, then restart ComfyUI:

```powershell
git pull --ff-only
```

Hard-refresh the browser once when frontend JavaScript changes. Existing legacy widget values are migrated by the extension; a refresh is not a substitute for workflow compatibility. Update the external NVEncC folder separately when newer filter options require it.

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

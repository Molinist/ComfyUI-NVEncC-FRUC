import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import torch
from typing_extensions import override

import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, io, ui


DATE_FORMAT_PARTS = {
    "yyyy": "%Y",
    "yy": "%y",
    "MM": "%m",
    "dd": "%d",
    "HH": "%H",
    "hh": "%I",
    "mm": "%M",
    "ss": "%S",
}
DATE_TOKEN = re.compile(r"%date:([^%]+)%")
DATE_FORMAT_PART = re.compile("|".join(DATE_FORMAT_PARTS))
NVEncCFilters = io.Custom("NVENCC_FILTERS")
NVENCC_FILTER_ARGUMENTS = (
    ("ngx_vsr", "--vpp-resize"),
    ("unsharp", "--vpp-unsharp"),
    ("edgelevel", "--vpp-edgelevel"),
    ("msharpen", "--vpp-msharpen"),
    ("cas", "--vpp-cas"),
    ("detailsharpen", "--vpp-detailsharpen"),
    ("warpsharp", "--vpp-warpsharp"),
    ("fruc", "--vpp-fruc"),
)
GIF_DITHER_OPTIONS = ("sierra2_4a", "bayer", "heckbert", "floyd_steinberg", "sierra2", "sierra3", "burkes", "atkinson", "none")
GIF_DITHER_TOOLTIP = "Palette dithering: sierra2_4a is a fast diffuse default; bayer uses an ordered pattern; heckbert is simple error diffusion; floyd_steinberg is classic error diffusion; sierra2, sierra3, burkes, and atkinson use their named diffusion kernels; none uses the nearest palette color without dithering."
GIF_BAYER_TOOLTIP = "For bayer dithering only: 0 makes the pattern most visible; 5 reduces visible patterning but can increase banding. Other dither modes ignore this value."


def expand_date_tokens(filename_prefix, now=None):
    now = now or datetime.now()

    def replace_date(match):
        date_format = DATE_FORMAT_PART.sub(lambda part: DATE_FORMAT_PARTS[part.group()], match.group(1))
        return now.strftime(date_format)

    return DATE_TOKEN.sub(replace_date, filename_prefix)


def find_nvencc(explicit_path=""):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("NVENCC_PATH")
    if env_path:
        candidates.append(env_path)

    names = ("NVEncC64.exe", "NVEncC.exe", "NVEncC64", "NVEncC")
    search_dirs = (
        os.path.join(folder_paths.base_path, "tools", "NVEncC"),
        os.path.join(folder_paths.base_path, "tools"),
    )
    for directory in search_dirs:
        candidates.extend(os.path.join(directory, name) for name in names)
    candidates.extend(path for name in names if (path := shutil.which(name)))

    for path in candidates:
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("NVEncC was not found. Set NVENCC_PATH, provide nvencc_path, or place NVEncC64.exe in ComfyUI/tools/NVEncC.")


def find_ffmpeg(explicit_path=""):
    path = explicit_path or os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if path and (resolved := shutil.which(path)):
        return resolved
    raise FileNotFoundError("FFmpeg was not found. Set FFMPEG_PATH, provide ffmpeg_path, or add ffmpeg to PATH.")


def build_command(executable, output_path, codec, preset, quality, audio_path=None, fruc=True, lossless=False,
                  input_path=None, input_decoder="hardware", copy_audio=False, trim=(0.0, 0.0, 1.0), cas_sharpness=0.0,
                  unsharp=None, msharpen=None, detailsharpen=None, filters=None):
    command = [executable]
    if input_path is None:
        command.extend(("--y4m", "-i", "-"))
    else:
        command.extend(("--avhw" if input_decoder == "hardware" else "--avsw", "-i", os.fspath(input_path)))
        start_time, duration, frame_rate = trim
        if duration:
            start_frame = round(start_time * frame_rate)
            frame_count = max(1, round(duration * frame_rate))
            command.extend(("--trim", f"{start_frame}:{start_frame + frame_count - 1}"))
        elif start_time:
            command.extend(("--seek", f"{start_time:.9g}"))
    command.extend((
        "--codec", codec,
        "--preset", preset,
        "--qvbr", str(quality),
        "--output-depth", "8",
        "--colormatrix", "bt709",
        "--colorprim", "bt709",
        "--transfer", "bt709",
    ))
    if fruc:
        command.extend(("--vpp-fruc", "double" if fruc is True else str(fruc)))
    if cas_sharpness > 0.0:
        command.extend(("--vpp-cas", f"sharpness={cas_sharpness:g}"))
    if unsharp is not None:
        command.extend(("--vpp-unsharp", unsharp))
    if msharpen is not None:
        command.extend(("--vpp-msharpen", msharpen))
    if detailsharpen is not None:
        command.extend(("--vpp-detailsharpen", detailsharpen))
    filters = filters or {}
    unknown_filters = set(filters).difference(name for name, _ in NVENCC_FILTER_ARGUMENTS)
    if unknown_filters:
        raise ValueError(f"Unsupported NVEncC filters: {', '.join(sorted(unknown_filters))}")
    for name, argument in NVENCC_FILTER_ARGUMENTS:
        if name in filters:
            parameters = filters[name]
            if name == "ngx_vsr":
                command.extend(("--output-res", parameters["output_res"]))
                parameters = parameters["parameters"]
            command.extend((argument, parameters))
    if lossless:
        command.append("--lossless")
    if audio_path is not None:
        command.extend(("--audio-source", f"{audio_path}:codec=aac;bitrate=192"))
    elif copy_audio:
        command.append("--audio-copy")
    command.extend(("--output", output_path))
    return command


def write_audio_wav(audio, path):
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] < 1:
        raise ValueError("Audio waveform must have shape [batch, channels, samples]")
    waveform = waveform[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    samples = (waveform.transpose(0, 1).contiguous().numpy() * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as output:
        output.setnchannels(waveform.shape[0])
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(samples.tobytes())


def _write_plane(output, plane, width, height):
    data = bytes(plane)
    line_size = plane.line_size
    if line_size == width:
        output.write(data)
        return
    for y in range(height):
        start = y * line_size
        output.write(data[start:start + width])


def write_y4m(images, fps, output, progress=None):
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Images must have shape [frames, height, width, channels] with at least three channels")
    frame_count, height, width, _ = images.shape
    if frame_count < 1:
        raise ValueError("Save Video with NVEncC requires at least one IMAGE frame")
    if width % 2 or height % 2:
        raise ValueError("NVEncC YUV420 input requires even image width and height")

    rate = Fraction(str(float(fps))).limit_denominator(100000)
    output.write(f"YUV4MPEG2 W{width} H{height} F{rate.numerator}:{rate.denominator} Ip A0:0 C420jpeg\n".encode())
    for image in images:
        rgb = image[..., :3].detach().mul(255.0).clamp(0.0, 255.0).to(device="cpu", dtype=torch.uint8).numpy()
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        output.write(b"FRAME\n")
        _write_plane(output, frame.planes[0], width, height)
        _write_plane(output, frame.planes[1], width // 2, height // 2)
        _write_plane(output, frame.planes[2], width // 2, height // 2)
        if progress is not None:
            progress.update(1)


def run_nvencc(command, images, fps):
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            write_y4m(images, fps, process.stdin, comfy.utils.ProgressBar(images.shape[0]))
            process.stdin.close()
            return_code = process.wait()
        except BrokenPipeError:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            return_code = process.wait()
        except BaseException:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

        stderr.seek(0)
        message = stderr.read().decode(errors="replace").strip()
        if return_code != 0:
            if len(message) > 4000:
                message = message[-4000:]
            raise RuntimeError(f"NVEncC failed with exit code {return_code}:\n{message}")


def run_nvencc_video(command):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    message = result.stderr.decode(errors="replace").strip()
    if result.returncode != 0 or "encoded 0 frames" in message:
        if len(message) > 4000:
            message = message[-4000:]
        raise RuntimeError(f"NVEncC failed with exit code {result.returncode}:\n{message}")


def run_ffmpeg(command, capture_stdout=False):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()[-4000:]
        raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}:\n{message}")
    return result.stdout


def get_video_input_path(video, temp_dir):
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)

    input_path = os.path.join(temp_dir, "input.mp4")
    source.seek(0)
    with open(input_path, "wb") as output:
        shutil.copyfileobj(source, output)
    return input_path


def encode_gif(video, dither, bayer_scale, ffmpeg_path, output_path=None):
    if dither not in GIF_DITHER_OPTIONS:
        raise ValueError(f"Unsupported GIF dithering: {dither}")
    if not isinstance(bayer_scale, int) or not 0 <= bayer_scale <= 5:
        raise ValueError("GIF Bayer scale must be an integer from 0 to 5")
    executable = find_ffmpeg(ffmpeg_path)
    with tempfile.TemporaryDirectory(prefix="comfy_nvencc_gif_") as temp_dir:
        input_path = get_video_input_path(video, temp_dir)
        start_time, duration = video.get_active_trim_window()
        source_args = ["-ss", f"{start_time:.9g}"] if start_time else []
        source_args.extend(("-i", input_path))
        trim_args = ["-t", f"{duration:.9g}"] if duration else []
        palette_path = os.path.join(temp_dir, "palette.png")
        run_ffmpeg([executable, "-hide_banner", "-loglevel", "error", *source_args, "-an", *trim_args,
                    "-vf", "palettegen=reserve_transparent=on", "-frames:v", "1", "-y", palette_path])
        paletteuse = f"paletteuse=dither={dither}"
        if dither == "bayer":
            paletteuse += f":bayer_scale={bayer_scale}"
        command = [executable, "-hide_banner", "-loglevel", "error", *source_args, "-i", palette_path,
                   "-filter_complex", f"[0:v][1:v]{paletteuse}", "-an", *trim_args, "-fps_mode", "passthrough", "-loop", "0"]
        if output_path is None:
            return run_ffmpeg(command + ["-f", "gif", "-"], capture_stdout=True)
        run_ffmpeg(command + ["-y", output_path])


def mux_gif_audio(video, gif_bytes, ffmpeg_path):
    with tempfile.TemporaryDirectory(prefix="comfy_nvencc_gif_audio_") as temp_dir:
        input_path = get_video_input_path(video, temp_dir)
        with av.open(input_path) as container:
            if not container.streams.audio:
                return gif_bytes

        gif_path = os.path.join(temp_dir, "video.gif")
        with open(gif_path, "wb") as output:
            output.write(gif_bytes)
        start_time, duration = video.get_active_trim_window()
        command = [find_ffmpeg(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-i", gif_path]
        if start_time:
            command.extend(("-ss", f"{start_time:.9g}"))
        command.extend(("-i", input_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1",
                        "-pix_fmt", "yuv444p", "-c:a", "copy", "-shortest"))
        if duration:
            command.extend(("-t", f"{duration:.9g}"))
        return run_ffmpeg(command + ["-f", "matroska", "-"], capture_stdout=True)


def select_bridge_frames(decoded, start_frame, end_frame, frame_count, curve=1.0):
    if len(decoded) < frame_count:
        raise RuntimeError(f"NVEncC FRUC produced {len(decoded)} bridge frames; need at least {frame_count}")
    positions = torch.linspace(0.0, 1.0, frame_count).pow(curve).mul_(len(decoded) - 1)
    indices = positions.round().to(dtype=torch.int64)
    bridge = decoded[indices].clone()
    bridge[0] = start_frame
    bridge[-1] = end_frame
    return bridge


def add_nvencc_filter(filters, name, parameters):
    configured = dict(filters or {})
    configured[name] = parameters
    return configured


def resolve_nvencc_filters(filters, width, height):
    resolved = dict(filters or {})
    config = resolved.get("ngx_vsr")
    if config is None:
        return resolved, width, height

    target_dimension = config["target_dimension"]
    if target_dimension not in ("height", "width"):
        raise ValueError("NGX VSR target dimension must be height or width")

    target_size = config["target_size"]
    if target_size < 2 or target_size % 2:
        raise ValueError("NGX VSR target size must be an even number of pixels")

    source_size = width if target_dimension == "width" else height
    if target_size <= source_size:
        raise ValueError(f"NGX VSR target {target_dimension} must be larger than the source {target_dimension}")

    if target_dimension == "width":
        output_width = target_size
        output_height = max(2, round(height * target_size / width / 2) * 2)
    else:
        output_height = target_size
        output_width = max(2, round(width * target_size / height / 2) * 2)

    resolved["ngx_vsr"] = {
        "parameters": f"algo=ngx-vsr,vsr-quality={config['quality']}",
        "output_res": f"{output_width}x{output_height}",
    }
    return resolved, output_width, output_height


def save_video_nvencc(images, fps, filename_prefix, container, codec, preset, quality, nvencc_path, filters=None, audio=None, input_decoder="hardware"):
    filters = dict(filters or {})
    image_input = isinstance(images, torch.Tensor)
    gif_input = False
    if image_input:
        minimum_frames = 2 if "fruc" in filters else 1
        if images.ndim != 4 or images.shape[0] < minimum_frames:
            raise ValueError(f"Save Video with NVEncC requires at least {minimum_frames} IMAGE frame{'s' if minimum_frames > 1 else ''}")
        height, width = images.shape[1:3]
    else:
        width, height = images.get_dimensions()
        source = images.get_stream_source()
        gif_input = isinstance(source, (str, os.PathLike)) and os.path.splitext(source)[1].lower() == ".gif"
        if isinstance(source, BytesIO):
            with av.open(source) as source_container:
                gif_input = source_container.streams.video[0].codec_context.name in ("gif", "ffv1")
    needs_gif_padding = gif_input and (width % 2 or height % 2)
    if width % 2 or height % 2:
        if not gif_input:
            raise ValueError("Save Video with NVEncC requires even video width and height")
        width += width % 2
        height += height % 2

    filters, output_width, output_height = resolve_nvencc_filters(filters, width, height)

    executable = find_nvencc(nvencc_path)
    expanded_prefix = expand_date_tokens(filename_prefix)
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        expanded_prefix, folder_paths.get_output_directory(), output_width, output_height
    )
    suffix = "fruc" if "fruc" in filters else "vsr" if "ngx_vsr" in filters else "nvencc"
    file = f"{filename}_{counter:05}_{suffix}.{container}"
    output_path = os.path.join(full_output_folder, file)

    try:
        with tempfile.TemporaryDirectory(prefix="comfy_nvencc_") as temp_dir:
            audio_path = None
            if audio is not None:
                audio_path = os.path.join(temp_dir, "audio.wav")
                write_audio_wav(audio, audio_path)
            if image_input:
                command = build_command(executable, output_path, codec, preset, quality, audio_path, fruc=False, filters=filters)
                run_nvencc(command, images, fps)
            else:
                input_path = get_video_input_path(images, temp_dir)
                if gif_input:
                    input_decoder = "software"
                    if needs_gif_padding:
                        padded_path = os.path.join(temp_dir, "padded.mkv")
                        run_ffmpeg([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", input_path,
                                    "-map", "0:v:0", "-map", "0:a?", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                    "-c:v", "ffv1", "-pix_fmt", "yuv444p", "-c:a", "copy", "-y", padded_path])
                        input_path = padded_path
                command = build_command(
                    executable, output_path, codec, preset, quality, audio_path, fruc=False, filters=filters,
                    input_path=input_path, input_decoder=input_decoder, copy_audio=audio is None,
                    trim=(*images.get_active_trim_window(), float(images.get_frame_rate())),
                )
                run_nvencc_video(command)
    except BaseException:
        if os.path.isfile(output_path):
            os.remove(output_path)
        raise

    video = InputImpl.VideoFromFile(output_path)
    return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class LoadGIFNVEncC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(file for file in os.listdir(input_dir) if file.lower().endswith(".gif") and os.path.isfile(os.path.join(input_dir, file)))
        return io.Schema(
            node_id="LoadGIFNVEncC",
            display_name="Load GIF as Video",
            category="video/nvencc",
            description="Loads an animated GIF from ComfyUI's input folder as a native VIDEO handle. The GIF file passes to downstream video nodes without loading every frame into an IMAGE batch. NVEncC uses FFmpeg software decoding for this source.",
            inputs=[io.Combo.Input("file", options=files, upload=io.UploadType.image, tooltip="Animated GIF in ComfyUI's input folder. Upload or select a .gif file; its frame timing is read from the file and no audio is available from GIF.")],
            outputs=[io.Video.Output("video", tooltip="Native VIDEO handle to the GIF file for NVEncC processing, FFmpeg GIF saving, or other VIDEO consumers.")],
        )

    @classmethod
    def execute(cls, file):
        path = folder_paths.get_annotated_filepath(file)
        if not path.lower().endswith(".gif") or not os.path.isfile(path):
            raise ValueError("Load GIF as Video requires an existing .gif file")
        video = InputImpl.VideoFromFile(path)
        video.get_dimensions()
        return io.NodeOutput(video, ui=ui.SavedImages([ui.SavedResult(os.path.basename(path), "", io.FolderType.input)], is_animated=True))

    @classmethod
    def fingerprint_inputs(cls, file):
        return os.path.getmtime(folder_paths.get_annotated_filepath(file))


class SaveGIFWithFFmpeg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveGIFWithFFmpeg",
            display_name="Save GIF with FFmpeg",
            category="video/nvencc",
            description="Saves a VIDEO as an animated GIF using a palette generated from the whole clip and the selected dithering method. GIF has no audio. Returns the saved GIF as a native VIDEO handle for downstream nodes.",
            inputs=[
                io.Video.Input("video", tooltip="VIDEO source, including a loaded GIF or the output of Save Video with NVEncC. FFmpeg reads its frame timestamps; GIF centisecond timing may shift individual frame delays. Audio is ignored."),
                io.Combo.Input("dither", options=list(GIF_DITHER_OPTIONS), default="sierra2_4a", tooltip=GIF_DITHER_TOOLTIP),
                io.Int.Input("bayer_scale", default=2, min=0, max=5, advanced=True, tooltip=GIF_BAYER_TOOLTIP),
                io.String.Input("filename_prefix", default="%date:yyyy-MM-dd%/gif", tooltip="Relative output folder and filename prefix inside ComfyUI's output directory. Date tokens are supported; paths outside output are rejected."),
                io.String.Input("ffmpeg_path", default="", advanced=True, tooltip="Optional full path to ffmpeg.exe. Blank uses FFMPEG_PATH or PATH."),
            ],
            outputs=[io.Video.Output("video", tooltip="Saved animated GIF as a native VIDEO handle for downstream VIDEO nodes.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video, dither, bayer_scale, filename_prefix, ffmpeg_path):
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            expand_date_tokens(filename_prefix), folder_paths.get_output_directory(), width, height
        )
        file = f"{filename}_{counter:05}.gif"
        output_path = os.path.join(full_output_folder, file)
        try:
            encode_gif(video, dither, bayer_scale, ffmpeg_path, output_path)
        except BaseException:
            if os.path.isfile(output_path):
                os.remove(output_path)
            raise
        saved = InputImpl.VideoFromFile(output_path)
        return io.NodeOutput(saved, ui=ui.SavedImages([ui.SavedResult(file, subfolder, io.FolderType.output)], is_animated=True))


class ConvertVideoToGIF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertVideoToGIF",
            display_name="Convert Video to GIF (FFmpeg)",
            category="video/nvencc",
            description="Applies GIF palette conversion and selected dithering in memory. With source audio, the VIDEO output carries the dithered frames as FFV1 with original audio for a single connection to Save Video with NVEncC. A standalone GIF cannot carry audio.",
            inputs=[
                io.Video.Input("video", tooltip="Source VIDEO, including audio embedded in a loaded video. FFmpeg reads frame timestamps; GIF centisecond timing may shift individual frame delays."),
                io.Combo.Input("dither", options=list(GIF_DITHER_OPTIONS), default="sierra2_4a", tooltip=GIF_DITHER_TOOLTIP),
                io.Int.Input("bayer_scale", default=2, min=0, max=5, advanced=True, tooltip=GIF_BAYER_TOOLTIP),
                io.String.Input("ffmpeg_path", default="", advanced=True, tooltip="Optional full path to ffmpeg.exe. Blank uses FFMPEG_PATH or PATH."),
            ],
            outputs=[io.Video.Output("video", tooltip="GIF-encoded VIDEO when silent; FFV1 VIDEO with GIF-dithered frames and original audio when the source has audio. No GIF is saved to the output folder.")],
        )

    @classmethod
    def execute(cls, video, dither, bayer_scale, ffmpeg_path):
        gif_bytes = encode_gif(video, dither, bayer_scale, ffmpeg_path)
        converted = InputImpl.VideoFromFile(BytesIO(mux_gif_audio(video, gif_bytes, ffmpeg_path)))
        converted.get_dimensions()
        return io.NodeOutput(converted)


def decode_bridge(path, start_frame, end_frame, frame_count):
    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).to(dtype=start_frame.dtype).div_(255.0))
    if not frames:
        raise RuntimeError("NVEncC FRUC produced no bridge frames")
    decoded = torch.stack(frames).to(device=start_frame.device)
    return select_bridge_frames(decoded, start_frame, end_frame, frame_count)


class NVEncCFRUCTailBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCFRUCTailBridge",
            display_name="Replace Video Tail with NVEncC FRUC",
            category="video/nvencc",
            description="Experimental motion-smoothing node. Replaces the final N frames with a uniform NvOFFRUC transition from the last preserved frame to the original or connected end frame. The IMAGE count and audio timing stay unchanged.",
            inputs=[
                io.Image.Input("images", tooltip="Decoded source video frames. The batch must contain at least replace_frames + 1 frames."),
                io.Int.Input("replace_frames", default=12, min=2, max=120, tooltip="Number of final frames to replace. At 24 fps, 12 frames is half a second."),
                io.Int.Input("quality", default=100, min=0, max=100, tooltip="How much of the original H3 tail guides the replacement. 100 uses every original tail frame with one FRUC refinement pass for best detail; 0 uses only the two endpoints and may warp on complex motion."),
                io.String.Input("nvencc_path", default="", advanced=True, tooltip="Optional full path to NVEncC64.exe. Leave blank to use the normal NVEncC search locations."),
                io.Image.Input("end_frame", optional=True, tooltip="Optional exact endpoint. When disconnected, the original final video frame is used."),
            ],
            outputs=[io.Image.Output("images")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images, replace_frames, quality, nvencc_path, end_frame=None):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("Replace Video Tail with NVEncC FRUC requires IMAGE frames")
        if images.shape[0] <= replace_frames:
            raise ValueError("replace_frames must be smaller than the source frame count")

        height, width = images.shape[1:3]
        if width % 2 or height % 2:
            raise ValueError("NVEncC FRUC requires even image width and height")

        rgb = images[..., :3]
        start_index = rgb.shape[0] - replace_frames - 1
        start = rgb[start_index]
        end = rgb[-1] if end_frame is None else end_frame[0, ..., :3]
        if end.shape != start.shape:
            raise ValueError("end_frame dimensions must match the source video")
        end = end.to(device=start.device, dtype=start.dtype)

        executable = find_nvencc(nvencc_path)
        tail = rgb[start_index:].clone()
        tail[-1] = end
        guide_count = 2 + round((replace_frames - 1) * quality / 100)
        guide_indices = torch.linspace(0, replace_frames, guide_count).round().to(dtype=torch.int64)
        bridge = tail[guide_indices]
        with tempfile.TemporaryDirectory(prefix="comfy_nvencc_bridge_") as temp_dir:
            pass_index = 0
            while bridge.shape[0] < replace_frames + 1 or pass_index == 0:
                output_path = os.path.join(temp_dir, f"bridge_{pass_index}.mkv")
                command = build_command(executable, output_path, "h264", "p1", 0.0, lossless=True)
                run_nvencc(command, bridge, 24.0)
                bridge = decode_bridge(output_path, start, end, bridge.shape[0] * 2 - 1)
                pass_index += 1
            curve = 1.0 + 0.35 * quality / 100
            bridge = select_bridge_frames(bridge, start, end, replace_frames + 1, curve)

        return io.NodeOutput(torch.cat((rgb[:start_index + 1], bridge[1:]), dim=0))


class NVEncCFrameDouble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCFrameDouble",
            display_name="NVEncC Frame Double (FRUC)",
            category="video/nvencc",
            description="Adds NVIDIA Optical Flow frame-rate up-conversion to an NVEncC filter chain. The save node then inserts one generated frame between every source-frame pair, doubling FPS without changing duration.",
            inputs=[NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves its other stages and replaces any earlier FRUC stage.")],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing FRUC frame doubling for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, filters=None):
        return io.NodeOutput(add_nvencc_filter(filters, "fruc", "double"))


class NVEncCNGXVSR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCNGXVSR",
            display_name="NVEncC Upscale (NGX VSR)",
            category="video/nvencc",
            description="Adds NVIDIA RTX Video Super Resolution to an NVEncC filter chain. The save node upscales the video to the selected width or height while preserving its aspect ratio, then encodes that enlarged result. It requires Windows, an RTX GPU, and NGX-capable NVEncC files.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier NGX VSR stage."),
                io.Combo.Input("target_dimension", options=["height", "width"], default="height", tooltip="Dimension used as the exact output target. height sets the requested output height and calculates an even width from the source aspect ratio; width sets the requested output width and calculates an even height."),
                io.Int.Input("target_size", default=1080, min=2, max=16384, step=2, tooltip="Exact size in pixels for the selected target dimension. It must be even and larger than the corresponding source dimension; the other dimension is calculated to preserve aspect ratio."),
                io.Int.Input("quality", default=4, min=1, max=4, tooltip="NGX VSR quality from 1 to 4. 1 uses the least processing and is fastest; 4 uses the highest-quality restoration and is the recommended default for offline output."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing NGX VSR output sizing and quality settings for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, target_dimension, target_size, quality, filters=None):
        parameters = {
            "target_dimension": target_dimension,
            "target_size": target_size,
            "quality": quality,
        }
        return io.NodeOutput(add_nvencc_filter(filters, "ngx_vsr", parameters))


class NVEncCCAS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCCAS",
            display_name="NVEncC Sharpen (CAS)",
            category="video/nvencc",
            description="Adds restrained, contrast-adaptive sharpening to an NVEncC filter chain. CAS sharpens luma by default, can use its HDR response for PQ or HLG sources, and can optionally process chroma.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier CAS stage."),
                io.Float.Input("strength", default=0.4, min=0.0, max=1.0, step=0.05, tooltip="CAS sharpness from 0 to 1. Around 0.2-0.4 is subtle; values near 1 can emphasize noise and compression artifacts. 0 still includes a zero-strength CAS stage."),
                io.Boolean.Input("hdr", display_name="HDR source", default=False, tooltip="True skips CAS's SDR gamma approximation for PQ or HLG HDR sources; false uses the normal SDR response."),
                io.Boolean.Input("chroma", display_name="sharpen chroma", default=False, tooltip="True sharpens chroma planes as well as luma and may emphasize color noise or ringing; false sharpens luma only."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing CAS for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, strength, filters=None, hdr=False, chroma=False):
        parameters = [f"sharpness={strength:g}"]
        if hdr:
            parameters.append("hdr=true")
        if chroma:
            parameters.append("chroma=true")
        return io.NodeOutput(add_nvencc_filter(filters, "cas", ",".join(parameters)))


class NVEncCUnsharp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCUnsharp",
            display_name="NVEncC Sharpen (Unsharp)",
            category="video/nvencc",
            description="Adds broad unsharp-mask enhancement to an NVEncC filter chain. It produces the most immediately visible edge sharpening but aggressive radius or weight can create halos and ringing.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier Unsharp stage."),
                io.Int.Input("radius", default=3, min=1, max=9, step=1, tooltip="Edge-detection radius in pixels. Larger values affect wider structures and increase halo risk."),
                io.Float.Input("weight", default=0.5, min=0.0, max=10.0, step=0.05, tooltip="Sharpening amplification. 0 has no visible effect; values above 1 are aggressive and can amplify ringing and noise."),
                io.Float.Input("threshold", default=10.0, min=0.0, max=255.0, step=1.0, tooltip="Minimum 8-bit brightness change sharpened. 0 includes weak variations and noise; higher values protect flatter areas."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing Unsharp for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, radius, weight, threshold, filters=None):
        return io.NodeOutput(add_nvencc_filter(filters, "unsharp", f"radius={radius},weight={weight:g},threshold={threshold:g}"))


class NVEncCEdgeLevel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCEdgeLevel",
            display_name="NVEncC Sharpen (EdgeLevel)",
            category="video/nvencc",
            description="Adds direct edge-level sharpening to an NVEncC filter chain. Separate dark- and bright-edge controls can emphasize outlines, while excessive values can create halos or crush fine edge transitions.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier EdgeLevel stage."),
                io.Float.Input("strength", default=5.0, min=-31.0, max=31.0, step=0.5, tooltip="Edge-sharpening strength from -31 to 31. Larger positive values produce stronger sharpening; 5 is NVEncC's default."),
                io.Float.Input("threshold", default=20.0, min=0.0, max=255.0, step=1.0, tooltip="Noise threshold in 8-bit luma levels. Higher values exclude progressively larger brightness changes from edge enhancement."),
                io.Float.Input("black", default=0.0, min=0.0, max=31.0, step=0.5, tooltip="Additional dark-side edge emphasis from 0 to 31. Higher values deepen dark outlines and increase dark-halo risk."),
                io.Float.Input("white", default=0.0, min=0.0, max=31.0, step=0.5, tooltip="Additional bright-side edge emphasis from 0 to 31. Higher values brighten edge highlights and increase bright-halo risk."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing EdgeLevel for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, strength, threshold, black, white, filters=None):
        parameters = f"strength={strength:g},threshold={threshold:g},black={black:g},white={white:g}"
        return io.NodeOutput(add_nvencc_filter(filters, "edgelevel", parameters))


class NVEncCMSharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCMSharpen",
            display_name="NVEncC Sharpen (MSharpen)",
            category="video/nvencc",
            description="Adds edge-selective MSharpen to an NVEncC filter chain. Its dark-area and compression-block protection make it a strong general-purpose showcase filter for encoded video.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier MSharpen stage."),
                io.Float.Input("strength", default=0.8, min=0.0, max=1.0, step=0.05, tooltip="Detected-edge amplification. 0 has no visible effect and 1 is the strongest supported value."),
                io.Float.Input("threshold", default=10.0, min=0.0, max=255.0, step=1.0, tooltip="Edge-detection threshold in 8-bit levels. Lower values include weaker detail and noise; higher values restrict sharpening to stronger edges."),
                io.Float.Input("slope", default=0.0, min=0.0, step=0.1, tooltip="Sigmoid mask slope. 0 uses the hard/default transition; higher nonnegative values soften how the edge mask ramps in."),
                io.Float.Input("luma_limit", default=16.0, min=0.0, max=255.0, step=1.0, tooltip="Dark-area protection in 8-bit luma levels. 0 disables protection; higher values attenuate sharpening in progressively brighter dark regions."),
                io.Float.Input("block_protect", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="Compression-block protection. 0 disables attenuation and 1 applies maximum protection near DCT block boundaries."),
                io.Boolean.Input("high_quality", default=True, tooltip="True uses higher-quality edge detection at additional processing cost; false uses the faster detector."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing MSharpen for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, strength, threshold, slope, luma_limit, block_protect, high_quality, filters=None):
        parameters = (
            f"strength={strength:g},threshold={threshold:g},slope={slope:g},luma_limit={luma_limit:g},"
            f"block_protect={block_protect:g},highq={str(high_quality).lower()}"
        )
        return io.NodeOutput(add_nvencc_filter(filters, "msharpen", parameters))


class NVEncCDetailSharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCDetailSharpen",
            display_name="NVEncC Sharpen (DetailSharpen)",
            category="video/nvencc",
            description="Adds nonlinear fine-texture enhancement to an NVEncC filter chain. Damping can protect weak noise, while excessive strength may exaggerate grain or create crawling texture across frames.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier DetailSharpen stage."),
                io.Float.Input("zero_point", default=4.0, min=0.001, max=64.0, step=0.1, tooltip="Detail amplitude around which enhancement transitions. Smaller values target finer, weaker texture."),
                io.Float.Input("strength", default=1.5, min=0.0, max=16.0, step=0.1, tooltip="Fine-texture amplification. 0 has no visible effect; high values can amplify grain, ringing, and compression texture."),
                io.Float.Input("power", default=4.0, min=1.0, max=16.0, step=0.1, tooltip="Nonlinear response power. Higher values concentrate enhancement more selectively around qualifying detail."),
                io.Float.Input("damping", default=1.0, min=0.0, max=1000.0, step=0.1, tooltip="Low-amplitude damping. 0 leaves weak texture unprotected; higher values suppress enhancement of subtle noise and low-level detail."),
                io.Combo.Input("blur_mode", options=["box", "gaussian"], default="box", tooltip="Detail baseline: box uses NVEncC mode 1 and its default neighborhood; gaussian uses mode 0 for a smoother weighted neighborhood."),
                io.Boolean.Input("median", default=False, tooltip="True applies a 3x3 median operation to the blur reference to reject isolated specks; false uses the selected blur directly."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing DetailSharpen for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, zero_point, strength, power, damping, blur_mode, median, filters=None):
        parameters = (
            f"z={zero_point:g},sstr={strength:g},power={power:g},ldmp={damping:g},"
            f"mode={1 if blur_mode == 'box' else 0},med={str(median).lower()}"
        )
        return io.NodeOutput(add_nvencc_filter(filters, "detailsharpen", parameters))


class NVEncCWarpSharp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVEncCWarpSharp",
            display_name="NVEncC Sharpen (WarpSharp)",
            category="video/nvencc",
            description="Adds edge-directed pixel warping to an NVEncC filter chain. WarpSharp can tighten outlines and fine structures more strongly than a conventional sharpening kernel, but excessive depth may bend details or create temporal shimmer.",
            inputs=[
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional existing NVEncC filter chain. Disconnected starts a new chain; connected preserves other stages and replaces an earlier WarpSharp stage."),
                io.Float.Input("threshold", default=128.0, min=0.0, max=255.0, step=1.0, tooltip="Edge-mask threshold in 8-bit levels. Higher values produce a stronger edge mask and more sharpening; lower values reduce the warp."),
                io.Int.Input("blur", default=2, min=0, step=1, tooltip="Number of edge-mask blur passes. 0 uses the unblurred mask; higher counts spread and weaken the sharpening effect."),
                io.Combo.Input("blur_type", options=["13x13", "5x5"], default="13x13", tooltip="Edge-mask blur kernel: 13x13 is NVEncC's broad default; 5x5 is a smaller higher-quality kernel that generally needs more blur passes."),
                io.Float.Input("depth", default=16.0, min=-128.0, max=128.0, step=1.0, tooltip="Base warp distance from -128 to 128. Larger positive magnitudes tighten edges more strongly; 0 disables displacement, while negative values reverse its direction."),
                io.Combo.Input("chroma_mode", options=["luma mask", "per-channel"], default="luma mask", tooltip="Chroma processing: luma mask reuses the luma edge mask for both chroma planes; per-channel detects and warps chroma edges independently."),
                io.Boolean.Input("adaptive_depth", default=False, advanced=True, tooltip="False uses the base depth uniformly and omits adaptive parameters; true interpolates from minimum to maximum depth according to edge-mask strength."),
                io.Float.Input("depth_min", default=4.0, min=-128.0, max=128.0, step=1.0, advanced=True, tooltip="Warp distance for weak edges when adaptive depth is true. Ignored when adaptive depth is false."),
                io.Float.Input("depth_max", default=16.0, min=-128.0, max=128.0, step=1.0, advanced=True, tooltip="Warp distance reached by strong edges when adaptive depth is true. Ignored when adaptive depth is false."),
                io.Float.Input("edge_threshold", default=192.0, min=1.0, max=255.0, step=1.0, advanced=True, tooltip="8-bit edge-mask level at which adaptive depth reaches its maximum. Ignored when adaptive depth is false."),
                io.Float.Input("gamma", default=1.0, min=0.01, max=8.0, step=0.05, advanced=True, tooltip="Adaptive response curve. Values below 1 favor weak edges and values above 1 concentrate the warp on strong edges; ignored when adaptive depth is false."),
            ],
            outputs=[NVEncCFilters.Output("filters", tooltip="NVEncC filter chain containing WarpSharp for connection to another suite filter or Save Video with NVEncC.")],
        )

    @classmethod
    def execute(cls, threshold, blur, blur_type, depth, chroma_mode, adaptive_depth, depth_min, depth_max, edge_threshold, gamma, filters=None):
        parameters = [
            f"threshold={threshold:g}",
            f"blur={blur}",
            f"type={1 if blur_type == '5x5' else 0}",
            f"depth={depth:g}",
            f"chroma={1 if chroma_mode == 'per-channel' else 0}",
        ]
        if adaptive_depth:
            parameters.extend((f"depth_min={depth_min:g}", f"depth_max={depth_max:g}", f"edge_thr={edge_threshold:g}", f"gamma={gamma:g}"))
        return io.NodeOutput(add_nvencc_filter(filters, "warpsharp", ",".join(parameters)))


class SaveVideoNVEncC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoNVEncC",
            display_name="Save Video with NVEncC",
            category="video/nvencc",
            description="Streams a native VIDEO directly to NVEncC, or accepts an IMAGE batch, applies an optional NVENCC_FILTERS chain, saves the encoded file, and returns a VIDEO handle. GIF files use software decoding; odd GIF dimensions are padded to even size before encoding, and transparency is discarded. VIDEO input keeps source FPS and copies source audio unless replacement audio is connected; IMAGE input uses the FPS widget.",
            inputs=[
                io.MultiType.Input("images", [io.Video, io.Image], display_name="video / images", tooltip="VIDEO streams directly from Load Video or Load GIF as Video without materializing every frame; IMAGE accepts generated frame batches. Odd GIF dimensions are padded by one pixel as needed."),
                NVEncCFilters.Input("filters", optional=True, tooltip="Optional chain from the NVEncC suite filter nodes. Disconnected performs encoding only, with no FRUC or sharpening stages."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01, tooltip="Frame rate for IMAGE input from 1 to 240 fps. VIDEO input ignores this widget and uses its source frame rate."),
                io.String.Input("filename_prefix", display_name="output folder / filename prefix", default="%date:yyyy-MM-dd%/nvencc", tooltip="Relative output folder and filename prefix inside ComfyUI's output directory. Date formatting tokens are supported; absolute and escaping paths are rejected by ComfyUI's save-path resolver."),
                io.Combo.Input("container", options=["mp4", "mkv"], default="mp4", tooltip="Output wrapper: mp4 offers broad playback compatibility; mkv tolerates a wider range of stream combinations. This does not select compression."),
                io.Combo.Input("codec", options=["av1", "hevc", "h264"], default="av1", tooltip="NVENC codec: av1 gives the best compression on supported players; hevc balances compression and compatibility; h264 has the widest playback support."),
                io.Combo.Input("preset", options=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], default="p4", advanced=True, tooltip="NVENC effort: p1 is fastest; p2 and p3 favor speed; p4 is balanced; p5 and p6 favor efficiency; p7 uses the most encoding effort."),
                io.Float.Input("quality", default=20.0, min=0.0, max=51.0, step=0.5, advanced=True, tooltip="QVBR target quality from 0 to 51. Lower values produce higher quality and larger files; 18-24 is a practical starting range."),
                io.Combo.Input("input_decoder", options=["hardware", "software"], default="hardware", advanced=True, tooltip="VIDEO decoder: hardware uses NVIDIA decoding; software uses NVEncC's software reader. GIF files always use software decoding. IMAGE input ignores this setting."),
                io.String.Input("nvencc_path", default="", advanced=True, tooltip="Optional full path to NVEncC64.exe. Blank searches NVENCC_PATH, ComfyUI/tools/NVEncC, ComfyUI/tools, and PATH."),
                io.Audio.Input("audio", optional=True, tooltip="Optional replacement audio encoded as AAC at 192 kb/s. With VIDEO input, disconnected copies source audio; IMAGE input without audio produces a silent video."),
            ],
            is_output_node=True,
            outputs=[io.Video.Output("video", display_name="saved video (optional)", tooltip="Saved processed VIDEO handle for preview, Save Video, or other native VIDEO consumers.")],
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, container, codec, preset, quality, input_decoder, nvencc_path, filters=None, audio=None):
        return save_video_nvencc(images, fps, filename_prefix, container, codec, preset, quality, nvencc_path, filters, audio, input_decoder)


class SaveVideoNVEncCFRUC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoNVEncCFRUC",
            display_name="Save Video with NVEncC (Legacy All-in-One)",
            category="video/nvencc/legacy",
            description="Streams a VIDEO directly to NVEncC, or accepts an IMAGE batch, then saves and returns the encoded VIDEO. FRUC frame doubling, adaptive CAS, broad Unsharp, edge-selective MSharpen, and fine-texture DetailSharpen are independent stages and may be stacked. NVEncC applies stacked sharpening in Unsharp, MSharpen, CAS, DetailSharpen order. Strong combinations can amplify noise, compression blocks, flicker, and halos. Requires Windows, a supported NVIDIA GPU, and NVEncC; FRUC also requires NvOFFRUC support.",
            inputs=[
                io.MultiType.Input("images", [io.Video, io.Image], display_name="video / images", tooltip="Connect VIDEO directly from Load Video to avoid decoding the whole file into an IMAGE batch. IMAGE input remains supported for generated frames."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01, tooltip="Frame rate for IMAGE input. VIDEO input uses the source video's frame rate and ignores this value."),
                io.String.Input(
                    "filename_prefix",
                    display_name="output folder / filename prefix",
                    default="%date:yyyy-MM-dd%/upframes",
                    tooltip="Relative folder path and filename prefix inside ComfyUI's output directory. Supports formatting tokens such as %date:yyyy-MM-dd%; for example, %date:yyyy-MM-dd%/upframes saves as output/YYYY-MM-DD/upframes_00001_fruc.mp4. Absolute paths and paths outside the output directory are not allowed.",
                ),
                io.Combo.Input("container", options=["mp4", "mkv"], default="mp4", tooltip="Output wrapper: mp4 offers broad playback compatibility; mkv tolerates a wider range of stream combinations. This does not select the compression codec."),
                io.Combo.Input("codec", options=["av1", "hevc", "h264"], default="av1", tooltip="NVENC codec: av1 gives the best compression on supported players; hevc balances compression and compatibility; h264 has the widest playback support."),
                io.Combo.Input("preset", options=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], default="p4", advanced=True, tooltip="NVENC effort preset: p1 is fastest, p2 and p3 favor speed, p4 is balanced, p5 and p6 favor efficiency, and p7 spends the most time for compression efficiency."),
                io.Float.Input("quality", default=20.0, min=0.0, max=51.0, step=0.5, advanced=True, tooltip="QVBR target quality. Lower values give higher quality and larger files; 18-24 is a useful starting range."),
                io.Combo.Input("input_decoder", options=["hardware", "software"], default="hardware", advanced=True, tooltip="VIDEO decoder: hardware uses NVIDIA decoding and avoids CPU decode; software uses NVEncC's software reader for codecs unsupported by the GPU. IMAGE input ignores this setting."),
                io.String.Input("nvencc_path", default="", advanced=True, tooltip="Optional full path to NVEncC64.exe. Leave blank to search NVENCC_PATH, ComfyUI/tools/NVEncC, ComfyUI/tools, and PATH."),
                io.Float.Input("sharpen", display_name="CAS strength", default=0.0, min=0.0, max=1.0, step=0.05, advanced=True, tooltip="Contrast-adaptive luma sharpening strength. 0 omits CAS even when enabled; 0.2-0.4 is subtle, while values near 1 can emphasize noise and compression artifacts."),
                io.Boolean.Input("enable_fruc", display_name="enable frame doubling (FRUC)", default=True, tooltip="True inserts one generated frame between every source-frame pair; false encodes at the source frame rate."),
                io.Boolean.Input("enable_sharpen", display_name="enable CAS sharpening", default=True, tooltip="True applies adaptive CAS when CAS strength is above 0; false omits CAS. This does not control the other sharpening stages."),
                io.Boolean.Input("enable_unsharp", display_name="enable Unsharp", default=False, advanced=True, tooltip="True applies broad unsharp-mask edge enhancement using radius, weight, and threshold; false omits Unsharp. It can be stacked with the other stages but may create halos."),
                io.Int.Input("unsharp_radius", default=3, min=1, max=9, step=1, advanced=True, tooltip="Unsharp edge-detection radius in pixels. Larger values affect wider structures and increase halo risk; used only when Unsharp is enabled."),
                io.Float.Input("unsharp_weight", default=0.5, min=0.0, max=10.0, step=0.05, advanced=True, tooltip="Unsharp amplification weight. 0 has no visible effect; values above 1 are aggressive and can strongly amplify ringing and noise."),
                io.Float.Input("unsharp_threshold", default=10.0, min=0.0, max=255.0, step=1.0, advanced=True, tooltip="Minimum 8-bit brightness change sharpened by Unsharp. 0 sharpens small variations including noise; higher values protect flatter areas."),
                io.Boolean.Input("enable_msharpen", display_name="enable MSharpen", default=False, advanced=True, tooltip="True applies edge-selective MSharpen using its mask controls; false omits MSharpen. It can be stacked with other stages."),
                io.Float.Input("msharpen_strength", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True, tooltip="MSharpen edge amplification. 0 has no visible effect and 1 is the strongest supported value."),
                io.Float.Input("msharpen_threshold", default=15.0, min=0.0, max=255.0, step=1.0, advanced=True, tooltip="MSharpen edge-detection threshold in 8-bit levels. Lower values include weaker detail and noise; higher values restrict sharpening to stronger edges."),
                io.Float.Input("msharpen_slope", default=0.0, min=0.0, step=0.1, advanced=True, tooltip="MSharpen sigmoid mask slope. 0 uses the filter's hard/default transition; higher nonnegative values soften how the edge mask ramps in."),
                io.Float.Input("msharpen_luma_limit", default=0.0, min=0.0, max=255.0, step=1.0, advanced=True, tooltip="Dark-area protection in 8-bit luma levels. 0 disables protection; higher values attenuate sharpening in progressively brighter dark regions."),
                io.Float.Input("msharpen_block_protect", default=0.0, min=0.0, max=1.0, step=0.05, advanced=True, tooltip="Compression-block protection. 0 disables attenuation and 1 applies maximum protection near DCT block boundaries."),
                io.Boolean.Input("msharpen_high_quality", default=True, advanced=True, tooltip="True uses higher-quality MSharpen edge detection at additional processing cost; false uses the faster edge detector."),
                io.Boolean.Input("enable_detailsharpen", display_name="enable DetailSharpen", default=False, advanced=True, tooltip="True enhances fine texture with nonlinear damping controls; false omits DetailSharpen. It can be stacked but may exaggerate grain or crawling texture."),
                io.Float.Input("detailsharpen_zero_point", display_name="DetailSharpen zero point", default=4.0, min=0.001, max=64.0, step=0.1, advanced=True, tooltip="DetailSharpen zero point controlling the detail amplitude around which enhancement transitions. Smaller values target finer, weaker texture."),
                io.Float.Input("detailsharpen_strength", display_name="DetailSharpen strength", default=1.5, min=0.0, max=16.0, step=0.1, advanced=True, tooltip="Fine-texture amplification strength. 0 has no visible effect; high values can strongly amplify grain, ringing, and compression texture."),
                io.Float.Input("detailsharpen_power", display_name="DetailSharpen power", default=4.0, min=1.0, max=16.0, step=0.1, advanced=True, tooltip="Nonlinear response power. Higher values concentrate enhancement more selectively around qualifying fine detail."),
                io.Float.Input("detailsharpen_damping", display_name="DetailSharpen damping", default=1.0, min=0.0, max=1000.0, step=0.1, advanced=True, tooltip="Low-amplitude damping. 0 leaves weak texture unprotected; higher values suppress enhancement of subtle noise and low-level detail."),
                io.Combo.Input("detailsharpen_blur_mode", display_name="DetailSharpen blur", options=["box", "gaussian"], default="box", advanced=True, tooltip="Detail baseline blur: box uses NVEncC mode 1 and preserves the filter default; gaussian uses mode 0 for a smoother weighted neighborhood."),
                io.Boolean.Input("detailsharpen_median", display_name="DetailSharpen median", default=False, advanced=True, tooltip="True applies a 3x3 median operation to the DetailSharpen blur reference to reject isolated specks; false uses the selected blur directly."),
                io.Audio.Input("audio", optional=True, tooltip="Optional replacement audio encoded as AAC at 192 kb/s. With VIDEO input, leaving this disconnected copies the source audio."),
            ],
            is_output_node=True,
            outputs=[io.Video.Output("video", display_name="saved video (optional)", tooltip="Optional downstream handle to the saved processed video. Leave this output disconnected when this is the final node.")],
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, container, codec, preset, quality, nvencc_path, audio=None, input_decoder="hardware", sharpen=0.0, enable_fruc=True, enable_sharpen=True,
                enable_unsharp=False, unsharp_radius=3, unsharp_weight=0.5, unsharp_threshold=10.0,
                enable_msharpen=False, msharpen_strength=1.0, msharpen_threshold=15.0, msharpen_slope=0.0,
                msharpen_luma_limit=0.0, msharpen_block_protect=0.0, msharpen_high_quality=True,
                enable_detailsharpen=False, detailsharpen_zero_point=4.0, detailsharpen_strength=1.5,
                detailsharpen_power=4.0, detailsharpen_damping=1.0, detailsharpen_blur_mode="box", detailsharpen_median=False):
        image_input = isinstance(images, torch.Tensor)
        if image_input:
            minimum_frames = 2 if enable_fruc else 1
            if images.ndim != 4 or images.shape[0] < minimum_frames:
                raise ValueError(f"Save Video with NVEncC requires at least {minimum_frames} IMAGE frame{'s' if minimum_frames > 1 else ''}")
            height, width = images.shape[1:3]
        else:
            width, height = images.get_dimensions()
        if width % 2 or height % 2:
            raise ValueError("Save Video with NVEncC requires even video width and height")

        executable = find_nvencc(nvencc_path)
        expanded_prefix = expand_date_tokens(filename_prefix)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            expanded_prefix, folder_paths.get_output_directory(), width, height
        )
        suffix = "fruc" if enable_fruc else "nvencc"
        file = f"{filename}_{counter:05}_{suffix}.{container}"
        output_path = os.path.join(full_output_folder, file)
        cas_sharpness = sharpen if enable_sharpen else 0.0
        unsharp = f"radius={unsharp_radius},weight={unsharp_weight:g},threshold={unsharp_threshold:g}" if enable_unsharp else None
        msharpen = (
            f"strength={msharpen_strength:g},threshold={msharpen_threshold:g},slope={msharpen_slope:g},"
            f"luma_limit={msharpen_luma_limit:g},block_protect={msharpen_block_protect:g},highq={str(msharpen_high_quality).lower()}"
            if enable_msharpen else None
        )
        detailsharpen = (
            f"z={detailsharpen_zero_point:g},sstr={detailsharpen_strength:g},power={detailsharpen_power:g},"
            f"ldmp={detailsharpen_damping:g},mode={1 if detailsharpen_blur_mode == 'box' else 0},med={str(detailsharpen_median).lower()}"
            if enable_detailsharpen else None
        )

        try:
            with tempfile.TemporaryDirectory(prefix="comfy_nvencc_fruc_") as temp_dir:
                audio_path = None
                if audio is not None:
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    write_audio_wav(audio, audio_path)
                if image_input:
                    command = build_command(executable, output_path, codec, preset, quality, audio_path, fruc=enable_fruc, cas_sharpness=cas_sharpness,
                                            unsharp=unsharp, msharpen=msharpen, detailsharpen=detailsharpen)
                    run_nvencc(command, images, fps)
                else:
                    input_path = get_video_input_path(images, temp_dir)
                    command = build_command(
                        executable, output_path, codec, preset, quality, audio_path,
                        fruc=enable_fruc, input_path=input_path, input_decoder=input_decoder, copy_audio=audio is None,
                        trim=(*images.get_active_trim_window(), float(images.get_frame_rate())), cas_sharpness=cas_sharpness,
                        unsharp=unsharp, msharpen=msharpen, detailsharpen=detailsharpen,
                    )
                    run_nvencc_video(command)
        except BaseException:
            if os.path.isfile(output_path):
                os.remove(output_path)
            raise

        video = InputImpl.VideoFromFile(output_path)
        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class NVEncCFRUCExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadGIFNVEncC,
            SaveVideoNVEncC,
            SaveGIFWithFFmpeg,
            ConvertVideoToGIF,
            NVEncCFrameDouble,
            NVEncCNGXVSR,
            NVEncCCAS,
            NVEncCUnsharp,
            NVEncCEdgeLevel,
            NVEncCMSharpen,
            NVEncCDetailSharpen,
            NVEncCWarpSharp,
            SaveVideoNVEncCFRUC,
            NVEncCFRUCTailBridge,
        ]


async def comfy_entrypoint() -> NVEncCFRUCExtension:
    return NVEncCFRUCExtension()

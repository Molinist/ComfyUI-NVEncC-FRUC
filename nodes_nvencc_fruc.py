import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime
from fractions import Fraction

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


def build_command(executable, output_path, codec, preset, quality, audio_path=None, fruc=True, lossless=False,
                  input_path=None, input_decoder="hardware", copy_audio=False, trim=(0.0, 0.0, 1.0)):
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
    if frame_count < 2:
        raise ValueError("NvOFFRUC frame doubling requires at least two frames")
    if width % 2 or height % 2:
        raise ValueError("NvOFFRUC YUV420 input requires even image width and height")

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


def get_video_input_path(video, temp_dir):
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)

    input_path = os.path.join(temp_dir, "input.mp4")
    source.seek(0)
    with open(input_path, "wb") as output:
        shutil.copyfileobj(source, output)
    return input_path


def select_bridge_frames(decoded, start_frame, end_frame, frame_count, curve=1.0):
    if len(decoded) < frame_count:
        raise RuntimeError(f"NVEncC FRUC produced {len(decoded)} bridge frames; need at least {frame_count}")
    positions = torch.linspace(0.0, 1.0, frame_count).pow(curve).mul_(len(decoded) - 1)
    indices = positions.round().to(dtype=torch.int64)
    bridge = decoded[indices].clone()
    bridge[0] = start_frame
    bridge[-1] = end_frame
    return bridge


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
            category="video",
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


class SaveVideoNVEncCFRUC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoNVEncCFRUC",
            display_name="Save Video with NVEncC FRUC",
            category="video",
            description="Final output node: connect a VIDEO directly from Load Video, or an IMAGE batch, and run the workflow; the file is saved automatically. Direct VIDEO input is decoded by NVEncC without materializing all frames in ComfyUI. NVIDIA Optical Flow FRUC inserts one frame between consecutive frames and NVENC encodes the result. Requires Windows, a supported NVIDIA GPU, and NVEncC with NvOFFRUC.",
            inputs=[
                io.MultiType.Input("images", [io.Video, io.Image], display_name="video / images", tooltip="Connect VIDEO directly from Load Video to avoid decoding the whole file into an IMAGE batch. IMAGE input remains supported for generated frames."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01, tooltip="Frame rate for IMAGE input. VIDEO input uses the source video's frame rate and ignores this value."),
                io.String.Input(
                    "filename_prefix",
                    display_name="output folder / filename prefix",
                    default="%date:yyyy-MM-dd%/upframes",
                    tooltip="Relative folder path and filename prefix inside ComfyUI's output directory. Supports formatting tokens such as %date:yyyy-MM-dd%; for example, %date:yyyy-MM-dd%/upframes saves as output/YYYY-MM-DD/upframes_00001_fruc.mp4. Absolute paths and paths outside the output directory are not allowed.",
                ),
                io.Combo.Input("container", options=["mp4", "mkv"], default="mp4", tooltip="Output container. This controls the file wrapper, not the video compression format."),
                io.Combo.Input("codec", options=["av1", "hevc", "h264"], default="av1", tooltip="NVENC video codec. AV1 offers high compression efficiency; H.264 has the widest playback compatibility."),
                io.Combo.Input("preset", options=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], default="p4", advanced=True, tooltip="NVENC speed/efficiency preset. P1 is fastest; P7 spends the most time for better compression efficiency. P4 is balanced."),
                io.Float.Input("quality", default=20.0, min=0.0, max=51.0, step=0.5, advanced=True, tooltip="QVBR target quality. Lower values give higher quality and larger files; 18-24 is a useful starting range."),
                io.Combo.Input("input_decoder", options=["hardware", "software"], default="hardware", advanced=True, tooltip="Decoder used for VIDEO input. Hardware keeps decoding on the GPU; use software for a source codec unsupported by NVIDIA's hardware decoder. IMAGE input ignores this option."),
                io.String.Input("nvencc_path", default="", advanced=True, tooltip="Optional full path to NVEncC64.exe. Leave blank to search NVENCC_PATH, ComfyUI/tools/NVEncC, ComfyUI/tools, and PATH."),
                io.Audio.Input("audio", optional=True, tooltip="Optional replacement audio encoded as AAC at 192 kb/s. With VIDEO input, leaving this disconnected copies the source audio."),
            ],
            is_output_node=True,
            outputs=[io.Video.Output("video", display_name="saved video (optional)", tooltip="Optional downstream handle to the saved interpolated video. Leave this output disconnected when this is the final node.")],
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, container, codec, preset, quality, nvencc_path, audio=None, input_decoder="hardware"):
        image_input = isinstance(images, torch.Tensor)
        if image_input:
            if images.ndim != 4 or images.shape[0] < 2:
                raise ValueError("Save Video with NVEncC FRUC requires at least two IMAGE frames")
            height, width = images.shape[1:3]
        else:
            width, height = images.get_dimensions()
        if width % 2 or height % 2:
            raise ValueError("Save Video with NVEncC FRUC requires even video width and height")

        executable = find_nvencc(nvencc_path)
        expanded_prefix = expand_date_tokens(filename_prefix)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            expanded_prefix, folder_paths.get_output_directory(), width, height
        )
        file = f"{filename}_{counter:05}_fruc.{container}"
        output_path = os.path.join(full_output_folder, file)

        try:
            with tempfile.TemporaryDirectory(prefix="comfy_nvencc_fruc_") as temp_dir:
                audio_path = None
                if audio is not None:
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    write_audio_wav(audio, audio_path)
                if image_input:
                    command = build_command(executable, output_path, codec, preset, quality, audio_path)
                    run_nvencc(command, images, fps)
                else:
                    input_path = get_video_input_path(images, temp_dir)
                    command = build_command(
                        executable, output_path, codec, preset, quality, audio_path,
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


class NVEncCFRUCExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideoNVEncCFRUC, NVEncCFRUCTailBridge]


async def comfy_entrypoint() -> NVEncCFRUCExtension:
    return NVEncCFRUCExtension()

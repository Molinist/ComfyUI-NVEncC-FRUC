import io
import wave
from datetime import datetime

import torch

from nodes_nvencc_fruc import build_command, expand_date_tokens, get_video_input_path, select_bridge_frames, write_audio_wav, write_y4m


def test_expand_date_tokens():
    now = datetime(2026, 8, 5, 14, 7, 9)

    assert expand_date_tokens("%date:yyyy-MM-dd%/upframes", now) == "2026-08-05/upframes"
    assert expand_date_tokens("%date:yyMMdd_HH-mm-ss%/clip", now) == "260805_14-07-09/clip"


def test_build_command_enables_fruc_and_audio(tmp_path):
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "output.mp4"
    command = build_command("NVEncC64.exe", str(output_path), "av1", "p4", 20.0, str(audio_path))

    assert command[:4] == ["NVEncC64.exe", "--y4m", "-i", "-"]
    assert command[command.index("--vpp-fruc") + 1] == "double"
    assert command[command.index("--audio-source") + 1] == f"{audio_path}:codec=aac;bitrate=192"
    assert command[-2:] == ["--output", str(output_path)]


def test_build_command_can_disable_fruc_and_enable_lossless(tmp_path):
    output_path = tmp_path / "base.mkv"
    command = build_command("NVEncC64.exe", str(output_path), "h264", "p1", 0.0, fruc=False, lossless=True)

    assert "--vpp-fruc" not in command
    assert "--lossless" in command


def test_build_command_accepts_target_fruc_fps(tmp_path):
    output_path = tmp_path / "bridge.mkv"
    command = build_command("NVEncC64.exe", str(output_path), "h264", "p1", 0.0, fruc="fps=48.0", lossless=True)

    assert command[command.index("--vpp-fruc") + 1] == "fps=48.0"


def test_build_command_accepts_direct_video_with_audio_and_trim(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    command = build_command(
        "NVEncC64.exe", str(output_path), "av1", "p4", 20.0,
        input_path=input_path, input_decoder="hardware", copy_audio=True, trim=(1.25, 2.5, 24.0),
    )

    assert command[:4] == ["NVEncC64.exe", "--avhw", "-i", str(input_path)]
    assert command[command.index("--trim") + 1] == "30:89"
    assert "--audio-copy" in command
    assert "--y4m" not in command


def test_build_command_uses_seek_for_open_ended_video_trim(tmp_path):
    command = build_command(
        "NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p1", 20.0,
        input_path=tmp_path / "input.mkv", trim=(1.25, 0.0, 24.0),
    )

    assert command[command.index("--seek") + 1] == "1.25"
    assert "--trim" not in command


def test_build_command_supports_software_video_decode(tmp_path):
    command = build_command(
        "NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p1", 20.0,
        input_path=tmp_path / "input.mkv", input_decoder="software",
    )

    assert command[1] == "--avsw"


def test_get_video_input_path_preserves_files_and_materializes_buffers(tmp_path):
    source_path = tmp_path / "source.mp4"

    class FileVideo:
        def get_stream_source(self):
            return str(source_path)

    class BufferVideo:
        def get_stream_source(self):
            return io.BytesIO(b"video data")

    assert get_video_input_path(FileVideo(), str(tmp_path)) == str(source_path)
    buffered_path = get_video_input_path(BufferVideo(), str(tmp_path))
    with open(buffered_path, "rb") as source:
        assert source.read() == b"video data"


def test_select_bridge_frames_returns_exact_count_and_endpoints():
    decoded = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1)
    start = torch.tensor([[[20.0]]])
    end = torch.tensor([[[30.0]]])

    bridge = select_bridge_frames(decoded, start, end, 5)

    assert bridge[:, 0, 0, 0].tolist() == [20.0, 2.0, 4.0, 6.0, 30.0]


def test_select_bridge_frames_supports_timing_curve():
    decoded = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1)
    start = torch.tensor([[[20.0]]])
    end = torch.tensor([[[30.0]]])

    bridge = select_bridge_frames(decoded, start, end, 5, curve=2.0)

    assert bridge[:, 0, 0, 0].tolist() == [20.0, 0.0, 2.0, 4.0, 30.0]


def test_write_y4m_streams_tightly_packed_yuv420():
    images = torch.zeros((2, 4, 6, 3), dtype=torch.float32)
    output = io.BytesIO()

    write_y4m(images, 23.976, output)

    data = output.getvalue()
    header, frames = data.split(b"\n", 1)
    assert header == b"YUV4MPEG2 W6 H4 F2997:125 Ip A0:0 C420jpeg"
    assert frames.count(b"FRAME\n") == 2
    assert len(frames) == 2 * (len(b"FRAME\n") + 6 * 4 * 3 // 2)


def test_write_y4m_rejects_odd_dimensions():
    images = torch.zeros((2, 5, 6, 3), dtype=torch.float32)

    try:
        write_y4m(images, 24.0, io.BytesIO())
    except ValueError as error:
        assert "even image width and height" in str(error)
    else:
        raise AssertionError("odd YUV420 dimensions should fail")


def test_write_audio_wav_uses_first_batch(tmp_path):
    path = tmp_path / "audio.wav"
    audio = {
        "waveform": torch.tensor([[[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]]]),
        "sample_rate": 48000,
    }

    write_audio_wav(audio, str(path))

    with wave.open(str(path), "rb") as source:
        assert source.getnchannels() == 2
        assert source.getframerate() == 48000
        assert source.getnframes() == 3

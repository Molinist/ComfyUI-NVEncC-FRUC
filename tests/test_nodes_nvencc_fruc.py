import asyncio
import io
import os
import shutil
import subprocess
import tomllib
import wave
from datetime import datetime
from pathlib import Path

import torch
import pytest
from PIL import Image

import nodes_nvencc_fruc as nvencc_module
from nodes_nvencc_fruc import (
    ConvertVideoToGIF,
    LoadGIFNVEncC,
    NVEncCCAS,
    NVEncCDetailSharpen,
    NVEncCEdgeLevel,
    NVEncCFrameDouble,
    NVEncCMSharpen,
    NVEncCNGXVSR,
    NVEncCUnsharp,
    NVEncCWarpSharp,
    SaveVideoNVEncC,
    SaveGIFWithFFmpeg,
    SaveVideoNVEncCFRUC,
    build_command,
    expand_date_tokens,
    get_video_input_path,
    resolve_nvencc_filters,
    select_bridge_frames,
    write_audio_wav,
    write_y4m,
)


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


def test_build_command_adds_optional_cas_sharpening(tmp_path):
    command = build_command(
        "NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p1", 20.0, cas_sharpness=0.3,
    )

    assert command[command.index("--vpp-cas") + 1] == "sharpness=0.3"


def test_build_command_disables_cas_at_zero(tmp_path):
    command = build_command("NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p1", 20.0)

    assert "--vpp-cas" not in command


def test_build_command_controls_fruc_and_cas_independently(tmp_path):
    output_path = str(tmp_path / "output.mkv")

    for fruc, sharpness, expected_fruc, expected_cas in (
        (False, 0.0, False, False),
        (True, 0.0, True, False),
        (False, 0.4, False, True),
        (True, 0.4, True, True),
    ):
        command = build_command("NVEncC64.exe", output_path, "h264", "p1", 20.0, fruc=fruc, cas_sharpness=sharpness)
        assert ("--vpp-fruc" in command) is expected_fruc
        assert ("--vpp-cas" in command) is expected_cas


def test_build_command_adds_independent_advanced_sharpening_filters(tmp_path):
    command = build_command(
        "NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p1", 20.0, fruc=False,
        unsharp="radius=5,weight=1.25,threshold=8",
        msharpen="strength=0.8,threshold=20,slope=1.5,luma_limit=24,block_protect=0.5,highq=false",
        detailsharpen="z=3,sstr=2.5,power=5,ldmp=2,mode=0,med=true",
    )

    assert "--vpp-fruc" not in command
    assert command[command.index("--vpp-unsharp") + 1] == "radius=5,weight=1.25,threshold=8"
    assert command[command.index("--vpp-msharpen") + 1] == "strength=0.8,threshold=20,slope=1.5,luma_limit=24,block_protect=0.5,highq=false"
    assert command[command.index("--vpp-detailsharpen") + 1] == "z=3,sstr=2.5,power=5,ldmp=2,mode=0,med=true"


def test_suite_filter_nodes_compose_and_replace_matching_stage():
    filters = NVEncCCAS.execute(0.3).args[0]
    filters = NVEncCUnsharp.execute(3, 0.5, 10.0, filters).args[0]
    filters = NVEncCEdgeLevel.execute(5.0, 20.0, 2.0, 1.0, filters).args[0]
    filters = NVEncCMSharpen.execute(0.8, 10.0, 0.0, 16.0, 0.5, True, filters).args[0]
    filters = NVEncCDetailSharpen.execute(4.0, 1.5, 4.0, 1.0, "box", False, filters).args[0]
    filters = NVEncCWarpSharp.execute(128.0, 2, "13x13", 16.0, "luma mask", False, 4.0, 16.0, 192.0, 1.0, filters).args[0]
    filters = NVEncCFrameDouble.execute(filters).args[0]
    filters = NVEncCCAS.execute(0.6, filters, hdr=True, chroma=True).args[0]

    assert filters == {
        "cas": "sharpness=0.6,hdr=true,chroma=true",
        "unsharp": "radius=3,weight=0.5,threshold=10",
        "edgelevel": "strength=5,threshold=20,black=2,white=1",
        "msharpen": "strength=0.8,threshold=10,slope=0,luma_limit=16,block_protect=0.5,highq=true",
        "detailsharpen": "z=4,sstr=1.5,power=4,ldmp=1,mode=1,med=false",
        "warpsharp": "threshold=128,blur=2,type=0,depth=16,chroma=0",
        "fruc": "double",
    }


def test_warpsharp_builds_adaptive_depth_parameters():
    filters = NVEncCWarpSharp.execute(160.0, 3, "5x5", 8.0, "per-channel", True, 4.0, 12.0, 192.0, 0.7).args[0]

    assert filters == {
        "warpsharp": "threshold=160,blur=3,type=1,depth=8,chroma=1,depth_min=4,depth_max=12,edge_thr=192,gamma=0.7",
    }


def test_ngx_vsr_node_composes_and_replaces_matching_stage():
    filters = NVEncCCAS.execute(0.3).args[0]
    filters = NVEncCNGXVSR.execute("height", 1080, 4, filters).args[0]
    filters = NVEncCNGXVSR.execute("width", 1920, 3, filters).args[0]

    assert filters == {
        "cas": "sharpness=0.3",
        "ngx_vsr": {
            "target_dimension": "width",
            "target_size": 1920,
            "quality": 3,
        },
    }


def test_resolve_ngx_vsr_preserves_aspect_ratio_and_builds_filter():
    filters = NVEncCNGXVSR.execute("height", 1080, 4).args[0]

    resolved, width, height = resolve_nvencc_filters(filters, 640, 360)

    assert (width, height) == (1920, 1080)
    assert resolved["ngx_vsr"] == {
        "parameters": "algo=ngx-vsr,vsr-quality=4",
        "output_res": "1920x1080",
    }


def test_resolve_ngx_vsr_rejects_invalid_targets():
    for filters, message in (
        (NVEncCNGXVSR.execute("height", 360, 4).args[0], "larger than"),
        (NVEncCNGXVSR.execute("height", 1081, 4).args[0], "even number"),
        (NVEncCNGXVSR.execute("depth", 1080, 4).args[0], "height or width"),
    ):
        try:
            resolve_nvencc_filters(filters, 640, 360)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid NGX VSR target must be rejected")


def test_build_command_adds_ngx_vsr_output_size(tmp_path):
    filters = NVEncCNGXVSR.execute("height", 1080, 4).args[0]
    filters, _, _ = resolve_nvencc_filters(filters, 640, 360)

    command = build_command("NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p4", 20.0, fruc=False, filters=filters)

    assert command[command.index("--output-res") + 1] == "1920x1080"
    assert command[command.index("--vpp-resize") + 1] == "algo=ngx-vsr,vsr-quality=4"


def test_build_command_emits_suite_filters_in_nvencc_order(tmp_path):
    filters = {
        "ngx_vsr": {
            "parameters": "algo=ngx-vsr,vsr-quality=4",
            "output_res": "1920x1080",
        },
        "fruc": "double",
        "detailsharpen": "z=4,sstr=1.5,power=4,ldmp=1,mode=1,med=false",
        "warpsharp": "threshold=128,blur=2,type=0,depth=16,chroma=0",
        "cas": "sharpness=0.4",
        "msharpen": "strength=0.8,threshold=10",
        "edgelevel": "strength=5,threshold=20,black=0,white=0",
        "unsharp": "radius=3,weight=0.5,threshold=10",
    }

    command = build_command("NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p4", 20.0, fruc=False, filters=filters)

    arguments = [command.index(name) for name in ("--vpp-resize", "--vpp-unsharp", "--vpp-edgelevel", "--vpp-msharpen", "--vpp-cas", "--vpp-detailsharpen", "--vpp-warpsharp", "--vpp-fruc")]
    assert arguments == sorted(arguments)


def test_build_command_rejects_unknown_suite_filter(tmp_path):
    try:
        build_command("NVEncC64.exe", str(tmp_path / "output.mkv"), "h264", "p4", 20.0, fruc=False, filters={"unknown": "value"})
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown suite filters must be rejected")


def test_suite_save_node_applies_filter_chain_to_native_video(monkeypatch, tmp_path):
    class Video:
        def get_dimensions(self):
            return 960, 544

        def get_stream_source(self):
            return str(tmp_path / "input.mp4")

        def get_active_trim_window(self):
            return 0.0, 0.0

        def get_frame_rate(self):
            return 24.0

    commands = []
    filters = NVEncCFrameDouble.execute(NVEncCMSharpen.execute(0.8, 10.0, 0.0, 16.0, 0.5, True).args[0]).args[0]
    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(nvencc_module.folder_paths, "get_save_image_path", lambda *args: (str(tmp_path), "test", 1, "", "test"))
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", commands.append)

    SaveVideoNVEncC.execute(Video(), 24.0, "test", "mp4", "h264", "p4", 20.0, "hardware", "", filters)

    command = commands[0]
    assert command[:4] == ["NVEncC64.exe", "--avhw", "-i", str(tmp_path / "input.mp4")]
    assert command[command.index("--vpp-msharpen") + 1].startswith("strength=0.8")
    assert command[command.index("--vpp-fruc") + 1] == "double"
    assert "--audio-copy" in command


def test_suite_save_node_resolves_ngx_vsr_output_dimensions(monkeypatch, tmp_path):
    class Video:
        def get_dimensions(self):
            return 640, 360

        def get_stream_source(self):
            return str(tmp_path / "input.mp4")

        def get_active_trim_window(self):
            return 0.0, 0.0

        def get_frame_rate(self):
            return 24.0

    commands = []
    save_dimensions = []
    filters = NVEncCNGXVSR.execute("height", 1080, 4).args[0]
    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(
        nvencc_module.folder_paths,
        "get_save_image_path",
        lambda prefix, output_dir, width, height: (save_dimensions.append((width, height)) or (str(tmp_path), "test", 1, "", "test")),
    )
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", commands.append)

    SaveVideoNVEncC.execute(Video(), 24.0, "test", "mp4", "h264", "p4", 20.0, "hardware", "", filters)

    assert save_dimensions == [(1920, 1080)]
    command = commands[0]
    assert command[command.index("--output-res") + 1] == "1920x1080"
    assert command[command.index("--vpp-resize") + 1] == "algo=ngx-vsr,vsr-quality=4"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for GIF output")
def test_gif_video_round_trip_with_explicit_dithering(monkeypatch, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(nvencc_module.folder_paths, "input_directory", str(input_dir))
    monkeypatch.setattr(nvencc_module.folder_paths, "output_directory", str(output_dir))
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=256x256:rate=10:duration=0.3", "-y", str(input_dir / "source.gif")], check=True)

    schema = LoadGIFNVEncC.define_schema()
    assert "source.gif" in schema.inputs[0].options
    video = LoadGIFNVEncC.execute("source.gif").args[0]
    assert video.get_dimensions() == (256, 256)

    commands = []
    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", commands.append)
    SaveVideoNVEncC.execute(video, 24.0, "nvencc", "mp4", "h264", "p4", 20.0, "hardware", "")
    assert commands[0][1:4] == ["--avsw", "-i", str(input_dir / "source.gif")]

    saved = SaveGIFWithFFmpeg.execute(video, "bayer", 2, "gif", ffmpeg).args[0]
    assert saved.get_dimensions() == (256, 256)
    with nvencc_module.av.open(saved.get_stream_source()) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3

    output_gifs = list(output_dir.rglob("*.gif"))
    converted = ConvertVideoToGIF.execute(video, "sierra2_4a", 2, ffmpeg).args[0]
    assert isinstance(converted.get_stream_source(), io.BytesIO)
    assert converted.get_container_format() == "gif"
    assert list(output_dir.rglob("*.gif")) == output_gifs
    with nvencc_module.av.open(converted.get_stream_source()) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3

    def check_buffered_gif(command):
        assert command[1] == "--avsw"
        with nvencc_module.av.open(command[command.index("-i") + 1]) as container:
            assert container.format.name == "gif"

    monkeypatch.setattr(nvencc_module, "run_nvencc_video", check_buffered_gif)
    SaveVideoNVEncC.execute(converted, 24.0, "converted", "mp4", "h264", "p4", 20.0, "hardware", "")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for GIF output")
def test_gif_converter_keeps_source_audio_in_video_connection(monkeypatch, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    source_path = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=256x256:rate=10:duration=0.5", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=0.5", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", "-y", str(source_path)], check=True)
    source = nvencc_module.InputImpl.VideoFromFile(str(source_path))

    converted = ConvertVideoToGIF.execute(source, "sierra2_4a", 2, ffmpeg).args[0]
    assert isinstance(converted.get_stream_source(), io.BytesIO)
    with nvencc_module.av.open(converted.get_stream_source()) as container:
        assert container.streams.video[0].codec_context.name == "ffv1"
        assert container.streams.audio[0].codec_context.name == "aac"

    def check_command(command):
        assert command[1] == "--avsw"
        assert "--audio-copy" in command
        with nvencc_module.av.open(command[command.index("-i") + 1]) as container:
            assert container.streams.video[0].codec_context.name == "ffv1"
            assert container.streams.audio[0].codec_context.name == "aac"

    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(nvencc_module.folder_paths, "get_save_image_path", lambda *args: (str(tmp_path), "test", 1, "", "test"))
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", check_command)
    SaveVideoNVEncC.execute(converted, 24.0, "test", "mp4", "h264", "p4", 20.0, "hardware", "")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required to pad odd GIF dimensions")
def test_odd_gif_is_padded_before_nvencc(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(nvencc_module.folder_paths, "input_directory", str(input_dir))
    monkeypatch.setattr(nvencc_module.folder_paths, "output_directory", str(output_dir))
    first = Image.new("RGB", (255, 255), "red")
    second = Image.new("RGB", (255, 255), "blue")
    first.save(input_dir / "odd.gif", save_all=True, append_images=[second], duration=100, loop=0)
    video = LoadGIFNVEncC.execute("odd.gif").args[0]

    def check_padded_input(command):
        assert command[1] == "--avsw"
        with nvencc_module.av.open(command[command.index("-i") + 1]) as container:
            assert (container.streams.video[0].width, container.streams.video[0].height) == (256, 256)

    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", check_padded_input)
    SaveVideoNVEncC.execute(video, 24.0, "odd", "mp4", "h264", "p4", 20.0, "hardware", "")


def test_execute_maps_independent_sharpening_widgets_to_nvencc(monkeypatch, tmp_path):
    class Video:
        def get_dimensions(self):
            return 960, 544

        def get_stream_source(self):
            return str(tmp_path / "input.mp4")

        def get_active_trim_window(self):
            return 0.0, 0.0

        def get_frame_rate(self):
            return 24.0

    commands = []
    monkeypatch.setattr(nvencc_module, "find_nvencc", lambda path: "NVEncC64.exe")
    monkeypatch.setattr(nvencc_module.folder_paths, "get_save_image_path", lambda *args: (str(tmp_path), "test", 1, "", "test"))
    monkeypatch.setattr(nvencc_module, "run_nvencc_video", commands.append)

    SaveVideoNVEncCFRUC.execute(
        Video(), 24.0, "test", "mp4", "h264", "p4", 20.0, "", enable_fruc=False,
        enable_sharpen=False, enable_unsharp=True, unsharp_radius=5, unsharp_weight=1.25, unsharp_threshold=8.0,
        enable_msharpen=True, msharpen_strength=0.8, msharpen_threshold=20.0, msharpen_slope=1.5,
        msharpen_luma_limit=24.0, msharpen_block_protect=0.5, msharpen_high_quality=False,
        enable_detailsharpen=True, detailsharpen_zero_point=3.0, detailsharpen_strength=2.5,
        detailsharpen_power=5.0, detailsharpen_damping=2.0, detailsharpen_blur_mode="gaussian", detailsharpen_median=True,
    )

    command = commands[0]
    assert "--vpp-cas" not in command
    assert command[command.index("--vpp-unsharp") + 1] == "radius=5,weight=1.25,threshold=8"
    assert command[command.index("--vpp-msharpen") + 1] == "strength=0.8,threshold=20,slope=1.5,luma_limit=24,block_protect=0.5,highq=false"
    assert command[command.index("--vpp-detailsharpen") + 1] == "z=3,sstr=2.5,power=5,ldmp=2,mode=0,med=true"


def test_widget_schema_preserves_every_historical_field_and_appends_sharpen_filters():
    inputs = SaveVideoNVEncCFRUC.define_schema().inputs
    widget_ids = [input.id for input in inputs if input.id != "audio"]

    assert widget_ids == [
        "images", "fps", "filename_prefix", "container", "codec", "preset", "quality",
        "input_decoder", "nvencc_path", "sharpen", "enable_fruc", "enable_sharpen",
        "enable_unsharp", "unsharp_radius", "unsharp_weight", "unsharp_threshold",
        "enable_msharpen", "msharpen_strength", "msharpen_threshold", "msharpen_slope",
        "msharpen_luma_limit", "msharpen_block_protect", "msharpen_high_quality",
        "enable_detailsharpen", "detailsharpen_zero_point", "detailsharpen_strength",
        "detailsharpen_power", "detailsharpen_damping", "detailsharpen_blur_mode", "detailsharpen_median",
    ]
    assert inputs[widget_ids.index("sharpen")].default == 0.0
    assert inputs[widget_ids.index("enable_fruc")].default is True
    assert inputs[widget_ids.index("enable_sharpen")].default is True
    assert inputs[widget_ids.index("enable_unsharp")].default is False
    assert inputs[widget_ids.index("enable_msharpen")].default is False
    assert inputs[widget_ids.index("enable_detailsharpen")].default is False


def test_node_info_card_documents_visible_contracts_and_combo_values():
    schema = SaveVideoNVEncCFRUC.define_schema()
    assert schema.description
    for input in schema.inputs:
        assert input.tooltip, input.id
        for option in getattr(input, "options", None) or []:
            assert option.lower() in input.tooltip.lower(), (input.id, option)
    for output in schema.outputs:
        assert output.tooltip, output.id

    boolean_ids = {
        "enable_fruc", "enable_sharpen", "enable_unsharp", "enable_msharpen", "msharpen_high_quality",
        "enable_detailsharpen", "detailsharpen_median",
    }
    inputs = {input.id: input for input in schema.inputs}
    for input_id in boolean_ids:
        help_text = inputs[input_id].tooltip.lower()
        assert "true" in help_text and "false" in help_text, input_id


def test_suite_node_schemas_are_stable_and_fully_documented():
    expected_inputs = {
        SaveVideoNVEncC: ["images", "filters", "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "audio"],
        NVEncCFrameDouble: ["filters"],
        NVEncCNGXVSR: ["filters", "target_dimension", "target_size", "quality"],
        NVEncCCAS: ["filters", "strength", "hdr", "chroma"],
        NVEncCUnsharp: ["filters", "radius", "weight", "threshold"],
        NVEncCEdgeLevel: ["filters", "strength", "threshold", "black", "white"],
        NVEncCMSharpen: ["filters", "strength", "threshold", "slope", "luma_limit", "block_protect", "high_quality"],
        NVEncCDetailSharpen: ["filters", "zero_point", "strength", "power", "damping", "blur_mode", "median"],
        NVEncCWarpSharp: ["filters", "threshold", "blur", "blur_type", "depth", "chroma_mode", "adaptive_depth", "depth_min", "depth_max", "edge_threshold", "gamma"],
        SaveGIFWithFFmpeg: ["video", "dither", "bayer_scale", "filename_prefix", "ffmpeg_path"],
        ConvertVideoToGIF: ["video", "dither", "bayer_scale", "ffmpeg_path"],
    }

    for node, input_ids in expected_inputs.items():
        schema = node.define_schema()
        assert schema.description
        assert [input.id for input in schema.inputs] == input_ids
        assert all(input.tooltip for input in schema.inputs)
        assert all(output.tooltip for output in schema.outputs)
        for input in schema.inputs:
            for option in getattr(input, "options", None) or []:
                assert option.lower() in input.tooltip.lower(), (node.__name__, input.id, option)

    msharpen_inputs = {input.id: input for input in NVEncCMSharpen.define_schema().inputs}
    assert "true" in msharpen_inputs["high_quality"].tooltip.lower()
    assert "false" in msharpen_inputs["high_quality"].tooltip.lower()
    detail_inputs = {input.id: input for input in NVEncCDetailSharpen.define_schema().inputs}
    assert all(option in detail_inputs["blur_mode"].tooltip.lower() for option in ("box", "gaussian"))
    assert "true" in detail_inputs["median"].tooltip.lower()
    assert "false" in detail_inputs["median"].tooltip.lower()
    cas_inputs = {input.id: input for input in NVEncCCAS.define_schema().inputs}
    for input_id in ("hdr", "chroma"):
        assert "true" in cas_inputs[input_id].tooltip.lower()
        assert "false" in cas_inputs[input_id].tooltip.lower()
    warpsharp_inputs = {input.id: input for input in NVEncCWarpSharp.define_schema().inputs}
    assert "true" in warpsharp_inputs["adaptive_depth"].tooltip.lower()
    assert "false" in warpsharp_inputs["adaptive_depth"].tooltip.lower()
    gif_schema = SaveGIFWithFFmpeg.define_schema()
    gif_dither = next(input for input in gif_schema.inputs if input.id == "dither")
    assert all(option in gif_dither.tooltip for option in gif_dither.options)
    convert_schema = ConvertVideoToGIF.define_schema()
    convert_dither = next(input for input in convert_schema.inputs if input.id == "dither")
    assert all(option in convert_dither.tooltip for option in convert_dither.options)
    assert LoadGIFNVEncC.define_schema().description
    assert all(input.tooltip for input in LoadGIFNVEncC.define_schema().inputs)
    assert all(output.tooltip for output in LoadGIFNVEncC.define_schema().outputs)


def test_real_loader_registers_node_and_frontend(monkeypatch):
    import nodes

    module_path = Path(__file__).parents[1]
    expected_web_dir = os.path.abspath(module_path / "web")
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", nodes.NODE_CLASS_MAPPINGS.copy())
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", nodes.NODE_DISPLAY_NAME_MAPPINGS.copy())
    monkeypatch.setattr(nodes, "EXTENSION_WEB_DIRS", nodes.EXTENSION_WEB_DIRS.copy())

    assert asyncio.run(nodes.load_custom_node(str(module_path)))
    assert {
        "LoadGIFNVEncC", "SaveGIFWithFFmpeg", "ConvertVideoToGIF", "SaveVideoNVEncC", "NVEncCFrameDouble", "NVEncCNGXVSR", "NVEncCCAS", "NVEncCUnsharp", "NVEncCEdgeLevel", "NVEncCMSharpen",
        "NVEncCDetailSharpen", "NVEncCWarpSharp", "SaveVideoNVEncCFRUC", "NVEncCFRUCTailBridge",
    }.issubset(nodes.NODE_CLASS_MAPPINGS)
    for node_id in (
        "LoadGIFNVEncC", "SaveGIFWithFFmpeg", "ConvertVideoToGIF", "SaveVideoNVEncC", "NVEncCFrameDouble", "NVEncCNGXVSR", "NVEncCCAS", "NVEncCUnsharp", "NVEncCEdgeLevel", "NVEncCMSharpen",
        "NVEncCDetailSharpen", "NVEncCWarpSharp", "SaveVideoNVEncCFRUC", "NVEncCFRUCTailBridge",
    ):
        assert nodes.NODE_CLASS_MAPPINGS[node_id].RELATIVE_PYTHON_MODULE == f"custom_nodes.{module_path.name}"
    assert expected_web_dir in nodes.EXTENSION_WEB_DIRS.values()


def test_pyproject_identifies_the_public_suite():
    module_path = Path(__file__).parents[1]
    with open(module_path / "pyproject.toml", "rb") as config_file:
        config = tomllib.load(config_file)

    assert config["project"]["name"] == "nvencc-suite"
    assert config["tool"]["comfy"]["DisplayName"] == "NVEncC Suite"


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


def test_write_y4m_accepts_one_frame_for_encode_only():
    output = io.BytesIO()

    write_y4m(torch.zeros((1, 4, 6, 3)), 24.0, output)

    assert output.getvalue().count(b"FRAME\n") == 1


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

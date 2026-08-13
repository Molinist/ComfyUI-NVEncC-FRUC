import assert from "node:assert/strict";
import test from "node:test";

import {
    CURRENT_WIDGET_ORDER,
    LEGACY_WIDGET_ORDERS,
    migrateWidgetValues,
    normalizeWidgetValues,
    restoreWidgetValues,
    serializeWidgetValues,
} from "../web/widget_compat.mjs";

const defaults = {
    fps: 24.0,
    filename_prefix: "%date:yyyy-MM-dd%/upframes",
    container: "mp4",
    codec: "av1",
    preset: "p4",
    quality: 20.0,
    input_decoder: "hardware",
    nvencc_path: "",
    sharpen: 0.0,
    enable_fruc: true,
    enable_sharpen: true,
    enable_unsharp: false,
    unsharp_radius: 3,
    unsharp_weight: 0.5,
    unsharp_threshold: 10.0,
    enable_msharpen: false,
    msharpen_strength: 1.0,
    msharpen_threshold: 15.0,
    msharpen_slope: 0.0,
    msharpen_luma_limit: 0.0,
    msharpen_block_protect: 0.0,
    msharpen_high_quality: true,
    enable_detailsharpen: false,
    detailsharpen_zero_point: 4.0,
    detailsharpen_strength: 1.5,
    detailsharpen_power: 4.0,
    detailsharpen_damping: 1.0,
    detailsharpen_blur_mode: "box",
    detailsharpen_median: false,
};

function makeNode() {
    return {
        widgets: CURRENT_WIDGET_ORDER.map((name) => ({ name, value: defaults[name] })),
    };
}

const stableValues = [23.976, "renders/clip", "mkv", "hevc", "p7", 18.5];

test("freezes every known positional widget order", () => {
    assert.deepEqual([...LEGACY_WIDGET_ORDERS.original], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "nvencc_path",
    ]);
    assert.deepEqual([...LEGACY_WIDGET_ORDERS.directVideo], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path",
    ]);
    assert.deepEqual([...LEGACY_WIDGET_ORDERS.misplacedSharpen], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "sharpen", "input_decoder", "nvencc_path",
    ]);
    assert.deepEqual([...LEGACY_WIDGET_ORDERS.preFeatureToggles], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "sharpen",
    ]);
    assert.deepEqual([...LEGACY_WIDGET_ORDERS.independentCasFruc], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "sharpen",
        "enable_fruc", "enable_sharpen",
    ]);
    assert.deepEqual([...CURRENT_WIDGET_ORDER], [
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "sharpen",
        "enable_fruc", "enable_sharpen", "enable_unsharp", "unsharp_radius", "unsharp_weight", "unsharp_threshold",
        "enable_msharpen", "msharpen_strength", "msharpen_threshold", "msharpen_slope", "msharpen_luma_limit",
        "msharpen_block_protect", "msharpen_high_quality", "enable_detailsharpen", "detailsharpen_zero_point",
        "detailsharpen_strength", "detailsharpen_power", "detailsharpen_damping", "detailsharpen_blur_mode", "detailsharpen_median",
    ]);
});

test("restores the original positional layout and keeps new defaults", () => {
    const node = makeNode();
    restoreWidgetValues(node, { widgets_values: [...stableValues, "D:/tools/NVEncC64.exe"] });
    const values = Object.fromEntries(node.widgets.map(({ name, value }) => [name, value]));

    assert.equal(values.nvencc_path, "D:/tools/NVEncC64.exe");
    assert.equal(values.input_decoder, "hardware");
    assert.equal(values.sharpen, 0.0);
    assert.equal(values.enable_fruc, true);
    assert.equal(values.enable_sharpen, true);
    assert.equal(values.enable_unsharp, false);
    assert.equal(values.enable_msharpen, false);
    assert.equal(values.enable_detailsharpen, false);
});

test("migrates direct-video, misplaced-sharpen, and current arrays", () => {
    assert.deepEqual(migrateWidgetValues([...stableValues, "software", "old.exe"]), {
        fps: 23.976, filename_prefix: "renders/clip", container: "mkv", codec: "hevc", preset: "p7",
        quality: 18.5, input_decoder: "software", nvencc_path: "old.exe",
    });
    assert.deepEqual(migrateWidgetValues([...stableValues, 0.35, "software", "bad-layout.exe"]), {
        fps: 23.976, filename_prefix: "renders/clip", container: "mkv", codec: "hevc", preset: "p7",
        quality: 18.5, sharpen: 0.35, input_decoder: "software", nvencc_path: "bad-layout.exe",
    });
    assert.deepEqual(migrateWidgetValues([...stableValues, "software", "current.exe", 0.4]), {
        fps: 23.976, filename_prefix: "renders/clip", container: "mkv", codec: "hevc", preset: "p7",
        quality: 18.5, input_decoder: "software", nvencc_path: "current.exe", sharpen: 0.4,
    });
    assert.equal(migrateWidgetValues([...stableValues, "hardware", "hardware", 0.2]).nvencc_path, "hardware");
    assert.deepEqual(migrateWidgetValues([...stableValues, "software", "new.exe", 0.4, false, true]), {
        fps: 23.976, filename_prefix: "renders/clip", container: "mkv", codec: "hevc", preset: "p7",
        quality: 18.5, input_decoder: "software", nvencc_path: "new.exe", sharpen: 0.4,
        enable_fruc: false, enable_sharpen: true,
    });
    const previousNode = makeNode();
    restoreWidgetValues(previousNode, { widgets_values: [...stableValues, "software", "new.exe", 0.4, false, true] });
    assert.equal(previousNode.widgets.find(({ name }) => name === "enable_unsharp").value, false);
    assert.equal(previousNode.widgets.find(({ name }) => name === "enable_msharpen").value, false);
    assert.equal(previousNode.widgets.find(({ name }) => name === "enable_detailsharpen").value, false);
    const current = CURRENT_WIDGET_ORDER.map((name) => defaults[name]);
    current[CURRENT_WIDGET_ORDER.indexOf("enable_unsharp")] = true;
    current[CURRENT_WIDGET_ORDER.indexOf("detailsharpen_blur_mode")] = "gaussian";
    assert.equal(migrateWidgetValues(current).enable_unsharp, true);
    assert.equal(migrateWidgetValues(current).detailsharpen_blur_mode, "gaussian");
});

test("restores named values by field and leaves missing fields at defaults", () => {
    const node = makeNode();
    restoreWidgetValues(node, { widgets_values: { codec: "h264", quality: 17.0 } });
    const values = Object.fromEntries(node.widgets.map(({ name, value }) => [name, value]));

    assert.equal(values.codec, "h264");
    assert.equal(values.quality, 17.0);
    assert.equal(values.input_decoder, "hardware");
    assert.equal(values.sharpen, 0.0);
    assert.equal(values.enable_fruc, true);
    assert.equal(values.enable_sharpen, true);
    assert.equal(values.enable_unsharp, false);
    assert.equal(values.unsharp_radius, 3);
    assert.equal(values.enable_msharpen, false);
    assert.equal(values.enable_detailsharpen, false);

    const disabled = makeNode();
    restoreWidgetValues(disabled, { widgets_values: { enable_fruc: false, enable_sharpen: false } });
    assert.equal(disabled.widgets.find(({ name }) => name === "enable_fruc").value, false);
    assert.equal(disabled.widgets.find(({ name }) => name === "enable_sharpen").value, false);
});

test("repairs NaN and invalid stale values predictably", () => {
    assert.deepEqual(normalizeWidgetValues({
        fps: "not-a-number", quality: null, sharpen: Number.NaN, input_decoder: "gpu", codec: 7,
        enable_fruc: "false", enable_sharpen: "invalid", enable_unsharp: "true", unsharp_radius: 2.5,
        msharpen_strength: 7, detailsharpen_zero_point: 0, detailsharpen_blur_mode: "triangle",
    }), {
        fps: 24.0, quality: 20.0, sharpen: 0.0, input_decoder: "hardware", codec: "av1",
        enable_fruc: false, enable_sharpen: true, enable_unsharp: true, unsharp_radius: 3,
        msharpen_strength: 1.0, detailsharpen_zero_point: 4.0, detailsharpen_blur_mode: "box",
    });

    const node = makeNode();
    restoreWidgetValues(node, { widgets_values: [...stableValues, Number.NaN, "hardware", "legacy.exe"] });
    assert.equal(node.widgets.find(({ name }) => name === "sharpen").value, 0.0);
    assert.equal(node.widgets.find(({ name }) => name === "nvencc_path").value, "legacy.exe");
});

test("serializes widget values by name", () => {
    const node = makeNode();
    node.widgets.find(({ name }) => name === "sharpen").value = 0.25;
    node.widgets.push({ name: "ignored", value: 1, type: "button" });
    const info = {};

    serializeWidgetValues(node, info);

    assert.equal(Array.isArray(info.widgets_values), false);
    assert.equal(info.widgets_values.sharpen, 0.25);
    assert.equal(info.widgets_values.enable_fruc, true);
    assert.equal(info.widgets_values.enable_unsharp, false);
    assert.equal(Object.hasOwn(info.widgets_values, "ignored"), false);
});

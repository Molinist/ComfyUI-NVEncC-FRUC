export const CURRENT_WIDGET_ORDER = Object.freeze([
    "fps",
    "filename_prefix",
    "container",
    "codec",
    "preset",
    "quality",
    "input_decoder",
    "nvencc_path",
    "sharpen",
    "enable_fruc",
    "enable_sharpen",
    "enable_unsharp",
    "unsharp_radius",
    "unsharp_weight",
    "unsharp_threshold",
    "enable_msharpen",
    "msharpen_strength",
    "msharpen_threshold",
    "msharpen_slope",
    "msharpen_luma_limit",
    "msharpen_block_protect",
    "msharpen_high_quality",
    "enable_detailsharpen",
    "detailsharpen_zero_point",
    "detailsharpen_strength",
    "detailsharpen_power",
    "detailsharpen_damping",
    "detailsharpen_blur_mode",
    "detailsharpen_median",
]);

export const LEGACY_WIDGET_ORDERS = Object.freeze({
    original: Object.freeze([
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "nvencc_path",
    ]),
    directVideo: Object.freeze([
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path",
    ]),
    misplacedSharpen: Object.freeze([
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "sharpen", "input_decoder", "nvencc_path",
    ]),
    preFeatureToggles: Object.freeze([
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "sharpen",
    ]),
    independentCasFruc: Object.freeze([
        "fps", "filename_prefix", "container", "codec", "preset", "quality", "input_decoder", "nvencc_path", "sharpen",
        "enable_fruc", "enable_sharpen",
    ]),
});

const DEFAULTS = Object.freeze({
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
});

const COMBOS = Object.freeze({
    container: new Set(["mp4", "mkv"]),
    codec: new Set(["av1", "hevc", "h264"]),
    preset: new Set(["p1", "p2", "p3", "p4", "p5", "p6", "p7"]),
    input_decoder: new Set(["hardware", "software"]),
    detailsharpen_blur_mode: new Set(["box", "gaussian"]),
});

function isDecoder(value) {
    return COMBOS.input_decoder.has(value);
}

function isFiniteNumber(value) {
    return (typeof value === "number" || (typeof value === "string" && value.trim() !== ""))
        && Number.isFinite(Number(value));
}

export function legacyWidgetOrder(values) {
    if (values.length > LEGACY_WIDGET_ORDERS.independentCasFruc.length) return CURRENT_WIDGET_ORDER;
    if (values.length >= 10) return LEGACY_WIDGET_ORDERS.independentCasFruc;
    if (values.length === 9) {
        if (isDecoder(values[6])) return LEGACY_WIDGET_ORDERS.preFeatureToggles;
        if (isDecoder(values[7])) return LEGACY_WIDGET_ORDERS.misplacedSharpen;
        return LEGACY_WIDGET_ORDERS.preFeatureToggles;
    }
    if (values.length === 8) return LEGACY_WIDGET_ORDERS.directVideo;
    return LEGACY_WIDGET_ORDERS.original;
}

export function migrateWidgetValues(values) {
    if (!values || typeof values !== "object") return {};
    if (!Array.isArray(values)) return { ...values };

    const order = legacyWidgetOrder(values);
    const migrated = {};
    for (let index = 0; index < Math.min(order.length, values.length); index++) {
        migrated[order[index]] = values[index];
    }
    return migrated;
}

function normalizeNumber(value, fallback, min, max) {
    if (!isFiniteNumber(value)) return fallback;
    const number = Number(value);
    return number >= min && number <= max ? number : fallback;
}

function normalizeInteger(value, fallback, min, max) {
    const number = normalizeNumber(value, fallback, min, max);
    return Number.isInteger(number) ? number : fallback;
}

export function normalizeWidgetValues(values) {
    const normalized = { ...values };
    if (Object.hasOwn(normalized, "fps")) normalized.fps = normalizeNumber(normalized.fps, DEFAULTS.fps, 1.0, 240.0);
    if (Object.hasOwn(normalized, "quality")) normalized.quality = normalizeNumber(normalized.quality, DEFAULTS.quality, 0.0, 51.0);
    if (Object.hasOwn(normalized, "sharpen")) normalized.sharpen = normalizeNumber(normalized.sharpen, DEFAULTS.sharpen, 0.0, 1.0);
    if (Object.hasOwn(normalized, "unsharp_radius")) normalized.unsharp_radius = normalizeInteger(normalized.unsharp_radius, DEFAULTS.unsharp_radius, 1, 9);
    for (const [name, min, max] of [
        ["unsharp_weight", 0.0, 10.0],
        ["unsharp_threshold", 0.0, 255.0],
        ["msharpen_strength", 0.0, 1.0],
        ["msharpen_threshold", 0.0, 255.0],
        ["msharpen_slope", 0.0, Number.POSITIVE_INFINITY],
        ["msharpen_luma_limit", 0.0, 255.0],
        ["msharpen_block_protect", 0.0, 1.0],
        ["detailsharpen_zero_point", 0.001, 64.0],
        ["detailsharpen_strength", 0.0, 16.0],
        ["detailsharpen_power", 1.0, 16.0],
        ["detailsharpen_damping", 0.0, 1000.0],
    ]) {
        if (Object.hasOwn(normalized, name)) normalized[name] = normalizeNumber(normalized[name], DEFAULTS[name], min, max);
    }

    for (const name of ["filename_prefix", "nvencc_path"]) {
        if (Object.hasOwn(normalized, name) && typeof normalized[name] !== "string") normalized[name] = DEFAULTS[name];
    }
    for (const [name, choices] of Object.entries(COMBOS)) {
        if (Object.hasOwn(normalized, name) && !choices.has(normalized[name])) normalized[name] = DEFAULTS[name];
    }
    for (const name of [
        "enable_fruc", "enable_sharpen", "enable_unsharp", "enable_msharpen", "msharpen_high_quality",
        "enable_detailsharpen", "detailsharpen_median",
    ]) {
        if (!Object.hasOwn(normalized, name) || typeof normalized[name] === "boolean") continue;
        normalized[name] = normalized[name] === "true" ? true : normalized[name] === "false" ? false : DEFAULTS[name];
    }
    return normalized;
}

function widget(node, name) {
    return node.widgets?.find((candidate) => candidate.name === name);
}

export function restoreWidgetValues(node, info) {
    const values = normalizeWidgetValues(migrateWidgetValues(info?.widgets_values));
    for (const [name, value] of Object.entries(values)) {
        const target = widget(node, name);
        if (target) target.value = value;
    }
}

export function repairWidgetValues(node) {
    const values = {};
    for (const name of CURRENT_WIDGET_ORDER) {
        const target = widget(node, name);
        if (target) values[name] = target.value;
    }
    const normalized = normalizeWidgetValues(values);
    for (const [name, value] of Object.entries(normalized)) widget(node, name).value = value;
}

export function serializeWidgetValues(node, info) {
    info.widgets_values = {};
    for (const candidate of node.widgets ?? []) {
        if (!candidate.name || candidate.type === "button" || candidate.options?.serialize === false) continue;
        info.widgets_values[candidate.name] = candidate.value;
    }
}

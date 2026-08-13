import { app } from "../../scripts/app.js";
import { repairWidgetValues, restoreWidgetValues, serializeWidgetValues } from "./widget_compat.mjs";

app.registerExtension({
    name: "comfy.nvencc-fruc.widget-compat",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SaveVideoNVEncCFRUC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            repairWidgetValues(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            restoreWidgetValues(this, info);
            repairWidgetValues(this);
            return result;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = onSerialize?.apply(this, arguments);
            repairWidgetValues(this);
            serializeWidgetValues(this, info);
            return result;
        };
    },
});

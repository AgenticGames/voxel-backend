// Cave demo Web Worker: owns the WASM cave generator so multi-second
// generation never blocks the page. Speaks the same "endpoints" the old
// HTTP server did; app.js's apiFetch shim is the only caller.
"use strict";

importScripts("pkg/voxel_wasm.js");

var demo = null;
var initPromise = wasm_bindgen("pkg/voxel_wasm_bg.wasm").then(function () {
    demo = new wasm_bindgen.CaveDemo();
});

self.onmessage = function (e) {
    var id = e.data.id;
    var path = e.data.path;
    var body = e.data.body;

    initPromise.then(function () {
        var status = 200;
        var text;
        try {
            switch (path) {
                case "/api/generate":
                    text = demo.generate(body || "");
                    break;
                case "/api/mine":
                    text = demo.mine(body || "{}");
                    break;
                case "/api/place-water":
                    text = demo.place_water(body || "{}");
                    break;
                case "/api/sleep":
                    text = demo.sleep();
                    break;
                default:
                    status = 404;
                    text = "Not found";
            }
            // The Rust side reports expected failures as {"error": "..."};
            // surface them like the server did: HTTP 400 with a plain message.
            if (status === 200 && text.lastIndexOf('{"error"', 0) === 0) {
                status = 400;
                text = JSON.parse(text).error;
            }
        } catch (err) {
            status = 500;
            text = "Cave generator crashed: " + String(err);
        }
        self.postMessage({ id: id, status: status, text: text });
    }, function (err) {
        self.postMessage({ id: id, status: 500, text: "WASM init failed: " + String(err) });
    });
};

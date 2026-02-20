(function () {
    "use strict";

    // State
    var report = null;
    var scene, camera, renderer;
    var currentMesh = null;
    var wireframeMode = false;

    // Fly camera state
    var flySpeed = 30;
    var flyActive = false; // true while right mouse is held
    var flyKeys = { w: false, a: false, s: false, d: false, q: false, e: false };
    var euler = new THREE.Euler(0, 0, 0, "YXZ");
    var prevTime = performance.now();

    // Mining raycaster state
    var raycaster = new THREE.Raycaster();
    var mouseNDC = new THREE.Vector2();

    // Stored mesh transform for stable mining coords
    var meshCenter = null;  // THREE.Vector3
    var meshScale = 1;

    // Pool rendering group
    var poolGroup = null;

    // DOM refs
    var summaryTotal = document.getElementById("stat-total");
    var summaryPassed = document.getElementById("stat-passed");
    var summaryFailed = document.getElementById("stat-failed");
    var summaryRate = document.getElementById("stat-rate");
    var resultsBody = document.getElementById("results-body");
    var viewerContainer = document.getElementById("viewer-container");
    var viewerPlaceholder = document.getElementById("viewer-placeholder");
    var meshInfo = document.getElementById("mesh-info");
    var wireframeToggle = document.getElementById("wireframe-toggle");
    var runBatchBtn = document.getElementById("run-batch-btn");
    var batchStatus = document.getElementById("batch-status");
    var genBtn = document.getElementById("gen-btn");
    var genStatus = document.getElementById("gen-status");
    var customExports = document.getElementById("custom-exports");
    var mineMode = document.getElementById("mine-mode");
    var mineRadius = document.getElementById("mine-radius");
    var mineRadiusVal = document.getElementById("mine-radius-val");
    var toastContainer = document.getElementById("toast-container");

    // ── Presets (localStorage persistence) ────────────────────────────
    var PRESET_KEY = "voxel-presets";
    var PRESET_SETTING_IDS = [
        "gen-cavern-freq", "gen-cavern-threshold", "gen-detail-octaves",
        "gen-detail-persistence", "gen-warp-amplitude",
        "gen-worms-per-region", "gen-worm-radius-min", "gen-worm-radius-max",
        "gen-worm-step-length", "gen-worm-max-steps", "gen-worm-falloff-power",
        "gen-chunk-size", "gen-max-edge-length",
        "gen-sandstone-depth", "gen-granite-depth", "gen-basalt-depth", "gen-slate-depth",
        "gen-iron-band-freq", "gen-iron-noise-freq", "gen-iron-perturbation", "gen-iron-threshold",
        "gen-copper-freq", "gen-copper-threshold",
        "gen-malachite-freq", "gen-malachite-threshold",
        "gen-kimberlite-pipe-freq", "gen-kimberlite-pipe-threshold",
        "gen-diamond-freq", "gen-diamond-threshold",
        "gen-sulfide-freq", "gen-sulfide-threshold", "gen-tin-threshold",
        "gen-pyrite-freq", "gen-pyrite-threshold",
        "gen-quartz-freq", "gen-quartz-threshold",
        "gen-gold-threshold",
        "gen-geode-freq", "gen-geode-center-threshold",
        "gen-geode-shell-thickness", "gen-geode-hollow-factor",
        // Pool settings
        "gen-pools-enabled",
        "gen-pool-placement-freq", "gen-pool-placement-threshold",
        "gen-pool-chance", "gen-pool-min-area", "gen-pool-max-radius",
        "gen-pool-basin-depth", "gen-pool-rim-height",
        "gen-pool-lava-fraction", "gen-pool-lava-depth-max", "gen-pool-min-air-above",
        // Formation settings
        "gen-formations-enabled",
        "gen-form-placement-threshold", "gen-form-stalactite-chance",
        "gen-form-stalagmite-chance", "gen-form-flowstone-chance", "gen-form-column-chance",
        "gen-form-length-min", "gen-form-length-max", "gen-form-max-radius",
        "gen-form-min-air-gap", "gen-form-min-clearance",
        // Stress & collapse settings
        "gen-stress-gravity", "gen-stress-lateral", "gen-stress-vertical",
        "gen-stress-prop-radius", "gen-stress-max-collapse",
        "gen-collapse-wood", "gen-collapse-metal", "gen-collapse-reinforce",
        "gen-collapse-stress-mult", "gen-collapse-max-cascade", "gen-collapse-rubble",
    ];
    var activePresetSlot = 0;

    function loadPresetsFromStorage() {
        try {
            var raw = localStorage.getItem(PRESET_KEY);
            return raw ? JSON.parse(raw) : {};
        } catch (e) { return {}; }
    }

    function savePresetsToStorage(presets) {
        localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
    }

    function collectSettings() {
        var settings = {};
        for (var i = 0; i < PRESET_SETTING_IDS.length; i++) {
            var el = document.getElementById(PRESET_SETTING_IDS[i]);
            if (el) settings[PRESET_SETTING_IDS[i]] = el.value;
        }
        return settings;
    }

    function applySettings(settings) {
        for (var id in settings) {
            var el = document.getElementById(id);
            if (el) el.value = settings[id];
        }
    }

    function clearAllSettings() {
        for (var i = 0; i < PRESET_SETTING_IDS.length; i++) {
            var el = document.getElementById(PRESET_SETTING_IDS[i]);
            if (el) el.value = "";
        }
    }

    function refreshPresetSlots() {
        var presets = loadPresetsFromStorage();
        var slots = document.querySelectorAll(".preset-slot");
        slots.forEach(function (btn) {
            var slot = btn.getAttribute("data-slot");
            var data = presets[slot];
            btn.classList.toggle("filled", !!data);
            btn.classList.toggle("active", parseInt(slot) === activePresetSlot);
            if (data && data.name) {
                btn.title = data.name + " — click to load";
            } else {
                btn.title = "Empty — click to load, select then Save to store";
            }
        });
    }

    // Slot click = select + load (empty slot clears all fields to defaults)
    document.querySelectorAll(".preset-slot").forEach(function (btn) {
        btn.addEventListener("click", function () {
            activePresetSlot = parseInt(this.getAttribute("data-slot"));
            var presets = loadPresetsFromStorage();
            var data = presets[activePresetSlot];
            clearAllSettings();
            if (data && data.settings) {
                applySettings(data.settings);
            }
            refreshPresetSlots();
        });
    });

    // Save button
    document.getElementById("preset-save").addEventListener("click", function () {
        var presets = loadPresetsFromStorage();
        var existing = presets[activePresetSlot];
        var defaultName = existing && existing.name ? existing.name : "Preset " + (activePresetSlot + 1);
        var name = prompt("Preset name:", defaultName);
        if (name === null) return;
        presets[activePresetSlot] = {
            name: name || defaultName,
            settings: collectSettings(),
        };
        savePresetsToStorage(presets);
        refreshPresetSlots();
    });

    // Delete button
    document.getElementById("preset-delete").addEventListener("click", function () {
        var presets = loadPresetsFromStorage();
        if (!presets[activePresetSlot]) return;
        if (!confirm("Delete preset \"" + (presets[activePresetSlot].name || "Preset " + (activePresetSlot + 1)) + "\"?")) return;
        delete presets[activePresetSlot];
        savePresetsToStorage(presets);
        refreshPresetSlots();
    });

    // Reset button — clear all settings to defaults (empty = use placeholders)
    document.getElementById("preset-reset").addEventListener("click", function () {
        clearAllSettings();
    });

    refreshPresetSlots();

    // ── Mine mode UI wiring ──────────────────────────────────────────
    mineMode.addEventListener("change", function () {
        if (mineMode.value === "sphere") {
            mineRadius.value = 5;
            mineRadiusVal.textContent = "5";
        } else {
            mineRadius.value = 2;
            mineRadiusVal.textContent = "2";
        }
    });

    mineRadius.addEventListener("input", function () {
        mineRadiusVal.textContent = mineRadius.value;
    });

    // ── Three.js setup ──────────────────────────────────────────────
    function initThree() {
        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
        camera.position.set(30, 25, 30);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setClearColor(0x12121f);
        renderer.setPixelRatio(window.devicePixelRatio);
        viewerContainer.appendChild(renderer.domElement);

        // Lights
        var ambient = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambient);
        var dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(50, 80, 50);
        scene.add(dir);
        var dir2 = new THREE.DirectionalLight(0x4fc3f7, 0.3);
        dir2.position.set(-40, 30, -40);
        scene.add(dir2);

        // Grid
        var grid = new THREE.GridHelper(100, 40, 0x0f3460, 0x0a0a1a);
        scene.add(grid);

        // Initialize euler from camera
        euler.setFromQuaternion(camera.quaternion);

        resizeViewer();
        window.addEventListener("resize", resizeViewer);
        initFlyCamera();
        initMining();
        animate();
    }

    function resizeViewer() {
        var w = viewerContainer.clientWidth;
        var h = viewerContainer.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }

    function animate() {
        requestAnimationFrame(animate);
        var now = performance.now();
        var dt = (now - prevTime) / 1000;
        prevTime = now;

        // Fly camera movement
        if (flyActive) {
            var moveVec = new THREE.Vector3(0, 0, 0);
            if (flyKeys.w) moveVec.z -= 1;
            if (flyKeys.s) moveVec.z += 1;
            if (flyKeys.a) moveVec.x -= 1;
            if (flyKeys.d) moveVec.x += 1;
            if (flyKeys.e) moveVec.y += 1;
            if (flyKeys.q) moveVec.y -= 1;
            if (moveVec.lengthSq() > 0) {
                moveVec.normalize().multiplyScalar(flySpeed * dt);
                // Transform moveVec by camera orientation
                var forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
                var right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
                var up = new THREE.Vector3(0, 1, 0);
                camera.position.addScaledVector(right, moveVec.x);
                camera.position.addScaledVector(up, moveVec.y);
                camera.position.addScaledVector(forward, -moveVec.z);
            }
        }

        renderer.render(scene, camera);
    }

    // ── Fly Camera ───────────────────────────────────────────────────
    function initFlyCamera() {
        var canvas = renderer.domElement;

        // Right-click: engage fly mode
        canvas.addEventListener("mousedown", function (e) {
            if (e.button === 2) {
                flyActive = true;
                canvas.requestPointerLock();
                e.preventDefault();
            }
        });

        canvas.addEventListener("mouseup", function (e) {
            if (e.button === 2) {
                flyActive = false;
                flyKeys.w = false;
                flyKeys.a = false;
                flyKeys.s = false;
                flyKeys.d = false;
                flyKeys.q = false;
                flyKeys.e = false;
                if (document.pointerLockElement === canvas) {
                    document.exitPointerLock();
                }
            }
        });

        // Prevent context menu on the canvas
        canvas.addEventListener("contextmenu", function (e) {
            e.preventDefault();
        });

        // Mouselook while pointer is locked
        document.addEventListener("mousemove", function (e) {
            if (document.pointerLockElement !== canvas) return;
            var sensitivity = 0.002;
            euler.setFromQuaternion(camera.quaternion);
            euler.y -= e.movementX * sensitivity;
            euler.x -= e.movementY * sensitivity;
            // Clamp pitch to avoid flipping
            euler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, euler.x));
            camera.quaternion.setFromEuler(euler);
        });

        // WASD + Q/E while flying
        document.addEventListener("keydown", function (e) {
            if (!flyActive) return;
            var key = e.key.toLowerCase();
            if (key in flyKeys) {
                flyKeys[key] = true;
                e.preventDefault();
            }
        });

        document.addEventListener("keyup", function (e) {
            var key = e.key.toLowerCase();
            if (key in flyKeys) {
                flyKeys[key] = false;
            }
        });

        // Scroll wheel to adjust fly speed
        canvas.addEventListener("wheel", function (e) {
            flySpeed += (e.deltaY > 0 ? -5 : 5);
            flySpeed = Math.max(1, Math.min(200, flySpeed));
            e.preventDefault();
        }, { passive: false });
    }

    // ── Mining interaction ────────────────────────────────────────────
    function initMining() {
        var canvas = renderer.domElement;

        canvas.addEventListener("click", function (e) {
            // Only mine on left-click when NOT flying
            if (flyActive) return;
            if (!currentMesh) return;

            var rect = canvas.getBoundingClientRect();
            mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouseNDC, camera);

            // Collect meshes to test
            var meshes = [];
            currentMesh.traverse(function (child) {
                if (child.isMesh) meshes.push(child);
            });

            var intersects = raycaster.intersectObjects(meshes, false);
            if (intersects.length === 0) return;

            var hit = intersects[0];

            // Convert hit point from world space back to original voxel coordinates.
            // Transform chain: original → translate(-center) → scale → world
            // Inverse: world → divide by scale → add center
            var originalPoint = hit.point.clone().multiplyScalar(1 / meshScale).add(meshCenter);

            // Face normal is in geometry-local space which matches original coords
            // (only uniform scale was applied, so normals are unchanged)
            var nx = hit.face.normal.x;
            var ny = hit.face.normal.y;
            var nz = hit.face.normal.z;

            var mode = mineMode.value;
            var radius = parseInt(mineRadius.value, 10);

            var body = JSON.stringify({
                x: originalPoint.x,
                y: originalPoint.y,
                z: originalPoint.z,
                mode: mode,
                radius: radius,
                nx: nx,
                ny: ny,
                nz: nz
            });

            meshInfo.textContent = "Mining...";
            fetch("/api/mine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: body
            })
            .then(function (resp) {
                if (!resp.ok) throw new Error("Mine request failed");
                return resp.json();
            })
            .then(function (data) {
                displayJsonMesh(data.mesh, { resetCamera: false, reuseTransform: true });
                renderPoolSurfaces(data.pools);
                showMineToast(data.mined);
            })
            .catch(function (err) {
                meshInfo.textContent = "Mine error: " + err.message;
            });
        });
    }

    // ── Deep Sleep ─────────────────────────────────────────────────
    var sleepBtn = document.getElementById("sleep-btn");
    var toggleBeforeAfterBtn = document.getElementById("toggle-before-after-btn");
    var sleepLog = document.getElementById("sleep-log");
    var sleepDiff = document.getElementById("sleep-diff");
    var sleepStatus = document.getElementById("sleep-status");
    var preSleepMeshData = null;   // JSON mesh data snapshot before sleep
    var postSleepMeshData = null;  // JSON mesh data snapshot after sleep
    var showingBefore = false;     // toggle state for before/after

    function renderSleepLog(transformLog) {
        sleepLog.innerHTML = "";
        if (!transformLog || transformLog.length === 0) {
            sleepLog.style.display = "none";
            return;
        }
        for (var i = 0; i < transformLog.length; i++) {
            var entry = transformLog[i];
            var div = document.createElement("div");
            div.className = "sleep-log-entry";
            div.innerHTML = '<span class="sleep-log-count">' + entry.count + 'x</span> ' + escapeHtml(entry.description);
            sleepLog.appendChild(div);
        }
        sleepLog.style.display = "block";
    }

    function renderSleepDiff(materialDiff) {
        sleepDiff.innerHTML = "";
        if (!materialDiff || Object.keys(materialDiff).length === 0) {
            sleepDiff.style.display = "none";
            return;
        }

        // Sort materials: losses first (negative), then gains (positive)
        var entries = [];
        for (var mat in materialDiff) {
            if (materialDiff.hasOwnProperty(mat)) {
                entries.push({ name: mat, diff: materialDiff[mat] });
            }
        }
        entries.sort(function (a, b) {
            return a.diff - b.diff; // negatives first
        });

        var html = '<table><thead><tr><th>Material</th><th>Change</th></tr></thead><tbody>';
        for (var i = 0; i < entries.length; i++) {
            var e = entries[i];
            var diffClass, diffText;
            if (e.diff > 0) {
                diffClass = "diff-positive";
                diffText = "+" + e.diff;
            } else if (e.diff < 0) {
                diffClass = "diff-negative";
                diffText = String(e.diff);
            } else {
                diffClass = "diff-zero";
                diffText = "0";
            }
            html += '<tr><td>' + escapeHtml(e.name) + '</td>'
                + '<td class="' + diffClass + '">' + diffText + '</td></tr>';
        }
        html += '</tbody></table>';
        sleepDiff.innerHTML = html;
        sleepDiff.style.display = "block";
    }

    function renderSleepStats(stats) {
        if (!stats) return;
        // Insert a stats summary bar after the log, before the diff
        var existing = document.querySelector(".sleep-stats");
        if (existing) existing.parentNode.removeChild(existing);

        var div = document.createElement("div");
        div.className = "sleep-stats";

        var items = [
            { label: "Chunks changed", value: stats.chunks_changed },
            { label: "Metamorphosed", value: stats.voxels_metamorphosed },
            { label: "Minerals grown", value: stats.minerals_grown },
            { label: "Supports degraded", value: stats.supports_degraded },
            { label: "Collapses", value: stats.collapses_triggered }
        ];

        for (var i = 0; i < items.length; i++) {
            if (items[i].value === undefined) continue;
            var span = document.createElement("span");
            span.className = "sleep-stat-item";
            span.innerHTML = items[i].label + ': <span class="sleep-stat-value">' + items[i].value + '</span>';
            div.appendChild(span);
        }

        // Insert after sleep-log
        sleepLog.parentNode.insertBefore(div, sleepDiff);
    }

    async function performSleep() {
        sleepBtn.disabled = true;
        sleepStatus.textContent = "Simulating deep sleep...";
        toggleBeforeAfterBtn.style.display = "none";
        showingBefore = false;
        toggleBeforeAfterBtn.classList.remove("showing-before");
        toggleBeforeAfterBtn.textContent = "Before/After";

        try {
            var resp = await fetch("/api/sleep", {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });
            if (!resp.ok) throw new Error("Sleep request failed (" + resp.status + ")");
            var data = await resp.json();

            // Store pre-sleep mesh (the mesh that was displayed before)
            // We need to capture it before we overwrite it
            // preSleepMeshData was already stored when generate or prior sleep ran

            // Store post-sleep mesh
            postSleepMeshData = data.mesh;

            // Display the post-sleep mesh
            if (data.mesh) {
                displayJsonMesh(data.mesh, { resetCamera: false, reuseTransform: true });
            }

            // Render transform log
            renderSleepLog(data.transform_log);

            // Render stats
            renderSleepStats(data.stats);

            // Render material diff
            renderSleepDiff(data.material_diff);

            // Show the before/after toggle if we have a pre-sleep snapshot
            if (preSleepMeshData) {
                toggleBeforeAfterBtn.style.display = "inline-block";
            }

            sleepStatus.textContent = "";
        } catch (err) {
            sleepStatus.textContent = "Error: " + err.message;
        } finally {
            sleepBtn.disabled = false;
        }
    }

    sleepBtn.addEventListener("click", function () {
        // Snapshot the current mesh data as pre-sleep state before performing sleep
        // We store it from the last generate or sleep result
        performSleep();
    });

    toggleBeforeAfterBtn.addEventListener("click", function () {
        if (!preSleepMeshData || !postSleepMeshData) return;

        showingBefore = !showingBefore;
        if (showingBefore) {
            displayJsonMesh(preSleepMeshData, { resetCamera: false, reuseTransform: true });
            toggleBeforeAfterBtn.textContent = "Showing: Before";
            toggleBeforeAfterBtn.classList.add("showing-before");
        } else {
            displayJsonMesh(postSleepMeshData, { resetCamera: false, reuseTransform: true });
            toggleBeforeAfterBtn.textContent = "Showing: After";
            toggleBeforeAfterBtn.classList.remove("showing-before");
        }
    });

    // ── Toast notifications ──────────────────────────────────────────
    function showMineToast(mined) {
        if (!mined || mined.length === 0) return;
        var parts = mined.map(function (m) {
            return m.count + " " + m.material;
        });
        var text = "Mined: " + parts.join(", ");

        var toast = document.createElement("div");
        toast.className = "mine-toast";
        toast.textContent = text;
        toastContainer.appendChild(toast);

        // Remove after animation completes
        setTimeout(function () {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 3100);
    }

    // ── Pool surface rendering ─────────────────────────────────────
    function clearPoolGroup() {
        if (poolGroup) {
            scene.remove(poolGroup);
            poolGroup.traverse(function (child) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            poolGroup = null;
        }
    }

    function renderPoolSurfaces(pools) {
        clearPoolGroup();
        if (!pools || pools.length === 0 || !meshCenter) return;

        poolGroup = new THREE.Group();

        for (var i = 0; i < pools.length; i++) {
            var pool = pools[i];
            var isLava = pool.fluid_type === "Lava";
            var color = isLava ? 0xff5014 : 0x1e78ff;
            var opacity = isLava ? 0.6 : 0.5;

            var geometry = new THREE.CircleGeometry(pool.radius, 24);
            // Rotate circle to lie flat (XZ plane) — default CircleGeometry faces +Z
            geometry.rotateX(-Math.PI / 2);

            var material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: opacity,
                side: THREE.DoubleSide,
                depthWrite: false,
            });

            var circle = new THREE.Mesh(geometry, material);
            // Position in original voxel coords, then apply meshCenter/meshScale transform
            circle.position.set(
                (pool.world_x - meshCenter.x) * meshScale,
                (pool.surface_y - meshCenter.y) * meshScale,
                (pool.world_z - meshCenter.z) * meshScale
            );
            circle.scale.setScalar(meshScale);
            poolGroup.add(circle);
        }

        scene.add(poolGroup);
    }

    // ── Display JSON mesh in the viewer ──────────────────────────────
    // opts: { resetCamera: bool, reuseTransform: bool }
    function displayJsonMesh(data, opts) {
        opts = opts || {};
        var resetCam = opts.resetCamera !== false;
        var reuseTransform = opts.reuseTransform === true;
        // Remove old mesh and free its resources
        if (currentMesh) {
            scene.remove(currentMesh);
            currentMesh.traverse(function (child) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(function (m) { m.dispose(); });
                    } else {
                        child.material.dispose();
                    }
                }
            });
            currentMesh = null;
        }
        clearPoolGroup();

        viewerPlaceholder.style.display = "none";

        // Build palette lookup: id -> {r, g, b} normalized
        var paletteLookup = {};
        if (data.palette) {
            for (var p = 0; p < data.palette.length; p++) {
                var entry = data.palette[p];
                var hex = entry.color;
                var r = parseInt(hex.substr(1, 2), 16) / 255;
                var g = parseInt(hex.substr(3, 2), 16) / 255;
                var b = parseInt(hex.substr(5, 2), 16) / 255;
                paletteLookup[entry.id] = { r: r, g: g, b: b };
            }
        }

        // Create geometry
        var geometry = new THREE.BufferGeometry();

        var positions = new Float32Array(data.positions);
        var normals = new Float32Array(data.normals);
        var indices = new Uint32Array(data.indices);

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        // Build per-vertex color from material_ids + palette
        var vertexCount = positions.length / 3;
        var colors = new Float32Array(vertexCount * 3);
        var materialIds = data.material_ids;
        for (var i = 0; i < vertexCount; i++) {
            var matId = materialIds[i];
            var col = paletteLookup[matId];
            if (col) {
                colors[i * 3] = col.r;
                colors[i * 3 + 1] = col.g;
                colors[i * 3 + 2] = col.b;
            } else {
                // Fallback: cyan
                colors[i * 3] = 0.31;
                colors[i * 3 + 1] = 0.76;
                colors[i * 3 + 2] = 0.97;
            }
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        var material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            flatShading: true,
            side: THREE.DoubleSide,
            wireframe: wireframeMode
        });

        var mesh = new THREE.Mesh(geometry, material);

        // Center and scale — reuse stored transform during mining so mesh doesn't jump
        if (!reuseTransform || !meshCenter) {
            geometry.computeBoundingBox();
            var box = geometry.boundingBox;
            meshCenter = box.getCenter(new THREE.Vector3());
            var size = box.getSize(new THREE.Vector3());
            var maxDim = Math.max(size.x, size.y, size.z);
            meshScale = maxDim > 0 ? 30 / maxDim : 1;
        }

        // Create a parent group to hold the transform
        var group = new THREE.Group();
        group.add(mesh);
        mesh.position.copy(meshCenter).negate();
        group.scale.setScalar(meshScale);

        scene.add(group);
        currentMesh = group;

        // Only reset camera on initial load, not during mining
        if (resetCam) {
            camera.position.set(30, 25, 30);
            camera.lookAt(0, 0, 0);
            euler.setFromQuaternion(camera.quaternion);
        }

        var totalTris = indices.length / 3;
        meshInfo.textContent = vertexCount.toLocaleString() + " vertices, "
            + Math.round(totalTris).toLocaleString() + " triangles";
    }

    // ── Shared: display OBJ text in the viewer (batch test compat) ──
    function displayObjText(text, label) {
        // Remove old mesh and free its resources
        if (currentMesh) {
            scene.remove(currentMesh);
            currentMesh.traverse(function (child) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(function (m) { m.dispose(); });
                    } else {
                        child.material.dispose();
                    }
                }
            });
            currentMesh = null;
        }

        var loader = new THREE.OBJLoader();
        var obj = loader.parse(text);

        // Apply material + count stats
        var totalVerts = 0;
        var totalTris = 0;
        obj.traverse(function (child) {
            if (child.isMesh) {
                child.material = new THREE.MeshPhongMaterial({
                    color: 0x4fc3f7,
                    wireframe: wireframeMode,
                    side: THREE.DoubleSide,
                    flatShading: true,
                });
                if (child.geometry) {
                    var pos = child.geometry.getAttribute("position");
                    if (pos) totalVerts += pos.count;
                    var idx = child.geometry.getIndex();
                    if (idx) {
                        totalTris += idx.count / 3;
                    } else if (pos) {
                        totalTris += pos.count / 3;
                    }
                }
            }
        });

        // Center and scale
        var box = new THREE.Box3().setFromObject(obj);
        var center = box.getCenter(new THREE.Vector3());
        var size = box.getSize(new THREE.Vector3());
        obj.position.sub(center);

        var maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0) {
            var scale = 30 / maxDim;
            obj.scale.setScalar(scale);
        }

        scene.add(obj);
        currentMesh = obj;

        camera.position.set(30, 25, 30);
        camera.lookAt(0, 0, 0);
        euler.setFromQuaternion(camera.quaternion);

        meshInfo.textContent = label + " | "
            + totalVerts.toLocaleString() + " vertices, "
            + Math.round(totalTris).toLocaleString() + " triangles";
    }

    // ── Report loading ──────────────────────────────────────────────
    async function loadReport() {
        try {
            var resp = await fetch("/api/report");
            if (!resp.ok) throw new Error("Failed to fetch report");
            report = await resp.json();
            renderSummary();
            renderTable();
        } catch (err) {
            resultsBody.innerHTML =
                '<tr><td colspan="4" style="color:#e94560;padding:20px">Error loading report: ' +
                err.message + "</td></tr>";
        }
    }

    function renderSummary() {
        summaryTotal.textContent = report.total_seeds;
        summaryPassed.textContent = report.passed;
        summaryFailed.textContent = report.failed;
        var rate = report.total_seeds > 0
            ? ((report.passed / report.total_seeds) * 100).toFixed(1) + "%"
            : "N/A";
        summaryRate.textContent = rate;
    }

    function renderTable() {
        // Support both new format (results array) and legacy (failures only)
        var allResults = report.results && report.results.length > 0
            ? report.results
            : null;

        var html = "";

        if (allResults) {
            for (var i = 0; i < allResults.length; i++) {
                var entry = allResults[i];
                var passed = entry.passed;
                var badge = passed
                    ? '<span class="badge badge-pass">pass</span>'
                    : '<span class="badge badge-fail">fail</span>';
                var reason = entry.reason ? escapeHtml(entry.reason) : "";
                var viewBtn = '<button class="btn-view" data-seed="' + entry.seed + '">View</button>';
                html += "<tr data-seed=\"" + entry.seed + "\">"
                    + "<td>" + entry.seed + "</td>"
                    + "<td>" + badge + "</td>"
                    + "<td>" + reason + "</td>"
                    + "<td>" + viewBtn + "</td>"
                    + "</tr>";
            }
        } else {
            var failMap = new Map();
            var failures = report.failures || [];
            for (var j = 0; j < failures.length; j++) {
                failMap.set(failures[j].seed, failures[j]);
            }
            for (var seed = 0; seed < report.total_seeds; seed++) {
                var failure = failMap.get(seed);
                var seedPassed = !failure;
                var seedBadge = seedPassed
                    ? '<span class="badge badge-pass">pass</span>'
                    : '<span class="badge badge-fail">fail</span>';
                var seedReason = failure ? escapeHtml(failure.reason) : "";
                var seedViewBtn = failure && failure.obj_path
                    ? '<button class="btn-view" data-seed="' + seed + '">View</button>'
                    : "";
                html += "<tr>"
                    + "<td>" + seed + "</td>"
                    + "<td>" + seedBadge + "</td>"
                    + "<td>" + seedReason + "</td>"
                    + "<td>" + seedViewBtn + "</td>"
                    + "</tr>";
            }
        }

        resultsBody.innerHTML = html;

        resultsBody.querySelectorAll(".btn-view").forEach(function (btn) {
            btn.addEventListener("click", function (e) {
                e.stopPropagation();
                var s = parseInt(this.getAttribute("data-seed"), 10);
                loadObj(s, this);
                clearExportSelection();
                resultsBody.querySelectorAll("tr").forEach(function (r) { r.classList.remove("selected"); });
                this.closest("tr").classList.add("selected");
            });
        });
    }

    // ── OBJ loading (on-demand, one at a time) ───────────────────────
    async function loadObj(seed, btn) {
        var origText = btn.textContent;
        btn.textContent = "Loading...";
        btn.disabled = true;
        meshInfo.textContent = "Loading seed " + seed + "...";
        viewerPlaceholder.style.display = "none";

        try {
            var resp = await fetch("/api/obj/" + seed);
            if (!resp.ok) throw new Error("OBJ not found for seed " + seed);
            var text = await resp.text();
            displayObjText(text, "Seed " + seed);
        } catch (err) {
            meshInfo.textContent = "Error: " + err.message;
        } finally {
            btn.textContent = origText;
            btn.disabled = false;
        }
    }

    // ── Custom OBJ file loading (multi-chunk exports) ────────────────
    async function loadObjFile(filename, btn) {
        var origText = btn.textContent;
        btn.textContent = "Loading...";
        btn.disabled = true;
        meshInfo.textContent = "Loading " + filename + "...";
        viewerPlaceholder.style.display = "none";

        try {
            var resp = await fetch("/api/obj-file/" + encodeURIComponent(filename));
            if (!resp.ok) throw new Error("File not found: " + filename);
            var text = await resp.text();
            displayObjText(text, filename);

            // Highlight this export item
            clearExportSelection();
            resultsBody.querySelectorAll("tr").forEach(function (r) { r.classList.remove("selected"); });
            btn.closest(".export-item").classList.add("selected");
        } catch (err) {
            meshInfo.textContent = "Error: " + err.message;
        } finally {
            btn.textContent = origText;
            btn.disabled = false;
        }
    }

    function clearExportSelection() {
        customExports.querySelectorAll(".export-item").forEach(function (el) {
            el.classList.remove("selected");
        });
    }

    // ── Load file list and render custom exports ─────────────────────
    async function loadCustomExports() {
        try {
            var resp = await fetch("/api/obj-files");
            if (!resp.ok) return;
            var files = await resp.json();
            renderCustomExports(files);
        } catch (err) {
            // silently ignore
        }
    }

    function renderCustomExports(files) {
        if (files.length === 0) {
            customExports.innerHTML = "";
            return;
        }
        var html = "";
        for (var i = 0; i < files.length; i++) {
            html += '<div class="export-item">'
                + '<span>' + escapeHtml(files[i]) + '</span>'
                + '<div class="export-actions">'
                + '<button class="btn-view" data-file="' + escapeHtml(files[i]) + '">View</button>'
                + '<button class="btn-delete" data-file="' + escapeHtml(files[i]) + '" title="Delete this file">&times;</button>'
                + '</div>'
                + '</div>';
        }
        customExports.innerHTML = html;

        customExports.querySelectorAll(".btn-view").forEach(function (btn) {
            btn.addEventListener("click", function () {
                var filename = this.getAttribute("data-file");
                loadObjFile(filename, this);
            });
        });

        customExports.querySelectorAll(".btn-delete").forEach(function (btn) {
            btn.addEventListener("click", async function (e) {
                e.stopPropagation();
                var filename = this.getAttribute("data-file");
                if (!confirm("Delete " + filename + "?")) return;
                try {
                    var resp = await fetch("/api/obj-file/" + encodeURIComponent(filename), {
                        method: "DELETE"
                    });
                    if (resp.ok) {
                        loadCustomExports(); // refresh file list
                    }
                } catch (err) {
                    console.error("Delete failed:", err);
                }
            });
        });
    }

    // ── Multi-chunk generate ─────────────────────────────────────────
    // Helper: append param to body parts only if the input has a value
    function appendParam(parts, key, elementId) {
        var el = document.getElementById(elementId);
        if (el && el.value !== "") {
            parts.push(key + "=" + encodeURIComponent(el.value));
        }
    }

    genBtn.addEventListener("click", async function () {
        var seed = document.getElementById("gen-seed").value || "1";
        var cx = document.getElementById("gen-x").value || "3";
        var cy = document.getElementById("gen-y").value || "3";
        var cz = document.getElementById("gen-z").value || "1";
        var closedEl = document.getElementById("gen-closed");

        genBtn.disabled = true;
        genStatus.textContent = "Generating " + cx + "x" + cy + "x" + cz + " (seed " + seed + ")...";

        try {
            var parts = [
                "seed=" + seed,
                "chunks_x=" + cx,
                "chunks_y=" + cy,
                "chunks_z=" + cz,
            ];
            if (closedEl && closedEl.checked) {
                parts.push("closed=1");
            }
            // Noise settings (only send if user filled in a value)
            appendParam(parts, "cavern_freq", "gen-cavern-freq");
            appendParam(parts, "cavern_threshold", "gen-cavern-threshold");
            appendParam(parts, "detail_octaves", "gen-detail-octaves");
            appendParam(parts, "detail_persistence", "gen-detail-persistence");
            appendParam(parts, "warp_amplitude", "gen-warp-amplitude");
            // Worm settings
            appendParam(parts, "worms_per_region", "gen-worms-per-region");
            appendParam(parts, "worm_radius_min", "gen-worm-radius-min");
            appendParam(parts, "worm_radius_max", "gen-worm-radius-max");
            appendParam(parts, "worm_step_length", "gen-worm-step-length");
            appendParam(parts, "worm_max_steps", "gen-worm-max-steps");
            appendParam(parts, "worm_falloff_power", "gen-worm-falloff-power");
            // Mesh quality settings
            appendParam(parts, "chunk_size", "gen-chunk-size");
            appendParam(parts, "max_edge_length", "gen-max-edge-length");
            // Host rock settings
            appendParam(parts, "sandstone_depth", "gen-sandstone-depth");
            appendParam(parts, "granite_depth", "gen-granite-depth");
            appendParam(parts, "basalt_depth", "gen-basalt-depth");
            appendParam(parts, "slate_depth", "gen-slate-depth");
            // Banded iron
            appendParam(parts, "iron_band_freq", "gen-iron-band-freq");
            appendParam(parts, "iron_noise_freq", "gen-iron-noise-freq");
            appendParam(parts, "iron_perturbation", "gen-iron-perturbation");
            appendParam(parts, "iron_threshold", "gen-iron-threshold");
            // Copper (dendritic)
            appendParam(parts, "copper_freq", "gen-copper-freq");
            appendParam(parts, "copper_threshold", "gen-copper-threshold");
            // Malachite
            appendParam(parts, "malachite_freq", "gen-malachite-freq");
            appendParam(parts, "malachite_threshold", "gen-malachite-threshold");
            // Kimberlite pipe
            appendParam(parts, "kimberlite_pipe_freq", "gen-kimberlite-pipe-freq");
            appendParam(parts, "kimberlite_pipe_threshold", "gen-kimberlite-pipe-threshold");
            appendParam(parts, "diamond_freq", "gen-diamond-freq");
            appendParam(parts, "diamond_threshold", "gen-diamond-threshold");
            // Sulfide blob
            appendParam(parts, "sulfide_freq", "gen-sulfide-freq");
            appendParam(parts, "sulfide_threshold", "gen-sulfide-threshold");
            appendParam(parts, "tin_threshold", "gen-tin-threshold");
            // Pyrite
            appendParam(parts, "pyrite_freq", "gen-pyrite-freq");
            appendParam(parts, "pyrite_threshold", "gen-pyrite-threshold");
            // Quartz reef
            appendParam(parts, "quartz_freq", "gen-quartz-freq");
            appendParam(parts, "quartz_threshold", "gen-quartz-threshold");
            // Gold
            appendParam(parts, "gold_threshold", "gen-gold-threshold");
            // Geode
            appendParam(parts, "geode_freq", "gen-geode-freq");
            appendParam(parts, "geode_center_threshold", "gen-geode-center-threshold");
            appendParam(parts, "geode_shell_thickness", "gen-geode-shell-thickness");
            appendParam(parts, "geode_hollow_factor", "gen-geode-hollow-factor");
            // Pool settings
            appendParam(parts, "pools_enabled", "gen-pools-enabled");
            appendParam(parts, "pool_placement_freq", "gen-pool-placement-freq");
            appendParam(parts, "pool_placement_threshold", "gen-pool-placement-threshold");
            appendParam(parts, "pool_chance", "gen-pool-chance");
            appendParam(parts, "pool_min_area", "gen-pool-min-area");
            appendParam(parts, "pool_max_radius", "gen-pool-max-radius");
            appendParam(parts, "pool_basin_depth", "gen-pool-basin-depth");
            appendParam(parts, "pool_rim_height", "gen-pool-rim-height");
            appendParam(parts, "pool_lava_fraction", "gen-pool-lava-fraction");
            appendParam(parts, "pool_lava_depth_max", "gen-pool-lava-depth-max");
            appendParam(parts, "pool_min_air_above", "gen-pool-min-air-above");
            // Formation settings
            appendParam(parts, "formations_enabled", "gen-formations-enabled");
            appendParam(parts, "form_placement_threshold", "gen-form-placement-threshold");
            appendParam(parts, "form_stalactite_chance", "gen-form-stalactite-chance");
            appendParam(parts, "form_stalagmite_chance", "gen-form-stalagmite-chance");
            appendParam(parts, "form_flowstone_chance", "gen-form-flowstone-chance");
            appendParam(parts, "form_column_chance", "gen-form-column-chance");
            appendParam(parts, "form_length_min", "gen-form-length-min");
            appendParam(parts, "form_length_max", "gen-form-length-max");
            appendParam(parts, "form_max_radius", "gen-form-max-radius");
            appendParam(parts, "form_min_air_gap", "gen-form-min-air-gap");
            appendParam(parts, "form_min_clearance", "gen-form-min-clearance");
            // Stress settings
            appendParam(parts, "stress_gravity", "gen-stress-gravity");
            appendParam(parts, "stress_lateral", "gen-stress-lateral");
            appendParam(parts, "stress_vertical", "gen-stress-vertical");
            appendParam(parts, "stress_prop_radius", "gen-stress-prop-radius");
            appendParam(parts, "stress_max_collapse", "gen-stress-max-collapse");
            // Sleep collapse settings
            appendParam(parts, "collapse_wood", "gen-collapse-wood");
            appendParam(parts, "collapse_metal", "gen-collapse-metal");
            appendParam(parts, "collapse_reinforce", "gen-collapse-reinforce");
            appendParam(parts, "collapse_stress_mult", "gen-collapse-stress-mult");
            appendParam(parts, "collapse_max_cascade", "gen-collapse-max-cascade");
            appendParam(parts, "collapse_rubble", "gen-collapse-rubble");
            var body = parts.join("&");
            var resp = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: body,
            });
            if (!resp.ok) throw new Error("Generate failed");
            var result = await resp.json();
            if (!result.ok) throw new Error(result.error || "Unknown error");

            genStatus.textContent = "Done! Loading mesh...";
            viewerPlaceholder.style.display = "none";

            // New JSON mesh path: display directly
            if (result.mesh) {
                var keepCam = document.getElementById("gen-keep-camera").checked;
                displayJsonMesh(result.mesh, { resetCamera: !keepCam });
                // Render pool surfaces
                renderPoolSurfaces(result.pools);
                // Snapshot for deep sleep before/after comparison
                preSleepMeshData = result.mesh;
                postSleepMeshData = null;
                showingBefore = false;
                toggleBeforeAfterBtn.style.display = "none";
                toggleBeforeAfterBtn.classList.remove("showing-before");
                sleepLog.style.display = "none";
                sleepDiff.style.display = "none";
                var oldStats = document.querySelector(".sleep-stats");
                if (oldStats) oldStats.parentNode.removeChild(oldStats);
                genStatus.textContent = "";
            } else {
                // Fallback: old OBJ file path (backward compat)
                await loadCustomExports();
                var viewBtn = customExports.querySelector('[data-file="' + result.filename + '"]');
                if (viewBtn) {
                    loadObjFile(result.filename, viewBtn);
                }
                genStatus.textContent = "";
            }
        } catch (err) {
            genStatus.textContent = "Error: " + err.message;
        } finally {
            genBtn.disabled = false;
        }
    });

    // ── Wireframe toggle ────────────────────────────────────────────
    wireframeToggle.addEventListener("change", function () {
        wireframeMode = this.checked;
        if (currentMesh) {
            currentMesh.traverse(function (child) {
                if (child.isMesh && child.material) {
                    child.material.wireframe = wireframeMode;
                }
            });
        }
    });

    // ── Run batch ───────────────────────────────────────────────────
    runBatchBtn.addEventListener("click", async function () {
        runBatchBtn.disabled = true;
        batchStatus.textContent = "Running batch test...";
        try {
            var resp = await fetch("/api/run-batch", { method: "POST" });
            if (!resp.ok) throw new Error("Batch run failed");
            await resp.text();
            batchStatus.textContent = "Batch complete. Reloading...";
            await loadReport();
            batchStatus.textContent = "Done.";
        } catch (err) {
            batchStatus.textContent = "Error: " + err.message;
        } finally {
            runBatchBtn.disabled = false;
        }
    });

    // ── Helpers ─────────────────────────────────────────────────────
    function escapeHtml(s) {
        var d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Init ────────────────────────────────────────────────────────
    initThree();
    loadReport();
    loadCustomExports();
})();

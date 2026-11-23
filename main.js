import * as THREE from "three";
import Stats from 'stats-gl';
import { Pane } from "./pane.js";
import * as Rsrc from './rsrc.js';
import { BvModel } from "./BvModel.js";
import { BvLoader } from "./BvLoader.js";

const panes = [];
let renderer, stats;

// gets shared with each pane;
// "what each pane gets to see of the app"
const app = {
    models: {},
    lastUsedControls: null,
    globalSettings: {},
};


async function initRenderer() {
    if (!navigator.gpu) {
        throw new Error();
    }
    const adapter = await navigator.gpu.requestAdapter();

    renderer = new THREE.WebGPURenderer({
        requiredLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
        trackTimestamp: false,
        antialias: true,
    });

    await renderer.init();
    if (renderer.backend.isWebGPUBackend !== true) {
        renderer.dispose();
        throw new Error();
    }
    renderer.setClearColor(0x000010, 1);
    renderer.setPixelRatio(window.devicePixelRatio);

    await Rsrc.init(renderer, adapter);

    // TODO: move this on the screen (or get rid of it)
    stats = new Stats({ trackGPU: true, });
    document.body.appendChild(stats.dom);
    stats.init(renderer);
}


// ==================== render frames! ====================
function frame() {
    panes.forEach(function (pane) {
        // const scene = pane.scene;
        const { scene, canvasTarget, camera, controls } = pane;

        controls.update();

        // the width/height of a particular canvasTarget can only be changed
        // when they are the renderer's current target, or else THREE explodes
        renderer.setCanvasTarget(canvasTarget);
        {
            const { width, height } = pane.element.getBoundingClientRect();
            canvasTarget.setSize(width, height, false);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }
        renderer.render(scene, camera);
    });

    stats.update();
}

// ==================== main ====================
{
    await initRenderer();
    // default model:
    {
        const bvLoader = new BvLoader(THREE.DefaultLoadingManager, renderer);

        const response = await fetch("./data/5/logo.bv");
        const text = await response.text();
        const model = bvLoader.parse(text);

        // two panes can display the same model without creating two copies of the geometry
        const pane = new Pane(app, model, 2);
        panes.push(pane);
        const pane2 = new Pane(app, model, 2);
        panes.push(pane2);
        document.getElementById('pane-container').style.setProperty('--cols', 2)
    }

    // ---------- ready to go ----------
    renderer.setAnimationLoop(frame);
}

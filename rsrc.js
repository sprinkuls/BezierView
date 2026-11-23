export const rsrc = {
    renderer: null,
    backend: null,
    device: null,
    quad: {
        evaluate: {
            shader: null,
            pipeline: null,
            bindGroupLayout: null,
        },
    },
    "tri": {
        "init": {

        },
        "subdiv": {

        },
    },
}

export async function fetchText(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`couldn't load ${url} (${response.status})`);
    }
    return response.text();
}

export async function init(renderer, adapter) {
    rsrc.adapter = adapter;
    rsrc.renderer = renderer;

    const backend = renderer.backend;
    rsrc.backend = backend;

    const device = backend.device;
    rsrc.device = device;


    // ========================================
    // BvQuad resources
    // ========================================

    // ========== evaluate resources ==========

    // TOOD: this
    rsrc.quad.evaluate.shader = rsrc.device.createShaderModule({
        label: 'Quad. evaluate shader module',
        code: await fetchText("evaluate_quad.wgsl")
    });
    rsrc.quad.evaluate.pipeline = rsrc.device.createComputePipeline({
        label: 'Quad. evaluate compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.evaluate.shader,
        },
    });
    rsrc.quad.evaluate.bindGroupLayout = rsrc.quad.evaluate.pipeline.getBindGroupLayout(0);
}

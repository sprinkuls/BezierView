// imports wgsl shader code
import * as Quad from './BvQuad.js';

export const rsrc = {
    backend: null,
    device: null,
    quad: {
        init: {
            shader: null,
            pipeline: null,
            bindGroupLayout: null,
        },
        subdiv: {
            uShader: null,
            uPipeline: null,
            uLayout: null,
            vShader: null,
            vPipeline: null,
            vLayout: null,
        },
    },
    "tri": {
        "init": {

        },
        "subdiv": {

        },
    },
}


export function init(renderer) {
    const backend = renderer.backend;
    rsrc.backend = backend;

    const device = backend.device;
    rsrc.device = device;

    ////////////////////
    // BvQuad resources
    ////////////////////

    // init resources
    rsrc.quad.init.shader = rsrc.device.createShaderModule({
        label: 'Quad. init. shader module',
        code: Quad.initPatchWGSL,
    });
    rsrc.quad.init.pipeline = rsrc.device.createComputePipeline({
        label: 'Quad. init. compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.init.shader,
        },
    });
    rsrc.quad.init.bindGroupLayout = rsrc.quad.init.pipeline.getBindGroupLayout(0);

    // subdiv resources

    // V-pass resources
    rsrc.quad.subdiv.vShader = rsrc.device.createShaderModule({
        label: 'V Pass shader module',
        code: Quad.vPassWGSL,
    });
    rsrc.quad.subdiv.vPipeline = rsrc.device.createComputePipeline({
        label: 'V Pass compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.subdiv.vShader,
        },
    });
    rsrc.quad.subdiv.vLayout = rsrc.quad.subdiv.vPipeline.getBindGroupLayout(0);
    // U-pass resources
    rsrc.quad.subdiv.uShader = rsrc.device.createShaderModule({
        label: 'U Pass shader module',
        code: Quad.uPassWGSL,
    });
    rsrc.quad.subdiv.uPipeline = rsrc.device.createComputePipeline({
        label: 'u Pass compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.subdiv.uShader,
        },
    });
    rsrc.quad.subdiv.uLayout = rsrc.quad.subdiv.uPipeline.getBindGroupLayout(0);

    ////////////////////
    // BvTri resources
    ////////////////////
}

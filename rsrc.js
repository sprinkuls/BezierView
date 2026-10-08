// imports wgsl shader code
import * as Quad from './BvQuad.js';

export const rsrc = {
    renderer: null,
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
        evaluate: {
            shader: null,
            pipeline: null,
            bindGroupLayout: null,
        },
        // TODO: remove this
        normal: {
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


export async function init(renderer, adapter) {
    rsrc.adapter = adapter;
    rsrc.renderer = renderer;

    const backend = renderer.backend;
    rsrc.backend = backend;

    const device = backend.device;
    rsrc.device = device;

    console.log('after alloc');

    // ========================================
    // BvQuad resources
    // ========================================

    // ========== subdiv resources ==========
    // V-pass
    // TODO: this
    const va = await fetch("v_pass.wgsl");
    const vb = await va.text();
    rsrc.quad.subdiv.vShader = rsrc.device.createShaderModule({
        label: 'V Pass shader module',
        // code: Quad.vPassWGSL,
        code: vb,
    });
    rsrc.quad.subdiv.vPipeline = rsrc.device.createComputePipeline({
        label: 'V Pass compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.subdiv.vShader,
        },
    });
    rsrc.quad.subdiv.vLayout = rsrc.quad.subdiv.vPipeline.getBindGroupLayout(0);

    // U-pass
    // TODO: this
    const ua = await fetch("u_pass.wgsl");
    const ub = await ua.text();
    rsrc.quad.subdiv.uShader = rsrc.device.createShaderModule({
        label: 'U Pass shader module',
        // code: Quad.uPassWGSL,
        code: ub,
    });
    rsrc.quad.subdiv.uPipeline = rsrc.device.createComputePipeline({
        label: 'u Pass compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.subdiv.uShader,
        },
    });
    rsrc.quad.subdiv.uLayout = rsrc.quad.subdiv.uPipeline.getBindGroupLayout(0);

    // ========== evaluate resources ==========

    // TOOD: this
    const ea = await fetch("evaluate_quad.wgsl");
    const eb = await ea.text();
    rsrc.quad.evaluate.shader = rsrc.device.createShaderModule({
        label: 'Quad. evaluate shader module',
        // code: Quad.evaluateWGSL,
        code: eb
    });
    rsrc.quad.evaluate.pipeline = rsrc.device.createComputePipeline({
        label: 'Quad. evaluate compute pipeline',
        layout: 'auto',
        compute: {
            module: rsrc.quad.evaluate.shader,
        },
    });
    rsrc.quad.evaluate.bindGroupLayout = rsrc.quad.evaluate.pipeline.getBindGroupLayout(0);


    // ========================================
    // BvQuad resources
    // ========================================
}

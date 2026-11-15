const SHADER = /* wgsl */`
@group(0) @binding(0) var<uniform> mvp : mat4x4f;

struct VOut { @builtin(position) clip : vec4f, @location(0) normal : vec3f };

@vertex fn vs_main(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> VOut {
    let v = ipass_vertex(vid, iid);
    return VOut(mvp * v.position, v.normal.xyz);
}

@fragment fn fs_main(in : VOut) -> @location(0) vec4f {
    let d = abs(dot(normalize(in.normal), normalize(vec3f(1.0, 1.0, 1.0)))); // two-sided
    return vec4f(vec3f(0.23, 0.66, 0.62) * (0.3 + 0.7 * d), 1.0);
}
`;


export const ipassSurface = {
    enabled: false,
    loaded : false,

    async init(device, canvas) {
        this.device   = device;
        this.context  = canvas.getContext('webgpu'); // alr configured by THREE
        this.ipass    = await createIPASS();
        this.pipeline = new this.ipass.DirectPipeline(device);

        this.mvpBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const module = device.createShaderModule({
            code: SHADER + this.ipass.DirectPipeline.getSurfaceShaderSource(),
        });
        const uniformLayout = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }],
        });
        this.bindGroup = device.createBindGroup({
            layout : uniformLayout,
            entries: [{ binding: 0, resource: { buffer: this.mvpBuffer } }],
        });
        this.renderPipeline = device.createRenderPipeline({
            // surface data is in ipass.SURFACE_BIND_GROUP
            layout: device.createPipelineLayout({
                bindGroupLayouts: [uniformLayout, this.pipeline.getSurfaceBindGroupLayout()],
            }),
            vertex  : { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }] },
            depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
        });
    },

    load(text) {
        const res = this.ipass.loadBVFile(new TextEncoder().encode(text).buffer);
        this.loaded = res.status === 0 && res.numPatches > 0 && this.pipeline.loadPatches(res.patchData()) === 0;
        if (!this.loaded) { console.warn('iPASS: failed to load patches'); }
    },

    render(camera) {
        if (!this.enabled || !this.loaded) return;

        const target = this.context.getCurrentTexture();
        if (this.depth?.width !== target.width || this.depth?.height !== target.height) {
            this.depth?.destroy();
            this.depth = this.device.createTexture({
                size: [target.width, target.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }

        const mvp = new Float32Array(camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).elements);
        this.pipeline.setMVP(mvp);
        this.pipeline.setViewport(target.width, target.height);
        this.device.queue.writeBuffer(this.mvpBuffer, 0, mvp);

        const encoder = this.device.createCommandEncoder();
        this.pipeline.execute(encoder);
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: target.createView(), loadOp: 'load', storeOp: 'store' }],
            depthStencilAttachment: {
                view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard',
            },
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.bindGroup);
        this.pipeline.draw(pass);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    },
};

import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float} from 'three/tsl';


export const initPatchWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cp_buffer : array<vec4<f32>>;

struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}
@group(0) @binding(2) var<uniform> p : patchInfo;

@compute @workgroup_size(1) fn init_patch() {
    // "highest (subdivision) level"
    const hl : u32 = 6;
    // (2 ^ hl), so for hl=6 this is 64
    const max_quads = 1 << hl;

    for (var i : u32 = 0; i <= p.u_deg; i++) {
        for (var j : u32 = 0; j <= p.v_deg; j++) {
            let src : u32 = ((p.v_deg+1) * i) + j;
            let dst : u32 = ((i * max_quads) * ((p.v_deg * max_quads) + 1)) + (j * max_quads);
            position_buffer[dst + p.index_offset] = cp_buffer[src];
        }
    }
}
`;

// TODO: when profiling either compute shader, the thing actually taking the most time
// (and by no small amount) is just memory accesses, since we keep reading/writing directly
// to the position buffer. seems to be roughly a 3 : 1 ratio of memory : calculations,
// address that!
export const vPassWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;

struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}
@group(0) @binding(1) var<uniform> p : patchInfo;

struct passInfo {
    nr_invocations : u32,
    curr_computed_level : u32,
}

@group(0) @binding(2) var<uniform> cur_pass : passInfo;


// "highest (subdivision) level"
const hl : u32 = 6;
// (2 ^ hl), so for hl=6 this is 64
const max_quads = 1 << hl;

@compute @workgroup_size(64, 1, 1) fn v_pass(
    @builtin( global_invocation_id ) globalId : vec3<u32>,
) {
    let instance_index : u32 = globalId.x;

    if (instance_index > cur_pass.nr_invocations - 1) {
        return;
    }


    // (2 ^ curr_level), so at curr_level=2 this is 4 quads per dimension
    let num_quads_per_dim : u32 = u32(1) << cur_pass.curr_computed_level;

    let stride : u32 = u32(1) << (hl - cur_pass.curr_computed_level);
    let halfstride : u32 = stride >> 1;

    // these are equivalent to below; just division and modulo by a power of 2
    // let grid_x : u32 = instance_index / num_quads_per_dim;
    // let grid_y : u32 = instance_index % num_quads_per_dim;
    let grid_x : u32 = instance_index >> cur_pass.curr_computed_level;
    let grid_y : u32 = instance_index & ((num_quads_per_dim) - 1);

    let x = grid_x * stride;
    let y = grid_y * stride * p.v_deg;
    let bw : u32 = (p.v_deg * max_quads) + 1;

    // we're just multiplying the x coordinate by the buffer width (since the
    // data is stored in row-major order), then adding the y coordinate.
    // pulled out because it gets used thrice every loop,
    // and doesn't change for the entire run of this shader.
    let base_index : u32 = p.index_offset + (bw * x);

    for (var i : u32 = 0; i < p.v_deg; i++) {
        for (var j : u32 = i; j < p.v_deg; j++) {
            let y_temp = y + (halfstride * i) + (stride * (j-i));

            let p1  = base_index + y_temp;
            let mid = base_index + y_temp + halfstride;
            let p2  = base_index + y_temp + stride;

            position_buffer[mid] = (position_buffer[p1] + position_buffer[p2]) * 0.5;
        }
    }
}
`;

export const uPassWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;

struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}
@group(0) @binding(1) var<uniform> p : patchInfo;

struct passInfo {
    nr_invocations : u32,
    curr_computed_level : u32,
}

@group(0) @binding(2) var<uniform> cur_pass : passInfo;

// "highest (subdivision) level"
const hl : u32 = 6;
// (2 ^ hl), so for hl=6 this is 64
const max_quads = 1 << hl;

@compute @workgroup_size(64, 1, 1) fn u_pass(
    @builtin( global_invocation_id ) globalId : vec3<u32>,
) {
    let instance_index : u32 = globalId.x;

    if (instance_index > cur_pass.nr_invocations - 1) {
        return;
    }


    // (2 ^ curr_level), so at curr_level=2 this is 4 quads per dimension
    let num_quads_per_dim : u32 = u32(1) << cur_pass.curr_computed_level;

    let stride : u32 = u32(1) << (hl - cur_pass.curr_computed_level);
    let halfstride : u32 = stride >> 1;

    // NOTE: the mod/div are swapped from v_pass!
    // TODO: i feel i should change that
    // let grid_x : u32 = instance_index % num_quads_per_dim;
    // let grid_y : u32 = instance_index / num_quads_per_dim;
    let grid_x : u32 = instance_index & ((num_quads_per_dim) - 1);
    let grid_y : u32 = instance_index >> cur_pass.curr_computed_level;

    let x = grid_x * stride * p.u_deg;
    let y = grid_y * halfstride;

    let bw : u32 = (p.v_deg * max_quads) + 1;

    // since the data is stored in row-major order, and the row being accessed changes
    // during runs of the inner loop, we can't pre-calculate the index as much as
    // we can in the v shader
    let base_index : u32 = p.index_offset + y;

    for (var i : u32 = 0; i < p.u_deg; i++) {
        for (var j : u32 = i; j < p.u_deg; j++) {
            let x_temp : u32 = x + ((halfstride * i) + (stride * (j-i)));

            // let p1  = p.index_offset + y + (bw * (x_temp));
            // let mid = p.index_offset + y + (bw * (x_temp + halfstride));
            // let p2  = p.index_offset + y + (bw * (x_temp + stride));
            let p1  = base_index + (bw * (x_temp));
            let mid = base_index + (bw * (x_temp + halfstride));
            let p2  = base_index + (bw * (x_temp + stride));

            position_buffer[mid] = (position_buffer[p1] + position_buffer[p2]) * 0.5;
        }
    }
}
`;

export class BvQuad {

    static indexBuffers = {};

    /**
     * @param {int} patchType
     * @param {int} uDeg
     * @param {int} vDeg
     * @param {THREE.Float32BufferAttribute} cpBuffer
     */
    constructor(patchType, uDeg, vDeg, cpBuffer) {
        this.patchType = patchType;
        this.uDeg = uDeg;
        this.vDeg = vDeg;
        this.cpBuffer = cpBuffer;

        // set later when this patch is assigned to a particular chunk in a BvGroup
        this.byteOffset = -1;
        this.indexOffset = -1;
        // set later when this patch is being initialized
        this.uniformBuffer = -1; // GPUBuffer that holds uDeg/vDeg/indexOffset
    }

    getMemNeededBytes() {
        // (number of vertices) * (4 bytes per float32) * (4 float32s per vertex)
        // (even if we request a vec3 we're always actually getting a vec4 under the hood)
        return (((this.uDeg * 64) + 1) * ((this.vDeg * 64) + 1)) * 4 * 4;
    }

    // Adds dispatches to the passed-in GPUComputePassEncoder that will
    // copy this patch's control points to the correct locations in the
    // position buffer.
    //
    // NOTE: this assumes that the correct pipeline has already been set!
    //       this is done because it allows the loop running this method on
    //       all patches to avoid adding unnecessary pass.setPipeline() calls;
    //       since this patch doesn't know what came before or after it, it
    //       doesn't have enough information to make that decision.
    //
    initPatch(rsrc, pass, chunkPosBuffer) {
        // The GPUBuffer for this patch's control points
        const patchCPBuffer = rsrc.backend.get(this.cpBuffer).buffer;

        // index_offset(u32) + u_deg(u32) + v_deg(u32)
        const uniformBufferSize = (4) + (4) + (4);
        const uniformBuffer = rsrc.device.createBuffer({
            label: `Patch info uniform (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const uniformValues = new Uint32Array([this.indexOffset, this.uDeg, this.vDeg]);
        rsrc.device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

        this.uniformBuffer = uniformBuffer;

        // TODO: cache these based on some key that isn't tied to indexOffset (and perhaps not to
        //       the chunk's position buffer either)
        const bindGroup = rsrc.device.createBindGroup({
            label: `Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
            layout: rsrc.quad.init.bindGroupLayout,
            entries: [
                {binding: 0, resource: chunkPosBuffer },
                {binding: 1, resource: patchCPBuffer },
                {binding: 2, resource: uniformBuffer },
            ]
        });

        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
    }


    // TODO: this should probably just take in an array plus a starting position
    // in that array and write to that, rather than return a new array that
    // we then go and append to another array
    /**
     * @param {int} level the tessellation level we want indices for
     * @returns {Uint32Array} Uint32Array with the indices, including offset
     */
    getIndexBuffer(level) {
        const key = `${level}-${this.uDeg}-${this.vDeg}`;
        if (key in BvQuad.indexBuffers) {
            const indexBufferCopy = new Uint32Array(BvQuad.indexBuffers[key]);
            for (let i = 0; i < indexBufferCopy.length; i++) {
                indexBufferCopy[i] += this.indexOffset;
            }
            return indexBufferCopy;
        }

        // console.log(`Index buffer cache miss: ${key}`)

        const quadsPerDim = 1 << level; // 2^n
        const stride = 1 << (6 - level);
        const uStride = this.uDeg * stride; // spacing in the u direction
        const vStride = this.vDeg * stride; // spacing in the v direction

        /*
         * data stored in u-major order
         *
         *    2   5
         *     x---x
         *     |   |
         *    1|  4|
         *     x---x
         *  ^  |   |
         *  | 0|  3|
         *  v  x---x
         *   u ->
         */
        const bufferSize = ((this.vDeg * 64) + 1);
        const flattenIndex = (i, j) => ((i * bufferSize) + j);

        const indices = new Uint32Array(quadsPerDim * quadsPerDim * 6);

        let idx = 0;
        for (let i = 0; i < quadsPerDim; i++) {
            for (let j = 0; j < quadsPerDim; j++) {
                const u = i * uStride;
                const v = j * vStride;

                const topLeft     = flattenIndex(u, v);
                const topRight    = flattenIndex(u, v + vStride);
                const bottomLeft  = flattenIndex(u + uStride, v);
                const bottomRight = flattenIndex(u + uStride, v + vStride);

                // Triangle 1
                indices[idx++] = bottomLeft;
                indices[idx++] = topRight;
                indices[idx++] = topLeft;
                // Triangle 2
                indices[idx++] = bottomLeft;
                indices[idx++] = bottomRight;
                indices[idx++] = topRight;
            }
        }
        BvQuad.indexBuffers[key] = indices;

        // only add offset AFTER caching the generated indices
        const indexBufferCopy = new Uint32Array(BvQuad.indexBuffers[key]);
        for (let i = 0; i < indexBufferCopy.length; i++) {
            indexBufferCopy[i] += this.indexOffset;
        }
        return indexBufferCopy;
    }


    subdivPatch({rsrc, pass, chunkPosBuffer, highestComputedLevel, newLevel}) {
        if (highestComputedLevel >= newLevel) {
            throw new Error();
        }

        for (let currLevel = highestComputedLevel; currLevel < newLevel; currLevel++) {
            // "number of quads per dimension"
            const nqpd = 1 << currLevel;

            // TODO: CACHE BIND GROUPS, AND MOVE SHADER INVOCATIONS OUTSIDE OF THEM (as in,
            // don't store shader invocations there, so we can cache the bind groups)
            // really, we should reuse bind groups for subdivision and for initialization,
            // since both use this 'uniformBuffer' and 'chunkPosBuffer'. we could just
            // put those in their own bind group which gets reused for everything, and then
            // have a second bind group that holds, like,
            // just the # of invocations, or just the CP buffer reference or something

            // TODO: we could save a LOT of pipeline changes by doing this by
            // patch type. maybe should consider storing patches in memory by patch type?
            // or e.g. by dispatching all U passes for all quads in the patch, then all
            // V passes, for a given level. would need to look into that far more because
            // that might mean worse performance overall even if there are fewer pipeline swaps


            // ====================
            // V Pass
            // ====================
            pass.setPipeline(rsrc.quad.subdiv.vPipeline);

            // uniform to send the # of invocations for this shader, along with the current
            // highest computed level
            const vShaderInvocations = ((nqpd*this.uDeg)+1)*nqpd;
            const vInvocationsBuffer = rsrc.device.createBuffer({
                size: 8, // two u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            rsrc.device.queue.writeBuffer(vInvocationsBuffer, 0, new Uint32Array([vShaderInvocations, currLevel]));

            const vBindGroup = rsrc.device.createBindGroup({
                label: `V Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
                layout: rsrc.quad.subdiv.vLayout,
                entries: [
                    {binding: 0, resource: chunkPosBuffer },
                    {binding: 1, resource: this.uniformBuffer },
                    {binding: 2, resource: vInvocationsBuffer },
                ]
            });
            pass.setBindGroup(0, vBindGroup);
            pass.dispatchWorkgroups(Math.ceil(vShaderInvocations / 64)); // (workgroup size is 64)



            // ====================
            // U Pass
            // ====================
            pass.setPipeline(rsrc.quad.subdiv.uPipeline);

            // uniform to send the # of invocations for this shader, along with the current
            // highest computed level
            const uShaderInvocations = ((nqpd*this.vDeg*2)+1)*nqpd;
            const uInvocationsBuffer = rsrc.device.createBuffer({
                size: 8, // two u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            rsrc.device.queue.writeBuffer(uInvocationsBuffer, 0, new Uint32Array([uShaderInvocations, currLevel]));

            const uBindGroup = rsrc.device.createBindGroup({
                label: `U Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
                layout: rsrc.quad.subdiv.uLayout,
                entries: [
                    {binding: 0, resource: chunkPosBuffer },
                    {binding: 1, resource: this.uniformBuffer },
                    {binding: 2, resource: uInvocationsBuffer },
                ]
            });
            pass.setBindGroup(0, uBindGroup);
            pass.dispatchWorkgroups(Math.ceil(uShaderInvocations / 64)); // (workgroup size is 64)
        }
    }
}

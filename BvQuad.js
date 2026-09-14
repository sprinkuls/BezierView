import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float} from 'three/tsl';


// TODO: write normals, curvature into buffers for level 0
export const initPatchWGSL = /* wgsl */ `
struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}

@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cp_buffer : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bb_buffer : array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p : patchInfo;

@compute @workgroup_size(1) fn init_patch() {
    // "highest (subdivision) level"
    const hl : u32 = 6;
    // (2 ^ hl), so for hl=6 this is 64
    const max_quads = 1 << hl;

    // write to position_buffer
    position_buffer[0 + p.index_offset] = cp_buffer[0];
    position_buffer[1 + p.index_offset] = cp_buffer[p.v_deg];
    position_buffer[2 + p.index_offset] = cp_buffer[(p.u_deg) * (p.v_deg+1)];
    position_buffer[3 + p.index_offset] = cp_buffer[(p.u_deg) * (p.v_deg+1) + p.v_deg];

    // position_buffer[0] = vec4<f32>(-1, -1, 0, 1);
    // position_buffer[1] = vec4<f32>(-1,  1, 1, 1);
    // position_buffer[2] = vec4<f32>(-1, -1, 2, 1);
    // position_buffer[3] = vec4<f32>( 1,  1, 3, 1);

    // write to bb_buffer
    for (var i : u32 = 0; i <= p.u_deg; i++) {
        for (var j : u32 = 0; j <= p.v_deg; j++) {
            let src : u32 = ((p.v_deg+1) * i) + j;
            let dst : u32 = ((i * max_quads) * ((p.v_deg * max_quads) + 1)) + (j * max_quads);

            bb_buffer[dst] = cp_buffer[src];
        }
    }
}
`;

// TODO: when profiling either compute shader, the thing actually taking the most time
// (and by no small amount) is just memory accesses, since we keep reading/writing directly
// to the position buffer. seems to be roughly a 3 : 1 ratio of memory : calculations,
// address that!
export const vPassWGSL = /* wgsl */ `
struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}

struct passInfo {
    nr_invocations : u32,
    curr_computed_level : u32,
}

@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bb_buffer : array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p : patchInfo;
@group(0) @binding(3) var<uniform> cur_pass : passInfo;


// "highest (subdivision) level"
const hl : u32 = 6;
// (2 ^ hl), so for hl=6 this is 64
const max_quads = 1 << hl;

const inside_patch_offset = array<u32, 7>(0, 4, 13, 38, 119, 408, 1497);

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
    // "buffer width", how many indices you need to move to increase the x coordinate by 1
    let bw : u32 = (p.v_deg * max_quads) + 1;

    // we're just multiplying the x coordinate by the buffer width (since the
    // data is stored in row-major order), then adding the y coordinate.
    // pulled out because it gets used thrice every loop,
    // and doesn't change for the entire run of this shader.
    //let base_index : u32 = p.index_offset + (bw * x);
    let base_index : u32 = (bw * x) + y;

    for (var i : u32 = 0; i < p.v_deg; i++) {
        for (var j : u32 = i; j < p.v_deg; j++) {
            let y_temp = (halfstride * i) + (stride * (j-i));

            let p1  = base_index + y_temp;
            let mid = base_index + y_temp + halfstride;
            let p2  = base_index + y_temp + stride;

            bb_buffer[mid] = (bb_buffer[p1] + bb_buffer[p2]) * 0.5;
        }
    }

    // copy data to the position buffer only if this shader invocation is somewhere
    // the results will show up at this level
    if (grid_x % p.u_deg == 0) {
        // both coords are multiplied by 2 in the v-pass shader.
        // the reason is, since the v-pass runs first (like, we run the v-pass, then the u-pass),
        // we can only subdivide the edges that already exist at the current level.
        // so if we're going from a 4x4 patch to an 8x8 patch, both coords can only be
        // 0, 1, 2, 3, or 4. but, we're writing data to a buffer for the NEXT level,
        // which has twice the range of coords. so we're just mapping, say,
        // {0, 1, 2, 3, 4} -> {0, 2, 4, 6, 8}
        let nqpd = num_quads_per_dim;
        let pos_grid_x : u32 = (grid_x / p.u_deg) * 2;
        //let pos_grid_y : u32 = (grid_y / p.v_deg) * 2;
        let pos_grid_y : u32 = (grid_y) * 2;

        let pos_buffer_width : u32 = (num_quads_per_dim << 1) + 1;
        let pos_zero_index : u32 = p.index_offset +
            inside_patch_offset[cur_pass.curr_computed_level + 1];

        let pos_base_index : u32 = pos_zero_index + (pos_grid_x * pos_buffer_width) + pos_grid_y;
        position_buffer[pos_base_index]      = bb_buffer[base_index];
        position_buffer[pos_base_index + 1]  = bb_buffer[base_index + (halfstride * p.v_deg)];
        position_buffer[pos_base_index + 2]  = bb_buffer[base_index + (stride * p.v_deg)];
    }
}
`;

export const uPassWGSL = /* wgsl */ `
struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}

struct passInfo {
    nr_invocations : u32,
    curr_computed_level : u32,
}

@group(0) @binding(0) var<storage, read_write> position_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bb_buffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p : patchInfo;
@group(0) @binding(3) var<uniform> cur_pass : passInfo;

// "highest (subdivision) level"
const hl : u32 = 6;
// (2 ^ hl), so for hl=6 this is 64
const max_quads = 1 << hl;

const inside_patch_offset = array<u32, 7>(0, 4, 13, 38, 119, 408, 1497);

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
    let base_index : u32 = (x * bw) + y;

    for (var i : u32 = 0; i < p.u_deg; i++) {
        for (var j : u32 = i; j < p.u_deg; j++) {
            let x_temp : u32 = ((halfstride * i) + (stride * (j-i)));

            // let p1  = p.index_offset + y + (bw * (x_temp));
            // let mid = p.index_offset + y + (bw * (x_temp + halfstride));
            // let p2  = p.index_offset + y + (bw * (x_temp + stride));
            let p1  = base_index + (bw * (x_temp));
            let mid = base_index + (bw * (x_temp + halfstride));
            let p2  = base_index + (bw * (x_temp + stride));

            bb_buffer[mid] = (bb_buffer[p1] + bb_buffer[p2]) * 0.5;
        }
    }

    if (grid_y % p.v_deg == 0) {
        let nqpd = num_quads_per_dim;
        let pos_grid_x : u32 = (grid_x) * 2;
        let pos_grid_y : u32 = (grid_y / p.v_deg);

        let pos_buffer_width : u32 = (num_quads_per_dim << 1) + 1;
        let pos_zero_index : u32 = p.index_offset +
            inside_patch_offset[cur_pass.curr_computed_level + 1];

        let pos_base_index : u32 = pos_zero_index + (pos_grid_x * pos_buffer_width) + pos_grid_y;
        let bb_index = base_index + (halfstride * p.u_deg * bw);

        position_buffer[pos_base_index + (1 * pos_buffer_width)]  = bb_buffer[bb_index];
    }
}
`;

export const normalWGSL = /* wgsl */ `
struct patchInfo {
    index_offset: u32,
    u_deg : u32,
    v_deg : u32,
}

struct passInfo {
    nr_invocations : u32,
    curr_computed_level : u32,
}

@group(0) @binding(0) var<storage, read_write> bb_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normal_buffer  : array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p : patchInfo;
@group(0) @binding(3) var<uniform> cur_pass : passInfo;

// "highest (subdivision) level"
const hl : u32 = 6;
// (2 ^ hl), so for hl=6 this is 64
const max_quads = 1 << hl;

const inside_patch_offset = array<u32, 7>(0, 4, 13, 38, 119, 408, 1497);

@compute @workgroup_size(64, 1, 1) fn normal(
    @builtin( global_invocation_id ) globalId : vec3<u32>,
) {
    let instance_index : u32 = globalId.x;

    if (instance_index > cur_pass.nr_invocations - 1) {
        return;
    }


    // ((2 ^ level) + 1)
    // so at level 2, this is (2^2)+1 = 5 vertices per dimension
    let num_verts_per_dim : u32 = (u32(1) << cur_pass.curr_computed_level) + 1;

    let stride : u32 = u32(1) << (hl - cur_pass.curr_computed_level);

    let grid_x : u32 = instance_index / num_verts_per_dim ;
    let grid_y : u32 = instance_index % num_verts_per_dim ;

    // "buffer width", how many indices you need to move to increase the x coordinate by 1
    let bw : u32 = (p.v_deg * max_quads) + 1;

    let x = grid_x * stride * p.u_deg * bw;
    let y = grid_y * stride * p.v_deg;
    let base_index = x + y;


    var v00 : vec3<f32>;
    var v01 : vec3<f32>;
    var v10 : vec3<f32>;
    var v11 : vec3<f32>;
    
    if (grid_x == num_verts_per_dim - 1 && grid_y == num_verts_per_dim - 1) {
        // "top right" corner

        // indices for the points we will use for normals
        let v00_index = base_index;
        let v01_index = base_index - (1 * stride);
        let v10_index = base_index - (1 * stride * bw);
        let v11_index = base_index - (1 * stride * bw) - (1 * stride);

        // normalize rational point
        v00 = bb_buffer[v00_index].xyz * (1 / bb_buffer[v00_index].w);
        v01 = bb_buffer[v01_index].xyz * (1 / bb_buffer[v01_index].w);
        v10 = bb_buffer[v10_index].xyz * (1 / bb_buffer[v10_index].w);
        v11 = bb_buffer[v11_index].xyz * (1 / bb_buffer[v11_index].w);

    } else if (grid_x == num_verts_per_dim - 1) {
        // "right" edge

        // indices for the points we will use for normals
        let v00_index = base_index;
        let v01_index = base_index - (1 * stride * bw);
        let v10_index = base_index + (1 * stride);
        let v11_index = base_index - (1 * stride * bw) + (1 * stride);

        // normalize rational point
        v00 = bb_buffer[v00_index].xyz * (1 / bb_buffer[v00_index].w);
        v01 = bb_buffer[v01_index].xyz * (1 / bb_buffer[v01_index].w);
        v10 = (bb_buffer[v10_index].xyz * (1 / bb_buffer[v10_index].w));
        v11 = (bb_buffer[v11_index].xyz * (1 / bb_buffer[v11_index].w));


    } else if (grid_y == num_verts_per_dim - 1) {
        // "top" edge

        // indices for the points we will use for normals
        let v00_index = base_index;
        let v01_index = base_index + (1 * stride * bw);
        let v10_index = base_index - (1 * stride);
        let v11_index = base_index + (1 * stride * bw) - (1 * stride);

        // normalize rational point
        v00 = bb_buffer[v00_index].xyz * (1 / bb_buffer[v00_index].w);
        v01 = bb_buffer[v01_index].xyz * (1 / bb_buffer[v01_index].w);
        v10 = bb_buffer[v10_index].xyz * (1 / bb_buffer[v10_index].w);
        v11 = bb_buffer[v11_index].xyz * (1 / bb_buffer[v11_index].w);

    } else {
        // "bottom left" corner; what gets used for most of the calculations

        // indices for the points we will use for normals
        let v00_index = base_index;
        let v01_index = base_index + (1 * stride); // +1 y
        let v10_index = base_index + (1 * stride * bw); // +1 x
        let v11_index = base_index + (1 * stride * bw) + (1 * stride); // +1 x, +1 y

        // normalize rational point
        v00 = bb_buffer[v00_index].xyz * (1 / bb_buffer[v00_index].w);
        v01 = bb_buffer[v01_index].xyz * (1 / bb_buffer[v01_index].w);
        v10 = bb_buffer[v10_index].xyz * (1 / bb_buffer[v10_index].w);
        v11 = bb_buffer[v11_index].xyz * (1 / bb_buffer[v11_index].w);

    }

    var r_v : vec3<f32> = v01 - v00;
    if (length(r_v) < 1e-6) {
        r_v = v11 - v00;
    }

    var r_u : vec3<f32> = v10 - v00;
    if (length(r_u) < 1e-6) {
        r_u = v11 - v00;
    }

    let normal = normalize(cross(r_u, r_v));
    //normal_buffer[base_index] = normal;

    let pos_zero_index : u32 = p.index_offset +
        inside_patch_offset[cur_pass.curr_computed_level];

    normal_buffer[pos_zero_index + instance_index] = vec4<f32>(normal, 1);
}
`

export class BvQuad {

    static indexBuffers = {};

    /**
     * @param {int} patchType
     * @param {int} uDeg
     * @param {int} vDeg
     * @param {THREE.Float32BufferAttribute} cpBuffer
     * @param {GPUBuffer} bbBuffer
     */
    constructor(patchType, uDeg, vDeg, cpBuffer, bbBuffer) {
        this.patchType = patchType;
        this.uDeg = uDeg;
        this.vDeg = vDeg;

        // TODO: annoying discrepancy between cpBuffer being THREE and everything else
        // being type GPUBuffer
        this.cpBuffer = cpBuffer;
        this.bbBuffer = bbBuffer;


        // set when this patch is assigned to a particular chunk in a BvGroup
        this.indexOffset = -1;

        // set when initPatch() is called.
        this.positionBuffer  = -1;
        this.normalBuffer    = -1;
        this.curvatureBuffer = -1;
        this.uniformBuffer   = -1; // GPUBuffer that holds indexOffset/uDeg/vDeg
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
    // TODO: write normals, curvature into buffers for level 0
    initPatch(rsrc, pass, positionBuffer, normalBuffer, curvatureBuffer) {
        this.positionBuffer  = positionBuffer;
        this.normalBuffer    = normalBuffer;
        this.curvatureBuffer = curvatureBuffer;

        // ====================
        // Copy control points to this patch's bbBuffer, and to the level 0 position buffer
        // ====================

        {
            // ==========
            // Create and cache a uniform buffer that stores indexOffset/uDeg/vDeg
            // ==========

            // index_offset(u32), u_deg(u32), v_deg(u32)
            const uniformBufferSize = (4) + (4) + (4);
            const uniformBuffer = rsrc.device.createBuffer({
                label: `Patch info uniform (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
                size: uniformBufferSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const uniformValues = new Uint32Array([this.indexOffset, this.uDeg, this.vDeg]);
            rsrc.device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

            this.uniformBuffer = uniformBuffer;
        }


        // The GPUBuffer for this patch's control points
        const patchCPBuffer = rsrc.backend.get(this.cpBuffer).buffer;

        const bindGroup = rsrc.device.createBindGroup({
            label: `Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
            layout: rsrc.quad.init.bindGroupLayout,
            entries: [
                {binding: 0, resource: positionBuffer },
                {binding: 1, resource: patchCPBuffer },
                {binding: 2, resource: this.bbBuffer },
                {binding: 3, resource: this.uniformBuffer },
            ]
        });

        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
    }



    // lookup table for where in a patch a specific subdiv level's positions are stored.
    // prefix / running sum of nr. of points needed for each buffer size
    static insidePatchOffset = [0, 4, 13, 38, 119, 408, 1497];

    getIndexBuffer(subdivLevel, arrayToWriteTo, firstIdxToWriteTo) {

        const key = `${subdivLevel}`;
        if (!(key in BvQuad.indexBuffers)) {
            // console.log(`Quad index buffer cache miss: level ${key}`)

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

            const quadsPerDim = 1 << subdivLevel; // 2^n
            // how many indices you have to jump to move over 1 in the u direction
            const bufferWidth = quadsPerDim + 1;
            const indices = new Uint32Array(quadsPerDim * quadsPerDim * 6);

            let idx = 0;
            for (let i = 0; i < quadsPerDim; i++) {
                for (let j = 0; j < quadsPerDim; j++) {

                    const bottomLeft    = (bufferWidth * (i))     + j;
                    const bottomRight   = (bufferWidth * (i))     + j + 1;
                    const topLeft       = (bufferWidth * (i + 1)) + j;
                    const topRight      = (bufferWidth * (i + 1)) + j + 1;

                    // Triangle 1
                    indices[idx++] = topLeft;
                    indices[idx++] = bottomRight;
                    indices[idx++] = bottomLeft;
                    // Triangle 2
                    indices[idx++] = topLeft;
                    indices[idx++] = topRight;
                    indices[idx++] = bottomRight;
                }
            }
            BvQuad.indexBuffers[key] = indices;
        }
        

        // only add offset after caching the generated indices
        for (let i = 0; i < BvQuad.indexBuffers[key].length; i++) {
            const val = BvQuad.indexBuffers[key][i] +
                this.indexOffset +
                BvQuad.insidePatchOffset[subdivLevel];

            // console.log(`writing < ${val} > to index < ${firstIdxToWriteTo + i} >`)
            arrayToWriteTo[firstIdxToWriteTo + i] = val;
        }

        // console.log("----------------------------------------")
        // console.log(arrayToWriteTo.slice(
        //     firstIdxToWriteTo, 
        //     firstIdxToWriteTo + BvQuad.indexBuffers[key].length)
        // );
        // console.log(arrayToWriteTo);
        // console.log("----------------------------------------")
    }


    subdivPatch({rsrc, pass, highestComputedLevel, newLevel}) {
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
            const vPassInfoBuffer = rsrc.device.createBuffer({
                size: 8, // two u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            rsrc.device.queue.writeBuffer(vPassInfoBuffer, 0, new Uint32Array([vShaderInvocations, currLevel]));

            const vBindGroup = rsrc.device.createBindGroup({
                label: `V Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
                layout: rsrc.quad.subdiv.vLayout,
                entries: [
                    {binding: 0, resource: this.positionBuffer },
                    {binding: 1, resource: this.bbBuffer },
                    {binding: 2, resource: this.uniformBuffer },
                    {binding: 3, resource: vPassInfoBuffer },
                ]
            });
            pass.setBindGroup(0, vBindGroup);
            pass.dispatchWorkgroups(Math.ceil(vShaderInvocations / 64)); // (workgroup size is 64)
            // console.log(`v: currLevel=${currLevel} invocations=${vShaderInvocations}`)



            // ====================
            // U Pass
            // ====================
            pass.setPipeline(rsrc.quad.subdiv.uPipeline);

            // uniform to send the # of invocations for this shader, along with the current
            // highest computed level
            const uShaderInvocations = ((nqpd*this.vDeg*2)+1)*nqpd;
            const uPassInfoBuffer = rsrc.device.createBuffer({
                size: 8, // two u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            rsrc.device.queue.writeBuffer(uPassInfoBuffer, 0, new Uint32Array([uShaderInvocations, currLevel]));

            const uBindGroup = rsrc.device.createBindGroup({
                label: `U Bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
                layout: rsrc.quad.subdiv.uLayout,
                entries: [
                    {binding: 0, resource: this.positionBuffer },
                    {binding: 1, resource: this.bbBuffer},
                    {binding: 2, resource: this.uniformBuffer },
                    {binding: 3, resource: uPassInfoBuffer },
                ]
            });
            pass.setBindGroup(0, uBindGroup);
            pass.dispatchWorkgroups(Math.ceil(uShaderInvocations / 64)); // (workgroup size is 64)
            // console.log(`u: currLevel=${currLevel} invocations=${uShaderInvocations}`)


            // ====================
            // Normals
            // ====================
            pass.setPipeline(rsrc.quad.normal.pipeline);
            this.calcNormals({rsrc, pass, level: currLevel+1});
        }
    }

    // this assumes the correct pipeline has already been set
    calcNormals({rsrc, pass, level}) {
        // runs once per vertex at the current level.
        // e.g. level 3 = 2^3 x 2^3 = 8x8 quads.
        // that means (8+1)*(8+1) = 81 vertices
        const nrOfInvocations = ((1 << level) + 1) * ((1 << level) + 1);
        const invocationsBuffer = rsrc.device.createBuffer({
            size: 8, // two u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        rsrc.device.queue.writeBuffer(
            invocationsBuffer,
            0,
            new Uint32Array([nrOfInvocations, level])
        );

        const bindGroup = rsrc.device.createBindGroup({
            label: `Normals bind group for (${this.indexOffset}, ${this.uDeg}, ${this.vDeg})`,
            layout: rsrc.quad.normal.bindGroupLayout,
            entries: [
                {binding: 0, resource: this.bbBuffer },
                {binding: 1, resource: this.normalBuffer },
                {binding: 2, resource: this.uniformBuffer },
                {binding: 3, resource: invocationsBuffer  },
            ]
        });
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(nrOfInvocations / 64));
    }
}

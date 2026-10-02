import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float} from 'three/tsl';


// TODO: when profiling either compute shader, the thing actually taking the most time
// (and by no small amount) is just memory accesses, since we keep reading/writing directly
// to the position buffer. seems to be roughly a 3 : 1 ratio of memory : calculations,
// address that!
// i believe there should be a way to e.g. 
export class BvQuad {

    static indexBuffers = {};

    /**
     * @param {int} patchType
     * @param {int} uDeg
     * @param {int} vDeg
     * @param {GPUBuffer} cpBuffer
     * @param {GPUBuffer} uniformBuffer
     */
    constructor(patchType, uDeg, vDeg, cpBuffer, uniformBuffer) {
        this.patchType = patchType;
        this.uDeg = uDeg;
        this.vDeg = vDeg;
        this.cpBuffer = cpBuffer;
        this.patchUniformBuffer = uniformBuffer;

        // buffersByLevel will look like this in use:
        // buffersByLevel['1'] = {
        //     // makes it easy to get p/n/c buffers
        //     parentChunk: <chunk>,
        //     indexOffset: <int>,
        // }
        this.buffersByLevel = {};

    }

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
            const val = BvQuad.indexBuffers[key][i] + this.buffersByLevel[subdivLevel].indexOffset;
            arrayToWriteTo[firstIdxToWriteTo + i] = val;
        }
    }


    vPass({rsrc, pass, currentLevel, targetLevel}) {
        const nqpd = 1 << currentLevel;

        const vShaderInvocations = ((nqpd*this.uDeg)+1)*nqpd;
        const vPassInfo = new Uint32Array([
            vShaderInvocations,
            currentLevel,
            targetLevel
        ]);
        const vPassInfoBuffer = rsrc.device.createBuffer({
            size: vPassInfo.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        rsrc.device.queue.writeBuffer(
            vPassInfoBuffer,
            0,
            vPassInfo
        );

        const vBindGroup = rsrc.device.createBindGroup({
            label: `V Bind group for (u=${this.uDeg}, v=${this.vDeg}, cL=${currentLevel}, tL=${targetLevel})`,
            layout: rsrc.quad.subdiv.vLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: rsrc.honkinBuffer,
                        size:   this.byteSize,
                        offset: this.byteOffset,
                    }
                },
                {
                    binding: 1,
                    resource: this.cpBuffer,
                },
                {
                    binding: 2,
                    resource: this.patchUniformBuffer 
                },
                {
                    binding: 3,
                    resource: vPassInfoBuffer
                },
            ]
        });
        pass.setBindGroup(0, vBindGroup);
        pass.dispatchWorkgroups(Math.ceil(vShaderInvocations / 64)); // (workgroup size is 64)

    }

    uPass({rsrc, pass, currentLevel, targetLevel}) {
        const nqpd = 1 << currentLevel;

        const uShaderInvocations = ((nqpd*this.vDeg*2)+1)*nqpd;
        const uPassInfo = new Uint32Array([
            uShaderInvocations,
            currentLevel,
            targetLevel
        ]);
        const uPassInfoBuffer = rsrc.device.createBuffer({
            size: uPassInfo.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        rsrc.device.queue.writeBuffer(
            uPassInfoBuffer,
            0,
            uPassInfo
        );

        const uBindGroup = rsrc.device.createBindGroup({
            label: `U Bind group for (u=${this.uDeg}, v=${this.vDeg}, cL=${currentLevel}, tL=${targetLevel})`,
            layout: rsrc.quad.subdiv.uLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: rsrc.honkinBuffer,
                        size:   this.byteSize,
                        offset: this.byteOffset,
                    }
                },
                {
                    binding: 1,
                    resource: this.patchUniformBuffer
                },
                {
                    binding: 2,
                    resource: uPassInfoBuffer
                },
            ]
        });
        pass.setBindGroup(0, uBindGroup);
        pass.dispatchWorkgroups(Math.ceil(uShaderInvocations / 64)); // (workgroup size is 64)
    }

    // this will (in the shader):
    // 1. copy the position from the scratch buffer to, well, the position buffer
    // 2. calculate the normal, write to normal buffer
    // 3. calculate the curvature, write to curvature buffer
    evaluate({rsrc, pass, currentLevel, targetLevel}) {

        ////////////////////
        // Create a uniform buffer that holds:
        //
        // - number of desired invocations for this shader
        //      - for early returns when more instances than needed run
        // - current subdivision level
        //      - i.e., the one we want to calculate normals/curvature for
        // - target subdivision level
        //      - stride between points in the buffer depends on what the target
        //        subdivision level is
        // - index offset in the position/normal/curvature buffers
        ////////////////////

        // runs once per vertex at the current level.
        // e.g. level 3 = 2^3 x 2^3 = 8x8 quads
        // that means (8+1)*(8+1) = 81 vertices
        const nrOfInvocations = ((1 << currentLevel) + 1) * ((1 << currentLevel) + 1);

        const passInfo = new Uint32Array([
                nrOfInvocations,
                currentLevel,
                targetLevel,
                this.buffersByLevel[currentLevel].indexOffset
        ]);
        const passInfoBuffer = rsrc.device.createBuffer({
            size: passInfo.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        rsrc.device.queue.writeBuffer(
            passInfoBuffer,
            0,
            passInfo
        );

        const bindGroup = rsrc.device.createBindGroup({
            label: `'Evaluate' bind group for (${this.uDeg}, ${this.vDeg})`,
            layout: rsrc.quad.evaluate.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: rsrc.honkinBuffer,
                        size:   this.byteSize,
                        offset: this.byteOffset,
                    }
                },
                {
                    binding: 1,
                    resource: this.buffersByLevel[currentLevel].parentChunk.positionBuffer,
                },
                {
                    binding: 2,
                    resource: this.buffersByLevel[currentLevel].parentChunk.normalBuffer,
                },
                {
                    binding: 3,
                    resource: this.buffersByLevel[currentLevel].parentChunk.curvatureBuffer,
                },
                {
                    binding: 4,
                    resource: this.patchUniformBuffer,
                },
                {
                    binding: 5,
                    resource: passInfoBuffer,
                },
                {
                    binding: 6,
                    resource: this.cpBuffer,
                }
            ],

        });
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(nrOfInvocations / 64));

        const DEBUG_MODE = false;
        if (DEBUG_MODE === true) {
            console.group("My uniform values:");
            console.log("Pass info:")
            console.log(passInfo);
            console.log("Patch info:")
            console.log(this.patchUniformBuffer);
            console.groupEnd();
        }
    }
}

import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float} from 'three/tsl';


export class BvQuad {

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

    // TODO: should we just do this on the GPU?
    getIndexBuffer(subdivLevel, arrayToWriteTo, firstIdxToWriteTo) {

        // data stored in u-major order
        //
        //    2   5
        //     x---x
        //     |   |
        //    1|  4|
        //     x---x
        //  ^  |   |
        //  | 0|  3|
        //  v  x---x
        //   u ->
        //

        const quadsPerDim = 1 << subdivLevel; // 2^n

        // how many indices you have to jump to move over 1 in the u direction
        const bufferWidth = quadsPerDim + 1;

        let idx = firstIdxToWriteTo;
        let offset = this.buffersByLevel[subdivLevel].indexOffset;
        for (let i = 0; i < quadsPerDim; i++) {
            for (let j = 0; j < quadsPerDim; j++) {

                const base = offset + (bufferWidth * i);
                const bottomLeft    = base + j;
                const bottomRight   = base + j + 1;
                const topLeft       = base + bufferWidth + j;
                const topRight      = base + bufferWidth + j + 1;
                // const bottomLeft    = offset + (bufferWidth * (i))     + j;
                // const bottomRight   = offset + (bufferWidth * (i))     + j + 1;
                // const topLeft       = offset + (bufferWidth * (i + 1)) + j;
                // const topRight      = offset + (bufferWidth * (i + 1)) + j + 1;

                // Triangle 1
                arrayToWriteTo[idx++] = topLeft;
                arrayToWriteTo[idx++] = bottomRight;
                arrayToWriteTo[idx++] = bottomLeft;
                // Triangle 2
                arrayToWriteTo[idx++] = topLeft;
                arrayToWriteTo[idx++] = topRight;
                arrayToWriteTo[idx++] = bottomRight;
            }
        }
    }

    // this will (in the shader):
    // 1. copy the position from the scratch buffer to, well, the position buffer
    // 2. calculate the normal, write to normal buffer
    // 3. calculate the curvature, write to curvature buffer
    evaluate({rsrc, scratchBuffer, pass, currentLevel, targetLevel}) {

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

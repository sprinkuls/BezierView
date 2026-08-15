import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float} from 'three/tsl';

const initPatch = Fn(({cpBuffer, positionStorage, uDeg, vDeg, indexOffset}) => {
    const cpStorage = storage(cpBuffer, 'vec4', cpBuffer.count);

    // equivalent to:
    // for (int i = 0; i <= uDeg; i++)
    //     for (int j = 0; j <= vDeg; j++)
    Loop( add(uDeg, 1), add(vDeg, 1), ( { i, j } ) => {
        const src = add(mul(i, add(vDeg, 1)), j);
        const dst = add(mul(mul(i, 64), add(mul(vDeg, 64), 1)), mul(j, 64));

        positionStorage.element(add(dst, indexOffset)).assign(cpStorage.element(src));
    });
});

const vPassCompute = Fn ( ({positionStorage, indexOffset, uDeg, vDeg, currComputedLevel}) => {
    const maxLevel = int(6);
    const numQuadPerDim = int(shiftLeft(1, currComputedLevel));

    // ranges from 1 to ((numQuadPerDim * uDeg)+1) (so really, 0 to (numQuadPerDim * uDeg))
    const gridX = int(div(instanceIndex, numQuadPerDim));
    // ranges from 1 to numQuadPerDim (so really, 0 to numQuadPerDim-1)
    const gridY = mod(int(instanceIndex), numQuadPerDim);


    const stride = shiftLeft(1, sub(maxLevel, int(currComputedLevel)));
    const halfstride = shiftRight(stride, int(1));

    // how many indices you have to move over before moving "right" 1 in the U direction
    const bufferSize = add(mul(int(vDeg), shiftLeft(1, maxLevel)), 1);
    const flattenInstanceIndex = (i, j) => (add(mul(i, bufferSize), j));

    const x = int(mul(gridX, stride));
    const y = int(mul(gridY, mul(stride, vDeg)));


    Loop( { start: 0, end: int(vDeg), type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

        Loop( { start: i, end: int(vDeg), type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

            const ytemp = add(y, add(mul(halfstride, i), mul(sub(j, i), stride)));

            const p1  = add(indexOffset, flattenInstanceIndex(x, ytemp));
            const mid = add(indexOffset, flattenInstanceIndex(x, add(ytemp, halfstride)));
            const p2  = add(indexOffset, flattenInstanceIndex(x, add(ytemp, stride)));

            positionStorage.element(mid).assign(mul(add(positionStorage.element(p1), positionStorage.element(p2)), 0.5));

        } );

    } );

});

const uPassCompute = Fn ( ({positionStorage, indexOffset, uDeg, vDeg, currComputedLevel}) => {
    const maxLevel = int(6);
    const numQuadPerDim = int(shiftLeft(int(1), currComputedLevel));

    // ranges from 0 to (numQuadPerDim-1)
    const gridX = mod(int(instanceIndex), numQuadPerDim);
    // ranges from 0 to (numQuadPerDim * vDeg * 2)
    const gridY = int(div(instanceIndex, numQuadPerDim));

    const stride = shiftLeft(1, sub(maxLevel, currComputedLevel));
    const halfstride = shiftRight(stride, int(1));

    const bufferSize = add(mul(int(vDeg), shiftLeft(1, maxLevel)), 1);
    const flattenInstanceIndex = (i, j) => (add(mul(i, bufferSize), j));

    const x = mul(gridX, mul(stride, uDeg));
    const y = mul(gridY, halfstride);

    Loop( { start: 0, end: int(uDeg), type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

        Loop( { start: i, end: int(uDeg), type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

            const xtemp = add(x, add(mul(halfstride, i), mul(stride, sub(j, i))));

            const p1  = add(indexOffset, flattenInstanceIndex(xtemp, y));
            const mid = add(indexOffset, flattenInstanceIndex(add(xtemp, halfstride), y));
            const p2  = add(indexOffset, flattenInstanceIndex(add(xtemp, stride), y));

            positionStorage.element(mid).assign(mul(add(positionStorage.element(p1), positionStorage.element(p2)), 0.5));

        } );

    } );
});

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
    }

    getMemNeededBytes() {
        // (number of vertices) * (4 bytes per float32) * (4 float32s per vertex)
        // (even if we request a vec3 we're always actually getting a vec4 under the hood)
        return (((this.uDeg * 64) + 1) * ((this.vDeg * 64) + 1)) * 4 * 4;
    }

    getInitComputeNode(positionStorage) {
        return initPatch({
            cpBuffer: this.cpBuffer,
            positionStorage: positionStorage,
            uDeg: this.uDeg,
            vDeg: this.vDeg,
            indexOffset: this.indexOffset
        }).compute(1);
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

        console.log(`Index buffer cache miss: ${key}`)

        const quadsPerDim = 1 << level; // 2^n
        const stride = 1 << (6 - level);
        const uStride = this.uDeg * stride; // spacing in the u direction
        const vStride = this.vDeg * stride; // spacing in the v direction

        /*
         * data stored in "v-major" order
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

    // returns compute nodes that, when computed, will subdivide this patch
    getSubdivComputeNodes(positionStorage, highestComputedLevel, newLevel) {
        if (highestComputedLevel >= newLevel) {
            throw new Error();
        }

        const nodes = [];
        for (let currLevel = highestComputedLevel; currLevel < newLevel; currLevel++) {
            const nqpd = 1 << currLevel; // "number of quads per dimension"
            const v = vPassCompute({
                positionStorage: positionStorage,
                indexOffset: uniform(this.indexOffset),
                uDeg: uniform(this.uDeg),
                vDeg: uniform(this.vDeg),
                currComputedLevel: uniform(currLevel)
            }).compute(((nqpd*this.uDeg)+1)*nqpd);
            const u = uPassCompute({
                positionStorage: positionStorage,
                indexOffset: uniform(this.indexOffset),
                uDeg: uniform(this.uDeg),
                vDeg: uniform(this.vDeg),
                currComputedLevel: uniform(currLevel)
            }).compute(((nqpd*this.vDeg*2)+1)*nqpd);

            nodes.push(v);
            nodes.push(u);
        }

        return nodes;
    }
}

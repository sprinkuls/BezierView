import * as THREE from 'three/webgpu';
import { vec3, storage, Fn, If, Loop, equal, notEqual, uniform, instanceIndex, objectWorldMatrix, color, screenUV, attribute, mul, add, sub, div, mod, shiftLeft, shiftRight, floor, abs, uint, int, float, sqrt, assign} from 'three/tsl';

// const initPatch = Fn(({cpBuffer, positionStorage, deg, indexOffset}) => {
const initPatch = Fn(({cpStorage, positionStorage, deg, indexOffset}) => {
    // const cpStorage = storage(cpBuffer, 'vec4', cpBuffer.count);

    // always only three points to assign, and positionStorage is always the same size
    positionStorage.element(add(0, indexOffset)).assign(cpStorage.element(0));

    // 'degree-th' triangle nr. index (e.g. deg 1 -> index 1, deg 3 -> index 6)
    const cpIdx = shiftRight(mul(deg, add(deg, 1)), 1);

    // 2080 is the beginning of the last row, 2144 is the end of the last row
    positionStorage.element(add(2080, indexOffset)).assign(cpStorage.element(cpIdx));
    positionStorage.element(add(2144, indexOffset)).assign(cpStorage.element(add(cpIdx, deg)));
});

const subdivCompute = Fn ( ({positionStorage, cpStorage, scratchStorage, indexOffset, deg, currComputedLevel}) => {

    // e.g. level 0 has 1 tri per edge, level 1 has 2
    const trisPerEdge = int(shiftLeft(int(1), currComputedLevel));

    // for every upright triangle at the current level of subdiv., 3 shader instances get dispatched
    const triIndex = int(div(instanceIndex, 3));

    // we index into tris like this with the triIndex. basically, start from the
    // top corner of each tri we currently have at this level of subdiv.
    //
    // example: current computed level = 1
    //
    // triIndex     ->   (row, col)
    //
    // 0                 (0,0)
    //   |\                    |\
    //   | \                   | \
    //   |  \                  |  \
    // 1 |___\ 2    ->   (1,0) |___\ (1,1)
    //   |\  |\                |\  |\
    //   | \ | \               | \ | \
    //   |  \|  \              |  \|  \
    //   |___|___\             |___|___\
    const triNr = (n) => (div(mul(n, add(n, 1)), 2));

    const row = floor(div(sub(sqrt(float(add(mul(triIndex, 8), 1))), 1), 2));
    const col = sub(triIndex, triNr(row));

    // go from finding our place at the current level of subdivision, to where we want to be
    // for the next subdiv. level. we multiply by two because there will be twice as many rows/cols
    // at the next level.
    row.assign(mul(row, 2));
    col.assign(mul(col, 2));
    trisPerEdge.assign(mul(trisPerEdge, 2));

    // 3 shader instances were dispatched per tri, and for each, we want to move
    // to a midpoint on one side of the triangle
    const midpoint_nr = mod(instanceIndex, 3);

    If (equal(midpoint_nr, 0), () => {
        row.assign(add(row, 1));
    }).ElseIf(equal(midpoint_nr, 1), () => {
        row.assign(add(row, 1));
        col.assign(add(col, 1));
    }).ElseIf(equal(midpoint_nr, 2), () => {
        row.assign(add(row, 2));
        col.assign(add(col, 1));
    });

    // find u,v,w weights for the point we want to calculate on the overall triangle.
    // i = trisPerEdge - row
    // j = row - col
    // k = col
    // then divide each of these by trisPerEdge to get normalized 0 to 1 values
    const u = div(float(sub(trisPerEdge, row)), float(trisPerEdge));
    const v = div(float(sub(row, col)), float(trisPerEdge));
    const w = div(float(col), float(trisPerEdge));

    // now, decasteljau's:


    const rcToIndex = (r, c) => (add(triNr(r), c));
    const scratchOffset = mul(instanceIndex, triNr(deg));

    // for the first step, read from the control points and write to scratch
    Loop( { start: 0, end: int(deg), type: 'int', name: 'row', condition: '<' }, ( { row } ) => {
        Loop( { start: 0, end: int(add(row, 1)), type: 'int', name: 'col', condition: '<' }, ( { col } ) => {
            // u * (u point) +
            // v * (v point) +
            // w * (w point)
            const cp_u = cpStorage.element(rcToIndex(row, col));
            const cp_v = cpStorage.element(rcToIndex(add(row, 1), col));
            const cp_w = cpStorage.element(rcToIndex(add(row, 1), add(col, 1)));

            const u_u = mul(u, cp_u);
            const v_v = mul(v, cp_v);
            const w_w = mul(w, cp_w);
            const scratchIdx = add(scratchOffset, rcToIndex(row, col));
            scratchStorage.element(scratchIdx).assign(add(u_u, v_v, w_w));
        } );

    } );

    // for the rest, both read/write from scratch
    // (operations are ordered such that these don't conflict)
    Loop( { start: sub(deg, 1), end: int(0), type: 'int', name: 'loop_deg', condition: '>' }, ( { loop_deg } ) => {

        Loop( { start: 0, end: int(loop_deg), type: 'int', name: 'row', condition: '<' }, ( { row } ) => {
            Loop( { start: 0, end: int(add(row, 1)), type: 'int', name: 'col', condition: '<' }, ( { col } ) => {
                const cp_u = scratchStorage.element(add(scratchOffset, rcToIndex(row, col)));
                const cp_v = scratchStorage.element(add(scratchOffset, rcToIndex(add(row, 1), col)));
                const cp_w = scratchStorage.element(add(scratchOffset, rcToIndex(add(row, 1), add(col, 1))));
                const u_u = mul(u, cp_u);
                const v_v = mul(v, cp_v);
                const w_w = mul(w, cp_w);
                const scratchIdx = add(scratchOffset, rcToIndex(row, col));
                scratchStorage.element(scratchIdx).assign(add(u_u, v_v, w_w));
            } );
        } );

    } );


    //write scratch[0] (or, well, scratch[scratchOffset]) to the corresponding place in memory

    // idea is just that 6 is the highest subdiv. level, so to get results half that size,
    // we subtract 1 (thus halving the value, as this is raising 2 to an exponent)
    // 1 << (5 - currComputedLevel)
    const stride = shiftLeft(int(1), sub(int(5), int(currComputedLevel)));
    const trueRow = mul(row, stride);
    const trueCol = mul(col, stride);

    // idx = indexOffset + (triNr(trueRow) + trueCol);
    const idx = add(indexOffset, add(triNr(trueRow), trueCol));
    positionStorage.element(idx).assign(scratchStorage.element(scratchOffset))
});

export class BvTri {

    static indexBuffers = {};

    /**
     * @param {int} patchType
     * @param {int} deg
     * @param {THREE.Float32BufferAttribute} cpBuffer
     */
    constructor(patchType, deg, cpBuffer) {
        this.patchType = patchType;
        this.deg = deg;
        this.cpBuffer = cpBuffer;
        this.cpStorage = storage(cpBuffer, 'vec4', cpBuffer.count);

        // set later when the group this patch is a part of, well, sets them
        this.byteOffset = -1;
        this.indexOffset = -1;
    }

    getMemNeededBytes() {
        // BvTris' vertices are stored as triangular arrays, not as 2D arrays that
        // only have data along the lower triangular part of the array

        // the triangle nr. for 65 is ((65)*(66)/2) = 2145 vertices, and
        // 2145 vertices * 4 float32/vertex * 4 bytes/float32 = 34320 bytes
        return 34320;
    }

    getInitComputeNode(positionStorage) {
        return initPatch({
            cpStorage: this.cpStorage,
            positionStorage: positionStorage,
            deg: this.deg,
            indexOffset: this.indexOffset
        }).compute(1);
    }

    getIndexBuffer(level) {
        const key = `${level}-${this.deg}`;
        if (key in BvTri.indexBuffers) {
            const indexBufferCopy = new Uint32Array(BvTri.indexBuffers[key]);
            for (let i = 0; i < indexBufferCopy.length; i++) {
                indexBufferCopy[i] += this.indexOffset;
            }
            return indexBufferCopy;
        }

        console.log(`Index buffer cache miss: ${key}`)
        const trisPerEdge = 1 << level;
        const stride = 1 << (6 - level);

        // the first index of every row is just the 'row-th' triangle number:
        // 0
        // 1 2
        // 3 4 5
        // 6 7 8 9
        const rcIndex = (r, c) => ((((r*stride)*((r*stride)+1)) >> 1) + (c*stride));

        // total nr of elements used for this constructor
        const indices = new Uint32Array(trisPerEdge * trisPerEdge * 3);

        let idx = 0;
        for (let r = 0; r < trisPerEdge; r++) {
            for (let c = 0; c < (r + 1); c++) {
                // console.log(`(${r}, ${c}) (aka (${r*stride}, ${c*stride}))`)
                // console.log(`${(((r*stride)*((r*stride)+1)) >> 1)} + ${c*stride}`)
                const topLeft = rcIndex(r, c);
                const botLeft = rcIndex(r+1, c);
                const botRight= rcIndex(r+1, c+1);
                // console.log(`(${r}, ${c}):`);
                // console.log(`(upright)`)
                // console.log(`\t(${r}, ${c})`);
                // console.log(`\t(${r+1}, ${c})`);
                // console.log(`\t(${r+1}, ${c+1})`);

                // "upright" triangle
                indices[idx++] = topLeft;
                indices[idx++] = botLeft;
                indices[idx++] = botRight;

                // only store a second triangle (an inverted one) if we're
                // not at the edge of the row
                if (c !== r) {
                    const topRight = rcIndex(r, c+1);

                    // "inverted" triangle
                    indices[idx++] = botRight;
                    indices[idx++] = topRight;
                    indices[idx++] = topLeft;
                    // console.log(`(inverted)`)
                    // console.log(`\t(${r}, ${c})`);
                    // console.log(`\t(${r}, ${c+1})`);
                    // console.log(`\t(${r+1}, ${c+1})`);
                }

            }
            // console.log(" ")
        }
        BvTri.indexBuffers[key] = indices;

        // only add offset AFTER caching the generated indices
        const indexBufferCopy = new Uint32Array(BvTri.indexBuffers[key]);
        for (let i = 0; i < indexBufferCopy.length; i++) {
            indexBufferCopy[i] += this.indexOffset;
        }
        return indexBufferCopy;
    }

    // TODO: like with other allocation, scratch should be allocated for the whole group
    getSubdivComputeNodes(positionStorage, highestComputedLevel, newLevel) {
        if (highestComputedLevel >= newLevel) {
            throw new Error();
        }
        // console.log("running tri subdiv");

        const triNr = (n) => (((n)*(n+1))/2);

        // "greatest number of invocations"
        const gni = triNr(1 << (newLevel-1)) * 3;

        // "scratch space per invocation"
        const sspi = triNr(this.deg);

        const spaceNeeded = gni * sspi;
        const scratch = new THREE.StorageBufferAttribute(
            spaceNeeded,
            3,
        );
        const scratchStorage = storage(scratch, "vec3", scratch.count);

        const nodes = [];
        for (let currLevel = highestComputedLevel; currLevel < newLevel; currLevel++) {
            const tpe = 1 << currLevel; // "triangles per edge"

            const node = subdivCompute({
                positionStorage: positionStorage,
                cpStorage: this.cpStorage,
                scratchStorage: scratchStorage,
                indexOffset: uniform(this.indexOffset),
                deg: uniform(this.deg),
                currComputedLevel: uniform(currLevel)
            }).compute(triNr(tpe) * 3);
            // console.log(`level ${currLevel}: dispatching ${triNr(tpe) * 3} computes`);

            nodes.push(node);
        }

        return nodes;
    }
}

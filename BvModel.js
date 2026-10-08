import * as THREE from "three/webgpu";

import {BvQuad} from './BvQuad.js';
import {BvTri} from './BvTri.js';
import {rsrc} from "./rsrc.js";

// TODO: yeah probably a better way to do this
const DEBUG_MODE = false;
// TODO: sync value with stateManager.js
const DEFAULT_SUBDIV_LEVEL = 2;

export class BvModel extends THREE.Group {

    /**
     * @param {[BvQuad]} allQuads
     * @param {[BvTri]}  allTris
     * @param {[     ]}  allPolys
     *
     * @param {Array<{name: string, patches: Array<(BvQuad | BvTri)>}>} patchesByGroup
     */
    constructor(allQuads, allTris, allPolys, patchesByGroup) {
        super();

        this.allQuads = allQuads;
        this.allTris = allTris;
        this.allPolys = allPolys;

        // level that's currently shown on screen (e.g. '2' for 4x4, '3' for 8x8)
        this.displayedLevel = -1;

        // what's the highest level we've already computed up to, i.e. the highest level
        // that we have the index/P/N/C buffers ready in memory
        this.computedLevel  = -1;



        // Essentially a map of <group name> to <group object>
        // dude we could totally rework the loader to just give the data
        // in this format. what are we doing.
        // TODO: that
        this.groups = {};
        const groupColors = [
            0x3aa99f, // cyan
            0x4385be, // blue
            0x8b7ec8, // purple
            0xce5d97, // magenta
            0xd14d41, // red
            0xda702c, // orange
            0xd0a215, // yellow
            0x879a39, // green
        ];
        let groupNr = 0;
        for (const group of patchesByGroup) {

            const regularMaterial = new THREE.MeshPhongNodeMaterial({
                color: groupColors[groupNr % groupColors.length],
                wireframe: false,
            });
            const wireframeMaterial = new THREE.MeshBasicNodeMaterial({
                // color: 0xcccccc,
                color: groupColors[groupNr % groupColors.length],
                wireframe: true,
            });

            this.groups[group.name] = {
                allPatches: group.patches,

                regularMaterial: regularMaterial,
                wireframeMaterial: wireframeMaterial,

                // This is like a map of <subdiv level> to <array of chunks>, e.g.:
                // chunksByLevel[0] = [ <chunk> ]
                // chunksByLevel[6] = [ <chunk>, <chunk> ]
                chunksByLevel: {},
            }
            groupNr++;
        }

        if (DEBUG_MODE) {
            console.log("Groups in BvModel, right after creation:");
            console.log(this.groups);
        }

        //====================
        // Pre-allocate buffers up to some reasonable size
        // 
        //====================
        this.allocatedLevel = -1;

        // basic heuristic to decide how many buffers get allocated up front
        const SIZE_THRESHOLD = 100_000_000;
        let preallocLevel;

        // evil k&r-esque loop; stop once a level is greater than SIZE_THRESHOLD
        for (
            preallocLevel = DEFAULT_SUBDIV_LEVEL;
            this.allQuads.length * (((1 << (preallocLevel+1))+ 1) * ((1 << (preallocLevel+1))+ 1) *  16) < SIZE_THRESHOLD &&
            preallocLevel <= 6;
            preallocLevel++
        );

        const largestSize = this.allQuads.length * (((1 << preallocLevel)+ 1) * ((1 << preallocLevel)+ 1) * 3 * 16);
        console.log(`Pre-allocating through to level ${preallocLevel} (${largestSize} bytes)`);
        this.allocChunks(preallocLevel);
    }

    setSubdivLevel(newLevel) {
        ////////////////////
        // Check for easy cases where we don't need to do any compute
        ////////////////////
        if (newLevel === this.displayedLevel) {
            // I'm not sure if lilgui lets this happen, but guard against changing the level
            // to what it's already set to
            return Promise.resolve();
        } else if (newLevel <= this.computedLevel) {
            // Already have positions/normals/curvatures/indices calculated for this level,
            // so only need to change which meshes are displayed

            // inherited THREE.Group method, remove all meshes from this group
            this.clear();

            for (const group of Object.values(this.groups)) {
                for (const chunk of group.chunksByLevel[newLevel]) {
                    // inherited THREE.Group method, add mesh to this group
                    this.add(chunk.threeJSMesh);
                }
            }

            // this.computedLevel = this.displayedLevel = newLevel;
            this.displayedLevel = newLevel;

            return Promise.resolve();
        }
        // Now past the early returns, so need to do all the work to subdivide the model

        ////////////////////
        // 1. Create, for every chunk of every group, index/pos/normal/curvature buffer
        ////////////////////
        this.allocChunks(newLevel);


        ////////////////////
        // 2. Now that index and p/n/c buffers have all been created, get to actually subdivide
        ////////////////////

        // TLDR-ish summary: basically, we can only use a finite amount of memory at a time,
        // and subdividing needs a lot of it. to avoid running out of memory, one big
        // buffer is shared between patches. each patch gets some piece of this buffer
        // to use as scratch, with all pieces being disjoint.
        // it's likely that this buffer won't have enough room for all the shaders to run
        // at once, so the idea is that we break the execution into 'phases'. basically,
        // if we run out of scratch space, but still have shaders to run, we run all the
        // shaders that fit (a 'phase'), and once those finish we then re-distribute pieces
        // of that buffer to the shaders that have yet to run. rinse repeat until all run.
        {
            const encoder = rsrc.device.createCommandEncoder({
                label: `setLevel encoder (lvl ${newLevel}) `,
            });
            const pass = encoder.beginComputePass({
                label: `setLevel pass (lvl ${newLevel}) `,
            });

            // TODO: error handling for out of memory
            const desiredSize = 100_000_000;
            const MY_SCRATCH = rsrc.device.createBuffer({
                label: `subdiv scratch buffer`,
                size: Math.min(rsrc.adapter.limits.maxBufferSize, desiredSize),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                mappedAtCreation: false,
            });
            console.log(`setSubdivLevel: allocated a ${MY_SCRATCH.size / 1_000_000} MB scratch buffer`);

            ////////////////////
            // Quads
            ////////////////////
            {
                // stupid mode (tested and it's just faster to change pipelines for each patch)
                for (const quad of this.allQuads) {
                    for (let currentLevel = 0; currentLevel < newLevel; currentLevel++){
                        pass.setPipeline(rsrc.quad.subdiv.vPipeline);
                        quad.vPass({
                            rsrc,
                            scratchBuffer: MY_SCRATCH,
                            pass,
                            currentLevel,
                            targetLevel: newLevel
                        });

                        pass.setPipeline(rsrc.quad.subdiv.uPipeline);
                        quad.uPass({
                            rsrc,
                            scratchBuffer: MY_SCRATCH,
                            pass,
                            currentLevel,
                            targetLevel: newLevel
                        });


                        // "don't re-calculate normals/curvature for a level we've already
                        // calculated them for"
                        if (currentLevel+1 > this.computedLevel) {
                            pass.setPipeline(rsrc.quad.evaluate.pipeline);
                            quad.evaluate({
                                rsrc,
                                scratchBuffer: MY_SCRATCH,
                                pass,
                                currentLevel: currentLevel+1,
                                targetLevel: newLevel
                            });
                        }
                    }
                }
            }


            ////////////////////
            // Tris
            ////////////////////
            {

            }

            //////////
            // Done with adding passes, submit and time execution
            //////////
            pass.end();
            const commandBuffer = encoder.finish();
            rsrc.device.queue.submit([commandBuffer]);
            MY_SCRATCH.destroy(); console.log("Scratch memory freed.");

            if (DEBUG_MODE) {
                rsrc.device.queue.onSubmittedWorkDone().then(() => {
                    // Retain sanity by copying back the position buffers for debugging
                    if (DEBUG_MODE) {
                        for (const [name, group] of Object.entries(this.groups)) {
                            const firstChunk = group.chunksByLevel[newLevel][0];

                            rsrc.renderer.getArrayBufferAsync(
                                firstChunk.threeJSMesh.geometry.getAttribute("position")
                            ).then((rawBuffer) => {
                                    console.log(`Running position buffer copy for group "${name}"...`);
                                    const fullArray = new Float32Array(rawBuffer);
                                    console.log(fullArray);
                                    console.log(`Its index buffer:`);
                                    console.log(firstChunk.threeJSMesh.geometry.index);
                                    // take small piece out for inspection?
                                    // const arr = Array.from(new Float32Array(rawBuffer));
                                    // console.log(arr.slice(52, Math.ceil((151 + 1) / 100) * 100));
                                    console.log(firstChunk.threeJSMesh);
                                    firstChunk.threeJSMesh.geometry.attributes['position'].array = fullArray;
                                });
                        }

                    }

                });
            }
        }

        ////////////////////
        // Update which meshes are shown
        // (remove those from the lower subdiv level, and now show this level's meshes)
        ////////////////////

        // inherited THREE.Group method, remove all meshes from this group
        this.clear();

        for (const group of Object.values(this.groups)) {
            for (const chunk of group.chunksByLevel[newLevel]) {
                // inherited THREE.Group method, add mesh to this group
                this.add(chunk.threeJSMesh);
            }
        }

        // we're done done done!
        this.computedLevel = this.displayedLevel = newLevel;

        const t0 = performance.now();
        return rsrc.device.queue.onSubmittedWorkDone().then(() => {
            const t1 = performance.now();
            console.log(`SSL: GPU side finished in ${(t1 - t0).toFixed(2)} ms (set level to ${newLevel})`);
        });
    }

    // create, and force a zeroing of, the chunks for levels above 'this.allocatedLevel'
    // and through to 'newLevel'
    //
    // e.g. calling this with this.allocatedLevel=3 and newLevel=6,
    // buffers will be created for levels 4, 5, and 6
    // (0-3 get skipped since they already exist)
    allocChunks(newLevel) {
        // "Do allocations through THREE, or manually?"
        //
        // the issue with using THREE for allocations is: whenever you create
        // a StorageBufferAttribute, THREE will ALWAYS create a CPU-side buffer of the same size,
        // and will ALWAYS copy it to the GPU. this makes buffer creation slower than it should
        // be in cases where you don't actually have anything to copy.
        const USE_THREE_ALLOC = false;
        let encoder;
        if (!USE_THREE_ALLOC) {
            encoder = rsrc.device.createCommandEncoder({label: "buffer init. encoder"});
        }


        const t0 = performance.now();
        if (this.allocatedLevel >= newLevel) {
            return;
        }

        const maxChunkSize = rsrc.adapter.limits.maxStorageBufferBindingSize;

        for (const [name, group] of Object.entries(this.groups)) {
            console.group(`Creating buffers for group: ${name}`);

            // Loop for all values between the highest computed, and newly set, level
            // e.g. if our highest computed level is 3, and user just set level to 5,
            // need to create buffers for BOTH levels 4 and 5
            for (let i = this.allocatedLevel+1; i <= newLevel; i++) {

                ////////////////////
                // 1. Figure out number of chunks needed to store P/N/C. This is
                //    bounded by the storage buffer size
                ////////////////////

                // for e.g. level 2 (aka 4x4) this gives (5*5 vec4) * (4 f32) * (4 bytes/f32)
                const quadByteSize = ((1 << i)+1) * ((1 << i)+1) * 4 * 4;
                // ((2^i)+1)-th triangle number, e.g. level 3 (aka 8x8) = ((2^3)+1) = 9th triangle nr
                const triByteSize  = (((1 << i)+1) * ((1 << i)+2)) >> 1;

                let currentChunk = Chunk();
                group.chunksByLevel[i] = [ currentChunk ];

                let byteOffset = 0;
                for (const patch of group.allPatches) {
                    let patchByteSize;
                    if (patch instanceof BvQuad) {
                        patchByteSize = quadByteSize;
                    } else if (patch instanceof BvTri) {
                        patchByteSize = triByteSize;
                    }

                    // If the current chunk won't have enough room for this patch,
                    // create a new chunk, and put this patch in it
                    if (byteOffset + patchByteSize > maxChunkSize) {
                        currentChunk = Chunk();
                        group.chunksByLevel[i].push(currentChunk);

                        byteOffset = 0;
                    }

                    patch.buffersByLevel[i] = {
                        indexOffset: byteOffset >> 4,
                        parentChunk: currentChunk
                    }

                    currentChunk.patches.push(patch);
                    currentChunk.byteSize += patchByteSize;

                    byteOffset += patchByteSize;

                    // TODO: this feels a bit hack
                    if (patch instanceof BvQuad) {
                        currentChunk.nrQuads++;
                    } else if (patch instanceof BvTri) {
                        currentChunk.nrTris++;
                    }
                }

                ////////////////////
                // 2. Now that we know how many chunks we need,
                //    create index/P/N/C buffers for each chunk
                ////////////////////

                // (number of quads) * (2 tris per quad) * (3 indicies per tri)
                const indicesPerQuad = ((1 << i) * (1 << i)) * 2 * 3;
                // (number of tris)  * (3 indicies per tri)
                const indicesPerTri  = ((1 << i) * (1 << i)) * 3;

                for (const chunk of group.chunksByLevel[i]) {

                    //========================================
                    // P/N/C buffers for whole chunk
                    //========================================
                    const t0_pnc = performance.now();
                    {
                        if (USE_THREE_ALLOC) {

                            // Positions
                            const positionBuffer = new THREE.StorageBufferAttribute(
                                chunk.byteSize >> 4,
                                4,
                            );
                            chunk.threeJSMesh.geometry.setAttribute("position", positionBuffer);
                            rsrc.backend.createStorageAttribute(
                                chunk.threeJSMesh.geometry.getAttribute("position")
                            );
                            chunk.positionBuffer = rsrc.backend.get(
                                chunk.threeJSMesh.geometry.getAttribute("position")
                            ).buffer;

                            // Normals
                            const normalBuffer = new THREE.StorageBufferAttribute(
                                chunk.byteSize >> 4,
                                4,
                            );
                            chunk.threeJSMesh.geometry.setAttribute("normal", normalBuffer);
                            rsrc.backend.createStorageAttribute(
                                chunk.threeJSMesh.geometry.getAttribute("normal")
                            );
                            chunk.normalBuffer = rsrc.backend.get(
                                chunk.threeJSMesh.geometry.getAttribute("normal")
                            ).buffer;

                            // Curvature
                            const curvatureBuffer = new THREE.StorageBufferAttribute(
                                chunk.byteSize >> 4,
                                4,
                            );
                            chunk.threeJSMesh.geometry.setAttribute("curvature", curvatureBuffer);
                            rsrc.backend.createStorageAttribute(
                                chunk.threeJSMesh.geometry.getAttribute("curvature")
                            );
                            chunk.curvatureBuffer = rsrc.backend.get(
                                chunk.threeJSMesh.geometry.getAttribute("curvature")
                            ).buffer;
                        } else {
                            // Positions
                            chunk.positionBuffer = rsrc.device.createBuffer({
                                label: `position buffer (lvl ${i})`,
                                size : chunk.byteSize,
                                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                                mappedAtCreation: false,
                            });

                            const threePosBuf = new THREE.StorageBufferAttribute(new Float32Array(4), 4);
                            threePosBuf.count = chunk.byteSize >> 4;

                            rsrc.backend.get(threePosBuf).buffer = chunk.positionBuffer;

                            chunk.threeJSMesh.geometry.boundingSphere =
                                new THREE.Sphere(new THREE.Vector3(), 1);
                            chunk.threeJSMesh.geometry.setAttribute("position", threePosBuf);



                            // Normals
                            chunk.normalBuffer = rsrc.device.createBuffer({
                                label: `normal buffer (lvl ${i})`,
                                size : chunk.byteSize,
                                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                                mappedAtCreation: false,
                            });
                            const threeNormalBuf = new THREE.StorageBufferAttribute(new Float32Array(4), 4);
                            threeNormalBuf.count = chunk.byteSize >> 4;

                            rsrc.backend.get(threeNormalBuf).buffer = chunk.normalBuffer;
                            chunk.threeJSMesh.geometry.setAttribute("normal", threeNormalBuf);


                            // Curvature
                            chunk.curvatureBuffer = rsrc.device.createBuffer({
                                label: `curvature buffer (lvl ${i})`,
                                size : chunk.byteSize,
                                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                                mappedAtCreation: false,
                            });
                            const threeCurvBuf = new THREE.StorageBufferAttribute(new Float32Array(4), 4);
                            threeCurvBuf.count = chunk.byteSize >> 4;

                            rsrc.backend.get(threeCurvBuf).buffer = chunk.curvatureBuffer;
                            chunk.threeJSMesh.geometry.setAttribute("curvature", threeCurvBuf);

                            ///////////
                            ///////////
                            encoder.clearBuffer(chunk.positionBuffer);
                            encoder.clearBuffer(chunk.normalBuffer);
                            encoder.clearBuffer(chunk.curvatureBuffer);
                        }
                    }
                    const t1_pnc = performance.now();
                    console.log(`PNC: ${t1_pnc - t0_pnc} ms (${chunk.byteSize} bytes = ${chunk.byteSize / 1_000_000} MB) (level ${i})`);

                    //========================================
                    // Index buffer for whole chunk
                    //========================================
                    {
                        // const t0_idx = performance.now();

                        const nrOfIndices =
                            (chunk.nrQuads * indicesPerQuad) +
                            (chunk.nrTris  * indicesPerTri);
                        const indices = new Uint32Array(nrOfIndices);

                        let idx = 0;
                        for (const patch of chunk.patches) {
                            patch.getIndexBuffer(i, indices, idx);

                            // TODO: this feels a bit hack
                            if (patch instanceof BvQuad) {
                                idx += indicesPerQuad;
                            } else if (patch instanceof BvTri) {
                                idx += indicesPerTri;
                            }
                        }
                        // const ty = performance.now();
                        // console.log(`${ty-t0_idx}ms to do CPU IB creation`);

                        const indexBuffer = new THREE.Uint32BufferAttribute(indices, 1);
                        chunk.indexBuffer = indexBuffer;
                        chunk.threeJSMesh.geometry.setIndex(indexBuffer);
                        chunk.threeJSMesh.geometry.index.version++;
                        const t1_idx = performance.now();
                        // console.log(`IDX: ${t1_idx - t0_idx} ms (${nrOfIndices * 4} bytes = ${(nrOfIndices * 4) / 1_000_000} MB) (level ${i})`);

                    }

                    //========================================
                    // Set material for whole chunk
                    //========================================
                    chunk.threeJSMesh.material = group.regularMaterial;

                }

            }

            console.groupEnd();
        }
        const t1 = performance.now();
        console.log(`allocChunks took ${(t1 - t0).toFixed(2)} ms (created buffers for levels ${this.allocatedLevel+1}-${newLevel})`);

        this.allocatedLevel = newLevel;
        if (!USE_THREE_ALLOC) {
            console.log("CLEARING BUFFERS...");
            const t0 = performance.now();
            rsrc.device.queue.submit([encoder.finish()]);
            return rsrc.device.queue.onSubmittedWorkDone().then(() => {
                const t1 = performance.now();
                console.log(`BUFFER CLEAR took ${(t1 - t0).toFixed(2)} ms`);
            })
        } else {
            return Promise.resolve();
        }
    }


    // De-alloc everything for this model. obviously only call 'destroy' when you're getting rid
    // of the model
    destroy() {
        for (const group of Object.values(this.groups)) {
            group.regularMaterial.dispose();
            group.wireframeMaterial.dispose();

            for (const patch of group.allPatches) {
                patch.cpBuffer.destroy();
                patch.patchUniformBuffer.destroy();
            }

            // "for each level, destroy each chunk"
            for (const chunkArray of Object.values(group.chunksByLevel)) {
                for (const chunk of chunkArray) {
                    chunk.threeJSMesh.geometry.dispose();
                }
            }
        }
    }
}


function Chunk() {
    return {
        patches: [],
        byteSize: 0,

        nrQuads : 0,
        nrTris  : 0,

        indexBuffer    : -1,
        positionBuffer : -1,
        normalBuffer   : -1,
        curvatureBuffer: -1,

        threeJSMesh: new THREE.Mesh(),

        // will be assigned to a storageBufferAttribute once needed size is known
        positionStorage: -1,
    };
}

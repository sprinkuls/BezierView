import * as THREE from "three/webgpu";

import {BvQuad} from './BvQuad.js';
import {BvTri} from './BvTri.js';
import {rsrc} from "./rsrc.js";

// TODO: yeah probably a better way to do this
const DEBUG_MODE = false;

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

        // TODO: don't get this from rsrc probably!
        // biiiig GPUBuffer that's used when we need to subdivide patches
        this.honkinBuffer = rsrc.honkinBuffer;

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

        this.setSubdivLevel(0);
    }

    setSubdivLevel(newLevel) {
        ////////////////////
        // Check for easy cases where we don't need to do any compute
        ////////////////////
        if (newLevel === this.displayedLevel) {
            // I'm not sure if lilgui lets this happen, but guard against changing the level
            // to what it's already set to
            return;
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

            return;
        }

        ////////////////////
        // Now past the early returns, so need to do all the work to subdivide the model
        //
        // 1. Create, for every chunk of every group, index/pos/normal/curvature buffer
        ////////////////////
        {
            const maxChunkSize = rsrc.adapter.limits.maxStorageBufferBindingSize;

            for (const [name, group] of Object.entries(this.groups)) {
                console.log(`creating buffers for group: ${name}`);

                // Loop for all values between the highest computed, and newly set, level
                // e.g. if our highest computed level is 3, and user just set level to 5,
                // need to create buffers for BOTH levels 4 and 5
                for (let i = this.computedLevel+1; i <= newLevel; i++) {

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

                        //////////
                        // Index buffer for whole chunk
                        //////////
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
                        const indexBuffer = new THREE.Uint32BufferAttribute(indices, 1);
                        chunk.indexBuffer = indexBuffer;
                        chunk.threeJSMesh.geometry.setIndex(indexBuffer);
                        chunk.threeJSMesh.geometry.index.version++;

                        //////////
                        // P/N/C buffers for whole chunk
                        //////////

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
                            3,
                        );
                        chunk.threeJSMesh.geometry.setAttribute("curvature", curvatureBuffer);
                        rsrc.backend.createStorageAttribute(
                            chunk.threeJSMesh.geometry.getAttribute("curvature")
                        );
                        chunk.curvatureBuffer = rsrc.backend.get(
                            chunk.threeJSMesh.geometry.getAttribute("curvature")
                        ).buffer;

                        //////////
                        // Set material
                        //////////
                        chunk.threeJSMesh.material = group.wireframeMaterial;
                        // chunk.threeJSMesh.material = group.regularMaterial;
                    }

                }

            }
        }

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


            ////////////////////
            // Quads
            ////////////////////
            {

                let phases = [];
                let currentPhase = [];
                phases.push(currentPhase);

                let byteOffset = 0;
                for (const quad of this.allQuads) {
                    let quadByteSize =
                        ((quad.uDeg * (1 << newLevel)) + 1) *
                            ((quad.vDeg * (1 << newLevel)) + 1) *
                            4 * 4;

                    // need to align to 256 bytes to use as offset/size (by spec)
                    const remainder = quadByteSize & (255); // mod by 256
                    if (remainder !== 0) {
                        quadByteSize = ((quadByteSize >> 8) + 1) * 256;
                        // console.log(`size: ${quadByteSize} (a: ${a}, rem: ${remainder})`);
                    }

                    // no scratch mean new phase must begin
                    if (byteOffset + quadByteSize > this.honkinBuffer.size) {
                    // TODO: TESTING: 128MB demo size
                    // if (byteOffset + quadByteSize > 134_217_728) {
                        currentPhase = [];
                        phases.push(currentPhase);

                        byteOffset = 0;
                    }

                    // store byte offset into honkin buffer, and how large this patch is
                    // TODO: possibly problematic to store this on patches themselves, since
                    // can become out of date
                    quad.byteOffset = byteOffset;
                    quad.byteSize = quadByteSize;
                    currentPhase.push(quad);

                    byteOffset += quadByteSize;
                }

                ////////////////////
                // Quad phases have been created, so we can now move to dispatching shaders
                ////////////////////
                console.log(`${phases.length} compute phases needed to subdivide to level ${newLevel}`);
                for (const phase of phases) {
                    // console.log('Phase:');
                    // console.log(phase);

                    // TODO: this
                    // we kinda have stupid special case for when level is 0
                    if (newLevel === 0) {
                        pass.setPipeline(rsrc.quad.evaluate.pipeline);
                        for (const quad of phase) {
                            quad.evaluate({
                                rsrc,
                                pass,
                                currentLevel: 0,
                                targetLevel: newLevel
                            });
                        }
                    } else {
                        // loop for all levels fun fun fun
                        for (let currentLevel = 0; currentLevel < newLevel; currentLevel++) {

                            // < do ALL v passes at once >
                            pass.setPipeline(rsrc.quad.subdiv.vPipeline);
                            for (const quad of phase) {
                                quad.vPass({rsrc, pass, currentLevel, targetLevel: newLevel});
                            }

                            // < do ALL u passes at once >
                            pass.setPipeline(rsrc.quad.subdiv.uPipeline);
                            for (const quad of phase) {
                                quad.uPass({rsrc, pass, currentLevel, targetLevel: newLevel});
                            }

                            // < do ALL evaluation at once >
                            // idea with currentLevel+1 is just that we've already subdivided the
                            // patch, so now the level has in fact increased
                            pass.setPipeline(rsrc.quad.evaluate.pipeline);
                            for (const quad of phase) {
                                quad.evaluate({
                                    rsrc,
                                    pass,
                                    currentLevel: currentLevel+1,
                                    targetLevel: newLevel
                                });
                            }
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

            const t0 = performance.now();
            rsrc.device.queue.submit([commandBuffer]);

            rsrc.device.queue.onSubmittedWorkDone().then(() => {
                const t1 = performance.now();
                console.log(`GPU side finished in ${(t1 - t0).toFixed(2)} ms (set level to ${newLevel})`);

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

        return;
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

            // for each LEVEL
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

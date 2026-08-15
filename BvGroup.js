import * as THREE from "three/webgpu";
import * as Quad from "./BvQuad.js";
import { storage, Fn, float, element, If, equal, assign, div, vertexIndex, positionGeometry } from 'three/tsl';

// a single BV file might be larger than the max size for a single alloc. (128 MiB by default),
// so we just break each file into 'chunks' that are below that max size. this isn't
// super commonly needed but can happen (e.g. the logo.bv file at subdiv. level 6)
function Chunk() {
    return {
        patches: [],
        byteSize: 0,
        // for the 7 subdiv. levels, 0 to 6
        indexBuffers: new Array(7),
        mesh: new THREE.Mesh(),
        // will be assigned to a storageBufferAttribute once needed size is known
        positionStorage: -1,
    };
}

export class BvGroup extends THREE.Group {
    /**
     * @param {string} name
     * @param {[{}]} patches
     * @param {THREE.WebGPURenderer} renderer
     */
    constructor(name, patches, renderer) {
        super();
        this.name = name;
        this.renderer = renderer;

        /** @type {int} */
        this.highestComputedLevel = 0;

        /** @type {[Chunk]} */
        this.chunks = [];
        this.chunks.push(Chunk());

        ////////// assign each patch a location in the large chunk buffer //////////

        // 134,217,728 bytes (128MiB) is the smallest guaranteed storage buffer size
        // so split up data into chunks NO LARGER THAN 128MiB
        // TODO: check for greater device-specific alloc. limit during initialization
        const maxChunkSize = 134_217_728;

        let byteOffset = 0;
        for (const patch of patches) {
            const byteSize = patch.getMemNeededBytes();
            if (byteOffset + byteSize > maxChunkSize) {
                // console.log("NEW CHUNK!");
                this.chunks.at(-1).byteSize = byteOffset;

                byteOffset = 0;
                this.chunks.push(Chunk());
            }
            patch.byteOffset = byteOffset;
            patch.indexOffset = byteOffset / (4 * 4); // each point is 4 Float32s (even vec3s are vec4s under the hood)
            this.chunks.at(-1).patches.push(patch);
            this.chunks.at(-1).byteSize += byteSize;

            byteOffset += byteSize;
            // console.log(`${maxChunkSize} vs ${byteOffset}`);
        }

        let lens = 0;
        for (const chunk of this.chunks) {
            // since BvGroup is extending a THREE group, we add all the
            // chunk meshes to this group
            this.add(chunk.mesh);
            lens += chunk.patches.length;
        }
        // TESTING: remove this
        console.log(`Total # of patches: ${lens}`);

        ////////// initialize chunks, and the patches they contain //////////
        const initComputeNodes = new Array(patches.length);
        let idx = 0;
        for (const chunk of this.chunks) {
            // create the position buffer for the entire chunk
            const positionSBA = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                4,
            );
            chunk.mesh.geometry.setAttribute("position", positionSBA);
            chunk.positionStorage = storage(
                chunk.mesh.geometry.getAttribute("position"),
                "vec4",
                chunk.mesh.geometry.getAttribute("position").count,
            );

            // copy the control points for each patch into this new buffer.
            // done on the GPU, so we get the compute nodes for each then compute all at once

            // TODO: precalc the size of this
            const indices = [];
            for (const patch of chunk.patches) {
                initComputeNodes[idx] = patch.getInitComputeNode(
                    chunk.positionStorage,
                );
                indices.push(...patch.getIndexBuffer(0));
                idx++;
            }
            const typedIndices = new THREE.Uint32BufferAttribute(indices, 1);
            chunk.indexBuffers[0] = typedIndices;
            chunk.mesh.geometry.setIndex(typedIndices);
            chunk.mesh.geometry.index.version++;
        }
        this.renderer.compute(initComputeNodes);



        // all patches in all chunks are now initialized

        // vertex shader to adjust all points by their rational value
        const rationalVS = Fn(({ vertexBuffer}) => {
            const w = float(vertexBuffer.element(vertexIndex).w);

            // TODO: should we use another value than 1?
            If (equal(w, 0), () => {
                w.assign(1);
            });

            return div(positionGeometry, w);
        });

        // assign a material and a vertex shader to all meshes
        for (const chunk of this.chunks) {
            const material = new THREE.MeshBasicNodeMaterial({
                color: 0xff00c0,
                wireframe: true,
            });

            material.positionNode = rationalVS({
                vertexBuffer: chunk.positionStorage,
            });

            chunk.mesh.material = material;
        }
    }

    setLevel(newLevel) {
        if (newLevel > 6 || newLevel < 0) {
            throw new Error();
        }

        // need to subdivide the patch if a higher level than we've previously subdivided is requested
        if (newLevel > this.highestComputedLevel) {
            const computeNodes = [];
            for (const chunk of this.chunks) {
                for (const patch of chunk.patches) {
                    // (get all the compute nodes)
                    computeNodes.push(
                        ...patch.getSubdivComputeNodes(
                            chunk.positionStorage,
                            this.highestComputedLevel,
                            newLevel,
                        ),
                    );
                }
            }

            // TESTING: remove this
            const oldHighestLevel = this.highestComputedLevel;
            console.time("setLevel")
            this.renderer
                .resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE)
                .then(() => {
                    this.renderer.computeAsync(computeNodes).then(() => {
                        console.timeEnd("setLevel")

                        this.renderer
                            .resolveTimestampsAsync(
                                THREE.TimestampQuery.COMPUTE,
                            )
                            .then((ms) => {
                                console.log(ms);
                                console.log(
                                    `Compute: ${oldHighestLevel} -> ${newLevel} took ${ms.toFixed(3)}ms`,
                                );
                            });
                    });
                });
            this.highestComputedLevel = newLevel;
        }

        // set index buffers
        // (we check index 0 of chunks because there will always be at least one chunk,
        // and all chunks will always be at the same level)
        if (this.chunks[0].indexBuffers[newLevel] !== undefined) {
            for (const chunk of this.chunks) {
                chunk.mesh.geometry.setIndex(chunk.indexBuffers[newLevel]);
            }
        } else {
            console.log(`Chunk cache miss: level ${newLevel}`);
            const indices = [];
            for (const chunk of this.chunks) {
                for (const patch of chunk.patches) {
                    indices.push(...patch.getIndexBuffer(newLevel));
                }
                const typedIndices = new THREE.Uint32BufferAttribute(
                    indices,
                    1,
                );
                chunk.indexBuffers[newLevel] = typedIndices;
                chunk.mesh.geometry.setIndex(typedIndices);
                chunk.mesh.geometry.index.version++;
            }
        }
    }
}

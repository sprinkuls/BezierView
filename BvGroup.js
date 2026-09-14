import * as THREE from "three/webgpu";
import { storage, Fn, float, element, If, equal, assign, div, vertexIndex, positionGeometry } from 'three/tsl';
import {rsrc} from "./rsrc.js";

// a single BV file might be larger than the max size for a single alloc. (128 MiB by default),
// so we just break each file into 'chunks' that are below that max size. this isn't
// super commonly needed but can happen (e.g. the logo.bv file at subdiv. level 6)
function Chunk() {
    return {
        patches: [],
        byteSize: 0,
        // for the 7 subdiv. levels, 0 to 6
        indexBuffers: new Array(7),

        positionBuffer: -1,
        normalBuffer: -1,
        curvatureBuffer: -1,

        mesh: new THREE.Mesh(),

        // will be assigned to a storageBufferAttribute once needed size is known
        positionStorage: -1,
    };
}


export class BvGroup extends THREE.Group {
    /**
     * @param {string} name
     * @param {[{}]} patches
     */

    constructor(name, patches) {
        super();
        this.name = name;
        this.highestComputedLevel = 0;

        /** @type {[Chunk]} */
        this.chunks = [];
        // always have SOMETHING in a group, hence at least one group
        this.chunks.push(Chunk()); 

        // ====================
        // Assign each patch a piece of the chunk's pos/normal/curvature buffers
        // ====================

        // 134,217,728 bytes (128MiB) is the smallest guaranteed storage buffer size
        // so split up data into chunks NO LARGER THAN 128MiB
        // TODO: check for greater device-specific alloc. limit during initialization
        const maxChunkSize = 134_217_728;
        const byteSize = 5722 * 4 * 4;

        let byteOffset = 0;
        for (const patch of patches) {
            // if not enough space for this patch, create a new chunk
            if (byteOffset + byteSize > maxChunkSize) {
                this.chunks.at(-1).byteSize = byteOffset;
                this.chunks.push(Chunk());

                byteOffset = 0;
            }

            patch.indexOffset = byteOffset >> 4;
            this.chunks.at(-1).patches.push(patch);
            this.chunks.at(-1).byteSize += byteSize;
            byteOffset += byteSize;
        }

        // now that all chunks are created, add their meshes to this BvGroup so THREE displays them
        for (const chunk of this.chunks) {
            this.add(chunk.mesh);
        }

        // TESTING: remove this
        let nrOfPatches = 0;
        for (const chunk of this.chunks) {
            nrOfPatches += chunk.patches.length;
        }
        console.log(`Total # of patches: ${nrOfPatches}`);


        // ====================
        // Create position, normal, curvature, and index buffers for each chunk 
        // ====================

        for (const chunk of this.chunks) {
            // Position buffer
            const positionBuffer = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                4,
            );
            chunk.mesh.geometry.setAttribute("position", positionBuffer);
            rsrc.backend.createStorageAttribute(chunk.mesh.geometry.getAttribute("position"));
            chunk.positionBuffer = rsrc.backend.get(chunk.mesh.geometry.getAttribute("position")).buffer;

            // this is kept around because our rational vertex shaders are still written in TSL
            chunk.positionStorage = storage(
                chunk.mesh.geometry.getAttribute("position"),
                "vec4",
                chunk.mesh.geometry.getAttribute("position").count,
            );


            // Normal buffer
            const normalBuffer = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                4,
            );
            // const normalBuffer = new THREE.StorageBufferAttribute(
            //     new Float32Array([
            //          1,  1, 1, 1,
            //          1,  1, 1, 1,
            //          1, -1, 1, 1,
            //          1, -1, 1, 1,
            //     ]),
            //     4,
            // );
            chunk.mesh.geometry.setAttribute("normal", normalBuffer);
            rsrc.backend.createStorageAttribute(chunk.mesh.geometry.getAttribute("normal"));
            chunk.normalBuffer = rsrc.backend.get(chunk.mesh.geometry.getAttribute("normal")).buffer;


            // Curvature buffer
            const curvatureBuffer = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                3,
            );
            chunk.mesh.geometry.setAttribute("curvature", curvatureBuffer);
            rsrc.backend.createStorageAttribute(chunk.mesh.geometry.getAttribute("curvature"));
            chunk.curvatureBuffer = rsrc.backend.get(chunk.mesh.geometry.getAttribute("curvature")).buffer;


            // Index buffer
            // TODO: nrIndices will be different once tris are in (3 per patch rather than 6)
            const nrIndices = 6 * chunk.patches.length;
            const indices = new Uint32Array(nrIndices);

            let idx = 0;
            for (const patch of chunk.patches) {
                patch.getIndexBuffer(0, indices, idx);
                idx += 6;
            }
            // TODO: TESTING: REMOVE ME
            // console.log("indices:");
            // console.log(indices);

            const indexBuffer0 = new THREE.Uint32BufferAttribute(indices, 1);
            chunk.indexBuffers[0] = indexBuffer0;
            chunk.mesh.geometry.setIndex(indexBuffer0);
            chunk.mesh.geometry.index.version++;
        }


        // ====================
        // Initialize patches, use shaders to copy control points to correct locations
        // in the position buffer
        // ====================


        // TODO: this only works on quads! fix that!
        const encoder = rsrc.device.createCommandEncoder({
            label: 'Quad. init. encoder',
        });
        const pass = encoder.beginComputePass({
            label: 'Quad. init. pass',
        });
        pass.setPipeline(rsrc.quad.init.pipeline);

        for (const chunk of this.chunks) {
            for (const patch of chunk.patches) {
                patch.initPatch(
                    rsrc,
                    pass,
                    chunk.positionBuffer,
                    chunk.normalBuffer,
                    chunk.curvatureBuffer
                );
            }
        }
        pass.end();

        const commandBuffer = encoder.finish();
        rsrc.device.queue.submit([commandBuffer]);


        // ====================
        // Assign vertex shaders, materials to all chunks
        // ====================

        for (const chunk of this.chunks) {

            // const material = new THREE.MeshStandardNodeMaterial({
            const material = new THREE.MeshPhongNodeMaterial({
                // color: 0x60a0ff,
                color: 0x24837b,
                wireframe: false,
                // roughness: 0.2,
            });
            const wireframeMaterial = new THREE.MeshBasicNodeMaterial({
                // color: 0x000000,
                color: 0xcccccc,
                wireframe: true,
            });

            // material.positionNode = BvGroup.rationalVS({
            //     vertexBuffer: chunk.positionStorage,
            // });
            // wireframeMaterial.positionNode = BvGroup.rationalVS({
            //     vertexBuffer: chunk.positionStorage,
            // });

            chunk.material = material;
            chunk.wireframeMaterial = wireframeMaterial;

            // TODO : TESTING : REMOVE THIS
            chunk.mesh.material = material;
            // chunk.mesh.material = wireframeMaterial;
        }
    }

    setLevel(newLevel, pass) {
        // console.log("i'm normal");
        // console.log(this.chunks[0].mesh.geometry.attributes.normal);

        if (newLevel > 6 || newLevel < 0) {
            throw new Error();
        }

        // Subdivide the patch if a higher level than we've previously subdivided is requested
        if (newLevel > this.highestComputedLevel) {
            for (const chunk of this.chunks) {
                for (const patch of chunk.patches) {
                    patch.subdivPatch({
                        rsrc,
                        pass,
                        highestComputedLevel: this.highestComputedLevel,
                        newLevel: newLevel,
                    });
                }
            }

            this.highestComputedLevel = newLevel;

            // if (newLevel >= this.highestComputedLevel) {
            //     for (const chunk of this.chunks) {
            //         for (const patch of chunk.patches) {
            //             patch.calcNormals({
            //                 rsrc,
            //                 pass,
            //                 level: newLevel,
            //             });
            //         }
            //     }
            // }
        }

        // Set index buffers
        // (we check index 0 of chunks because there will always be at least one chunk,
        // and all chunks will always be at the same level)
        if (this.chunks[0].indexBuffers[newLevel] !== undefined) {
            for (const chunk of this.chunks) {
                chunk.mesh.geometry.setIndex(chunk.indexBuffers[newLevel]);
                chunk.mesh.geometry.index.version++;
            }
        } else {
            for (const chunk of this.chunks) {
                const indicesPerPatch = ((1 << newLevel) * (1 << newLevel)) * 6;
                const nrIndices = indicesPerPatch * chunk.patches.length;
                const indices = new Uint32Array(nrIndices);

                let idx = 0;
                for (const patch of chunk.patches) {
                    patch.getIndexBuffer(newLevel, indices, idx);
                    idx += indicesPerPatch;
                }
                const indexBuffer = new THREE.Uint32BufferAttribute(indices, 1);
                chunk.indexBuffers[newLevel] = indexBuffer;
                chunk.mesh.geometry.setIndex(indexBuffer);
                chunk.mesh.geometry.index.version++;
            }
        }

        // TODO: TESTING : REMOVE ME
        // copy over the position buffer for debugging
        // setTimeout(() => {
        //     console.log(`running copy (level ${newLevel}...`);
        //     rsrc.renderer.getArrayBufferAsync(
        //         this.chunks[0].mesh.geometry.getAttribute("position")
        //     ).then((rawBuffer) => {
        //         const arr = Array.from(new Float32Array(rawBuffer));
        //         console.log(arr.slice(52, Math.ceil((151 + 1) / 100) * 100));
        //     });
        // }, 3000);
    }

    // TODO: merge this into setLevel; broken out on its own for now to just get the thing working
    calcNormals(level, pass) {
        if (level >= this.highestComputedLevel) {
            for (const chunk of this.chunks) {
                for (const patch of chunk.patches) {
                    patch.calcNormals({
                        rsrc,
                        pass,
                        level,
                    });
                }
            }
        }
    }

    // Vertex shader to adjust all points by their rational value
    // TODO: swap for WGSL?
    static rationalVS = Fn(({ vertexBuffer}) => {
        const w = float(vertexBuffer.element(vertexIndex).w);

        // TODO: should we use another value than 1?
        If (equal(w, 0), () => {
            w.assign(1);
        });

        return div(positionGeometry, w);
    });
 
}

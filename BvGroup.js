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
                // create a new chunk
                this.chunks.at(-1).byteSize = byteOffset;

                byteOffset = 0;
                this.chunks.push(Chunk());
            }
            patch.byteOffset = byteOffset;
            // each point is 4 Float32s (even vec3s are vec4s under the hood)
            patch.indexOffset = byteOffset / (4 * 4); 
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


        // ====================
        // Create the position and index buffers for each chunk
        // ====================

        for (const chunk of this.chunks) {
            // Create the position buffer for the entire chunk
            const positionBuffer = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                4,
            );
            chunk.mesh.geometry.setAttribute("position", positionBuffer);
            rsrc.backend.createStorageAttribute(chunk.mesh.geometry.getAttribute("position"));

            // this is kept around because our vertex shaders are still written in TSL
            chunk.positionStorage = storage(
                chunk.mesh.geometry.getAttribute("position"),
                "vec4",
                chunk.mesh.geometry.getAttribute("position").count,
            );

            // Create the index buffer for the entire chunk
            // TODO: precalc. the size of this (and just do this more efficiently)
            // (precalc size of array buffer and then create a single Uint32Array TypedArray,
            //  with each patch writing into that array at a given offset)
            const indices = [];
            for (const patch of chunk.patches) {
                indices.push(...patch.getIndexBuffer(0));
            }
            const typedIndices = new THREE.Uint32BufferAttribute(indices, 1);

            chunk.mesh.geometry.setIndex(typedIndices);
            chunk.mesh.geometry.index.version++;
            chunk.indexBuffers[0] = typedIndices;
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
            // The GPUBuffer that points to this chunk's position buffer
            const chunkPosBuffer = rsrc.backend.get(
                chunk.mesh.geometry.getAttribute('position')
            ).buffer;

            for (const patch of chunk.patches) {
                patch.initPatch(rsrc, pass, chunkPosBuffer);
            }
        }
        pass.end();

        const commandBuffer = encoder.finish();
        rsrc.device.queue.submit([commandBuffer]);


        // ====================
        // Assign vertex shaders, materials to all chunks
        // ====================

        for (const chunk of this.chunks) {

            const material = new THREE.MeshStandardNodeMaterial({
            // const material = new THREE.MeshPhongNodeMaterial({
                color: 0x60a0ff,
                wireframe: false,
                roughness: 0.2,
            });
            const wireframeMaterial = new THREE.MeshBasicNodeMaterial({
                // color: 0x000000,
                color: 0xcccccc,
                wireframe: true,
            });

            material.positionNode = BvGroup.rationalVS({
                vertexBuffer: chunk.positionStorage,
            });
            wireframeMaterial.positionNode = BvGroup.rationalVS({
                vertexBuffer: chunk.positionStorage,
            });

            chunk.material = material;
            chunk.wireframeMaterial = wireframeMaterial;

            // TODO : TESTING : REMOVE THIS
            chunk.mesh.material = material;
            // chunk.mesh.material = wireframeMaterial;
        }
    }

    setLevel(newLevel, pass) {
        if (newLevel > 6 || newLevel < 0) {
            throw new Error();
        }

        // need to subdivide the patch if a higher level than we've previously subdivided is requested
        if (newLevel > this.highestComputedLevel) {
            for (const chunk of this.chunks) {
                const chunkPosBuffer = rsrc.backend.get(
                    chunk.mesh.geometry.getAttribute('position')
                ).buffer;

                for (const patch of chunk.patches) {
                    patch.subdivPatch({
                        rsrc,
                        pass,
                        chunkPosBuffer: chunkPosBuffer,
                        highestComputedLevel: this.highestComputedLevel,
                        newLevel: newLevel,
                    });
                }
            }

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
            // console.log(`Chunk cache miss: level ${newLevel}`);
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

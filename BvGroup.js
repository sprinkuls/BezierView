import * as THREE from "three/webgpu";
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

        /** @type {GPUDevice} */
        this.device = renderer.backend.device;
        /** @type {GPUCanvasContext} */
        this.context = renderer.getContext();

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
        // TODO: these (i.e. pipelines, shaders) should be cached across all BvGroups
        const module = this.device.createShaderModule({
            label: 'Quad. init. shader module',
            code: /* wgsl */ `
            @group(0) @binding(0) var<storage, read_write> vertex_buffer: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> cp_buffer : array<vec4<f32>>;

            struct patchInfo {
                index_offset: u32,
                u_deg : u32,
                v_deg : u32,
            }
            @group(0) @binding(2) var<uniform> p : patchInfo;

            @compute @workgroup_size(1) fn initPatch() {
                // "highest (subdivision) level"
                const hl : u32 = 6;
                // (2 ^ hl), so for hl=6 this is 64
                const max_quads = 1 << hl;

                for (var i : u32 = 0; i <= p.u_deg; i++) {
                    for (var j : u32 = 0; j <= p.v_deg; j++) {
                        let src : u32 = ((p.v_deg+1) * i) + j;
                        let dst : u32 = ((i * max_quads) * ((p.v_deg * max_quads) + 1)) + (j * max_quads);
                        vertex_buffer[dst + p.index_offset] = cp_buffer[src];
                    }
                }
            }
            `,
        });

        const pipeline = this.device.createComputePipeline({
            label: 'Quad. init. compute pipeline',
            layout: 'auto',
            compute: {
                module: module,
                entryPoint: 'initPatch',
            },
        });

        const encoder = this.device.createCommandEncoder({
            label: 'Quad. init. encoder',
        });
        const pass = encoder.beginComputePass({
            label: 'Quad. init. pass', 
        });
        pass.setPipeline(pipeline);

        // dude, we can just do this for the patch-specific pipelines/passes 
        // initModules = {
        //      quad: quadInitShaderModule,
        //      tri : triInitShaderModule,
        //      PNQuad: PNQuadInitShaderModule,
        //      ...
        // }
        //
        // initPipelines = {
        //      quad: ...
        //      ... (you get the idea)
        // }
        //
        // from there, we just need a nice way to actually change these.
        // i'm thinking of just using the same approach that we currently have,
        // i.e. we have a function defined on the patch objects that will
        // fill in all this STUFF for us. that should work fine, and would
        // help to break apart all this information. the only thing, though,
        // is...
        //
        // well, no, i was going to say "oh, but these WebGPU objects are all one-time use,
        // so that means we won't have a good way to "
        //

        for (const [i, chunk] of this.chunks.entries()) {
            // Create the position buffer for the entire chunk
            const positionBuffer = new THREE.StorageBufferAttribute(
                chunk.byteSize / (4 * 4),
                4,
            );
            chunk.mesh.geometry.setAttribute("position", positionBuffer);
            this.renderer.backend.createStorageAttribute(chunk.mesh.geometry.getAttribute("position"));

            // TODO: remove this?
            chunk.positionStorage = storage(
                chunk.mesh.geometry.getAttribute("position"),
                "vec4",
                chunk.mesh.geometry.getAttribute("position").count,
            );

            // ====================
            // Create the index buffer for the entire chunk
            // TODO: precalc. the size of this (and just do this more efficiently)
            // (precalc size of array buffer and then create a single Uint32Array TypedArray,
            //  with each patch writing into that array at a given offset)
            // ====================

            const indices = [];
            for (const patch of chunk.patches) {
                indices.push(...patch.getIndexBuffer(0));
            }
            const typedIndices = new THREE.Uint32BufferAttribute(indices, 1);

            chunk.mesh.geometry.setIndex(typedIndices);
            chunk.mesh.geometry.index.version++;
            chunk.indexBuffers[0] = typedIndices;

            // ====================
            // Run the init compute fns. to copy a bunch of stuff to the vertex buffer
            // TODO: this only works on quads! fix that!
            // ====================

            // The GPUBuffer that points to this chunk's vertex buffer
            const chunkGPUBuffer = this.renderer.backend.get(
                chunk.mesh.geometry.getAttribute('position')
            ).buffer;

            for (const [j, patch] of chunk.patches.entries()) {
                // The GPUBuffer that points to this patch's control point buffer
                const patchGPUBuffer = this.renderer.backend.get(patch.cpBuffer).buffer;

                // index_offset(u32) + u_deg(u32) + v_deg(u32)
                const uniformBufferSize = (4) + (4) + (4);
                const uniformBuffer = this.device.createBuffer({
                    label: `Uniforms for (chunk, patch): (${i}, ${j})`,
                    size: uniformBufferSize,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                const uniformValues = new Uint32Array([patch.indexOffset, patch.uDeg, patch.vDeg]);
                this.device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

                const bindGroup = this.device.createBindGroup({
                    label: `Bind group for (chunk, patch): (${i}, ${j})`,
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        {binding: 0, resource: chunkGPUBuffer },
                        {binding: 1, resource: patchGPUBuffer },
                        {binding: 2, resource: uniformBuffer },
                    ]
                });

                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(1);
            }
        }
        pass.end();

        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);


        // All patches in all chunks are now initialized.

        // Vertex shader to adjust all points by their rational value
        // TODO: swap for WGSL?
        const rationalVS = Fn(({ vertexBuffer}) => {
            const w = float(vertexBuffer.element(vertexIndex).w);

            // TODO: should we use another value than 1?
            If (equal(w, 0), () => {
                w.assign(1);
            });

            return div(positionGeometry, w);
        });

        // Assign a material and a vertex shader to all meshes
        for (const chunk of this.chunks) {

            const material = new THREE.MeshStandardNodeMaterial({
                color: 0x60a0ff,
                wireframe: false,
            });
            const wireframeMaterial = new THREE.MeshBasicNodeMaterial({
                color: 0x000000,
                wireframe: true,
            });

            material.positionNode = rationalVS({
                vertexBuffer: chunk.positionStorage,
            });
            wireframeMaterial.positionNode = rationalVS({
                vertexBuffer: chunk.positionStorage,
            });

            chunk.material = material;
            chunk.wireframeMaterial = wireframeMaterial;

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
            const mode = '';
            if (mode === 'TIMING') {
                const oldHighestLevel = this.highestComputedLevel;
                console.time("setLevel");
                this.renderer
                    .resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE)
                    .then(() => {
                        console.time("ThreeJS rsrc. creation");
                        this.renderer.computeAsync(computeNodes).then(() => {
                            console.timeEnd("setLevel")

                            this.renderer
                                .resolveTimestampsAsync(
                                    THREE.TimestampQuery.COMPUTE,
                                )
                                .then((ms) => {
                                    console.log(
                                        `Compute: ${oldHighestLevel} -> ${newLevel} took ${ms.toFixed(3)}ms on the GPU`,
                                    );
                                });
                        });
                        console.timeEnd("ThreeJS rsrc. creation");
                    });
            } else {
                this.renderer.compute(computeNodes);
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
}

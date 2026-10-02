import {FileLoader, Float32BufferAttribute, Group, Loader} from 'three/webgpu';
import {BvQuad} from './BvQuad.js';
import {BvTri} from './BvTri.js';
import {BvModel} from './BvModel.js';

/**
 * A loader for the Bv format.
 *
 * The [Bv format]{@link https://www.cise.ufl.edu/research/SurfLab/bview/#file-format} is a simple data-format that
 * represents 3D beizer patches.
 *
 * ```js
 * const loader = new BvLoader();
 * const object = await loader.loadAsync( 'models/monster.obj' );
 * scene.add( object );
 * ```
 *
 * @augments Loader
 * @three_import import { BvLoader } from 'three/addons/loaders/BvLoader.js';
 */
class BvLoader extends Loader {

    /**
     * Constructs a new Bv loader.
     *
     * @param {LoadingManager} [manager] - The loading manager.
     * @param {WebGPURenderer} [renderer] - The renderer for the scene.
     */
    // BvPatch objects need a handle to the renderer, and so this loader does too to create the BvPatch objects
    constructor( manager, renderer ) {

        super( manager );

        this.renderer = renderer;
        this.materials = null;

    }

    /**
     * Starts loading from the given URL and passes the loaded OBJ asset
     * to the `onLoad()` callback.
     *
     * @param {string} url - The path/URL of the file to be loaded. This can also be a data URI.
     * @param {function(Group)} onLoad - Executed when the loading process has been finished.
     * @param {onProgressCallback} onProgress - Executed while the loading is in progress.
     * @param {onErrorCallback} onError - Executed when errors occur.
     */
    load( url, onLoad, onProgress, onError ) {

        const scope = this;

        const loader = new FileLoader( this.manager );
        loader.setPath( this.path );
        loader.setRequestHeader( this.requestHeader );
        loader.setWithCredentials( this.withCredentials );
        loader.load( url, function ( text ) {

            try {

                onLoad( scope.parse( text ) );

            } catch ( e ) {

                if ( onError ) {

                    onError( e );

                } else {

                    console.error( e );

                }

                scope.manager.itemError( url );

            }

        }, onProgress, onError );

    }

    /**
     * Sets the material creator for this OBJ. This object is loaded via {@link MTLLoader}.
     *
     * @param {MaterialCreator} materials - An object that creates the materials for this OBJ.
     * @return {BvLoader} A reference to this loader.
     */
    setMaterials( materials ) {

        this.materials = materials;

        return this;

    }

    /**
     * Parses the given Bv data and returns the resulting group.
     *
     * @param {string} text - The raw Bv data as a string.
     * @return {Group} The parsed Bv.
     */
    parse(text) {
        console.time('text parsing');

        const groups = [];
        const unnamed = {
            name: "(unnamed group)",
            patches: []
        }
        groups.push(unnamed);
        let currentGroup = unnamed;

        const input = text.split(/\s+/);
        let idx = 0;

        const allQuads = [];
        const allTris  = [];
        const allPolys = [];

        // TODO: can easily rewrite this to just use a Float32 array since we literally
        // just need to size it to (numPoints * 4)

        // utility fn to read (x, y, z) / (x, y, z, w) points from input
        // note that this DOES modify 'idx'
        function readPoints(numPoints, isRational) {
            let points = new Float32Array(numPoints * 4);
            let p_idx = 0;

            if (isRational) {
                // if the patch is rational, we need to read the 4th 'W' value
                for (let i = 0; i < numPoints; i++) {
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = Number(input[idx++]);
                }
            } else {
                for (let i = 0; i < numPoints; i++) {
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = Number(input[idx++]);
                    points[p_idx++] = 1.0;
                }
            }
            return points;
        }

        // TODO: for storing control point buffers, it would be easy to e.g. create one array
        // upfront for all control points, then while parsing simply record the offset in that array,
        // instead of creating new CP buffers per-patch. then, also, to copy the data over
        // we could either keep asking the GPU to just copy a range, or we could do the whole
        // thing at once.
        //
        // only difficulty is of not necessarily knowing the size of the buffer we would need
        // without first scanning the full input; maybe it would be fine to do a preliminary scan,
        // and then just change our methodology throughout to use this additional info much better.
        while (idx < input.length && input[idx] !== '') {
            if (input[idx].toLowerCase() === "group") {
                // ====================
                // Parse a group
                // ====================

                // Need to consume "group" token
                idx++;
                const groupID = Number(input[idx++]); // TODO: when does this actually get used?
                const groupName = input[idx++];

                // Check if we've already seen this group while parsing;
                // if we haven't, create it
                const groupAlreadyExists = groups.find(g => g.name === groupName);
                if (groupAlreadyExists) {
                    // console.log(`preexisting group: ${groupName}`);
                    currentGroup = groupAlreadyExists;
                } else {
                    // console.log(`new group: ${groupName}`);
                    const newGroup = {
                        name: groupName,
                        patches: [],
                        nrQuads: 0,
                        nrTris: 0,
                    }
                    groups.push(newGroup);
                    currentGroup = newGroup;
                }
            } else {
                // ====================
                // Parse a patch
                // ====================
                const patchType = Number(input[idx++]);
                switch (patchType) {

                    // ====================
                    // Tri or rational tri
                    // ====================
                    case 3:
                    case 11: {
                        const isRational = patchType === 11;
                        const deg = Number(input[idx++]);
                        const numControlPoints = ((deg+2) * (deg+1)) >> 1;
                        // Order the control points like a lower triangular matrix, i.e.:
                        // 0
                        // 1 2
                        // 3 4 5
                        const cpBufferCPU = new Float32Array(numControlPoints * 4);
                        for (let row = deg; row >= 0; row--) {
                            for (let col = 0; col < row + 1; col++) {
                                const bufIdx = ((((row)*(row+1)) >> 1) + col) * 4;
                                cpBufferCPU[bufIdx]   = Number(input[idx++]);
                                cpBufferCPU[bufIdx+1] = Number(input[idx++]);
                                cpBufferCPU[bufIdx+2] = Number(input[idx++]);

                                if (isRational) {
                                    cpBufferCPU[bufIdx+3] = Number(input[idx++]);
                                } else {
                                    cpBufferCPU[bufIdx+3] = 1;
                                }
                            }
                        }

                        const cpSizeInBytes = numControlPoints * 4 * 4;
                        const cpBuffer = this.renderer.backend.device.createBuffer({
                            label: `cpBuffer for tri (deg=${deg})`,
                            size : cpSizeInBytes,
                            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                        });
                        this.renderer.backend.device.queue.writeBuffer(
                            cpBuffer,
                            0,
                            cpBufferCPU
                        );

                        const newTri = new BvTri(patchType, deg, cpBuffer)
                        currentGroup.patches.push(newTri);
                        allTris.push(newTri);

                        break;
                    }

                    // ====================
                    // Quad or rational quad
                    // ====================
                    case 4:
                    case 5:
                    case 8: {
                        const isRational = patchType === 8;
                        let uDeg, vDeg;
                        if (patchType === 4) {
                            uDeg = vDeg = Number(input[idx++]);
                        } else {
                            uDeg = Number(input[idx++]);
                            vDeg = Number(input[idx++]);
                        }

                        //////////
                        // Create storage buffer for control points
                        //////////
                        const numControlPoints = (uDeg + 1) * (vDeg + 1);
                        const cpBufferCPU = readPoints(numControlPoints, isRational);

                        const cpBuffer = this.renderer.backend.device.createBuffer({
                            label: `cpBuffer for quad (u=${uDeg}, v=${vDeg})`,
                            size : cpBufferCPU.byteLength,
                            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                        });
                        this.renderer.backend.device.queue.writeBuffer(
                            cpBuffer,
                            0,
                            cpBufferCPU
                        );

                        //////////
                        // Create uniform buffer for u / v deg
                        //////////
                        const uniformValues = new Uint32Array([
                            uDeg,
                            vDeg,
                        ]);
                        const uniformBuffer = this.renderer.backend.device.createBuffer({
                            label: `Patch info uniform (${uDeg}, ${vDeg})`,
                            size : uniformValues.byteLength,
                            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                        });
                        this.renderer.backend.device.queue.writeBuffer(
                            uniformBuffer,
                            0,
                            uniformValues
                        );

                        const newQuad = new BvQuad(patchType, uDeg, vDeg, cpBuffer, uniformBuffer);

                        allQuads.push(newQuad);
                        currentGroup.patches.push(newQuad);
                        currentGroup.nrQuads++;

                        break;
                    }



                    // TODO: POLY
                    case 1: {
                        throw new Error("TODO");
                    }
                    // TODO: PN TRI
                    case 9: {
                        throw new Error("TODO");
                    }
                    // TODO: PN QUAD
                    case 10: {
                        throw new Error("TODO");
                    }
                    default: {
                        throw new Error("Unsupported patch type");
                    }
                }
            }
        }

        // Remove the default unnamed group if it wasn't used (i.e., every patch
        // in the model was part of some named group)
        if (unnamed.patches.length === 0) {
            groups.splice(0, 1);
        }

        // Sort in descending order of patch degree
        allQuads.sort((a, b) => (b.uDeg * b.vDeg) - (a.uDeg * a.vDeg));

        console.timeEnd('text parsing');

        const totalNumPatches = allQuads.length + allTris.length + allPolys.length;
        console.log(`${totalNumPatches} patches in the model.`);

        const bvModel = new BvModel(allQuads, allTris, allPolys, groups);
        return bvModel;
    }
}

export { BvLoader };

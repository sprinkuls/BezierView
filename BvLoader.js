import {FileLoader, Float32BufferAttribute, Group, Loader} from 'three/webgpu';
import {BvQuad} from './BvQuad.js';
import {BvTri} from './BvTri.js';
import {BvGroup} from './BvGroup.js';

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
        const groups = [];
        const unnamed = {
            name: "(unnamed group)",
            patches: []
        }
        groups.push(unnamed);
        let currentGroup = unnamed;

        const input = text.split(/\s+/);
        let idx = 0;

        // utility fn to read (x, y, z) / (x, y, z, w) points from the file
        function readPoints(numPoints, isRational) {
            let points = [];
            // if the patch is rational, we need to read a 4th value
            if (isRational) {
                for (let i = 0; i < numPoints; i++) {
                    points.push(Number(input[idx++]));
                    points.push(Number(input[idx++]));
                    points.push(Number(input[idx++]));
                    points.push(Number(input[idx++]));
                }
            } else {
                for (let i = 0; i < numPoints; i++) {
                    points.push(Number(input[idx++]));
                    points.push(Number(input[idx++]));
                    points.push(Number(input[idx++]));
                    points.push(1);
                }
            }
            return points;
        }

        while (idx < input.length && input[idx] !== '') {
            if (input[idx].toLowerCase() === "group") {
                idx++; // consume group token
                const groupID = Number(input[idx++]);
                const groupName = input[idx++];
                console.log(`new group: ${groupName}`);

                // check if group with this name already exists, and if not, create it
                const foundGroup = groups.find(g => g.name === groupName);
                if (foundGroup) {
                    currentGroup = foundGroup;
                } else {
                    const newGroup = {
                        name: groupName,
                        patches: []
                    }
                    groups.push(newGroup);
                    currentGroup = newGroup;
                }
            } else {
                const patchType = Number(input[idx++]);

                switch (patchType) {
                    case 1: {
                        throw new Error("TODO");
                    }
                    case 3:
                    case 11: {
                        const isRational = patchType === 11;
                        const deg = Number(input[idx++]);
                        const numControlPoints = ((deg+2) * (deg+1)) / 2;
                        // order control points like a lower triangular matrix, i.e.:
                        // 0
                        // 1 2
                        // 3 4 5
                        const cpBufferInit = new Float32Array(numControlPoints * 4);
                        for (let row = deg; row >= 0; row--) {
                            for (let col = 0; col < row + 1; col++) {
                                const bufIdx = ((((row)*(row+1)) >> 1) + col) * 4;
                                cpBufferInit[bufIdx]   = Number(input[idx++]);
                                cpBufferInit[bufIdx+1] = Number(input[idx++]);
                                cpBufferInit[bufIdx+2] = Number(input[idx++]);

                                if (isRational) {
                                    cpBufferInit[bufIdx+3] = Number(input[idx++]);
                                } else {
                                    cpBufferInit[bufIdx+3] = 1;
                                }
                            }
                        }

                        const cpBuffer = new Float32BufferAttribute(cpBufferInit, 4);
                        currentGroup.patches.push(new BvTri(patchType, deg, cpBuffer));

                        break;
                    }
                    case 4:
                    case 5:
                    case 8: {
                        let uDeg, vDeg;
                        if (patchType === 4) {
                            uDeg = vDeg = Number(input[idx++]);
                        } else {
                            uDeg = Number(input[idx++]);
                            vDeg = Number(input[idx++]);
                        }

                        const numControlPoints = (uDeg + 1) * (vDeg + 1);

                        let cpBuffer;
                        if (patchType === 8) {
                            // we have a rational point to read
                            cpBuffer = new Float32BufferAttribute(readPoints(numControlPoints, true), 4);
                        } else {
                            cpBuffer = new Float32BufferAttribute(readPoints(numControlPoints, false), 4);
                        }
                        currentGroup.patches.push(new BvQuad(patchType, uDeg, vDeg, cpBuffer));

                        break;
                    }
                    case 9: {
                        throw new Error("TODO");
                    }
                    case 10: {
                        throw new Error("TODO");
                    }
                    default: {
                        throw new Error("Unsupported patch type");
                    }
                }
            }
        }

        // remove the default unnamed group if it went unused
        if (unnamed.patches.length === 0) {
            groups.splice(0, 1);
        }

        // now that parsing is done, create all the BvGroups in this file
        const bvFile = new Group();
        for (const group of groups) {
            // TODO
            console.log(group.name, group.patches);
            bvFile.add(new BvGroup(group.name, group.patches, this.renderer));
        }
        return bvFile;
    }
}

export { BvLoader };

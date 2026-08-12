import {FileLoader, Float32BufferAttribute, Group, Loader} from 'three/webgpu';
import {BvPatch} from './BvPatch.js';
import {BvQuad} from './BvQuad.js';
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
    parse( text ) {
        console.log("parsing file");

        // 'root' group, holds the BezierView groups that get found in parsing
        const root = new Group();

        let input = text.split(/\s+/);
        let idx = 0;
        console.log(input);

        // create a default group in case none are specified in the file
        const unnamed = new Group();
        unnamed.name = "(unnamed group)";
        root.add(unnamed);

        let currentGroup = unnamed;

        // utility fn to read xyz/xyzw points from the file
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
                // if the patch isn't rational, make every 4th value 1
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

        while (idx < input.length && input[idx] != '') {
            // check for the start of a group
            if (input[idx].toUpperCase() == "GROUP") {
                let groupID = Number(input[++idx]);
                let groupName = input[++idx];
                //console.log("group with ID", groupID, "and name", groupName);

                // if this group doesn't exist, create it
                if (!root.getObjectByName(groupName)) {
                    let newGroup = new Group();
                    newGroup.name = groupName;
                    root.add(newGroup);
                }

                // set this group as the current one
                currentGroup = root.getObjectByName(groupName);
                idx++;
            }

            // whether we just read a group label or not, should now hit the start of a patch
            let patchType = Number(input[idx++]);

            switch (patchType) {
                // tensor products
                case 4: // square (deg_u == deg_v)
                case 5: // quad (deg_u independent of deg_v)
                case 8: { // rational, 4th value to read
                    let deg_u, deg_v;
                    if (patchType == 4) {
                        deg_u = deg_u = Number(input[idx]);
                        deg_v = deg_v = Number(input[idx++]);
                    } else {
                        deg_u = Number(input[idx++]);
                        deg_v = Number(input[idx++]);
                    }
                    let numPoints = (deg_u + 1) * (deg_v + 1);
                    let controlPoints = readPoints(numPoints, patchType === 8);

                    // let myNewPatch = new BvPatch(this.renderer, controlPoints, patchType, deg_u, deg_v, 2);
                    let myNewPatch = new BvPatch(this.renderer, controlPoints, patchType, deg_u, deg_v, 0);
                    currentGroup.add(myNewPatch);

                    break;
                }
                default: {
                    throw new Error("TODO");
                }
            }
        }

        // check if 'unnamed group' went unused (ie, if all data was stored in named groups), and remove it if did
        if (root.getObjectByName("(unnamed group)").children.length === 0) {
            root.remove(root.getObjectByName("(unnamed group)"));
        }

        // set level of all meshes at once
        const init_level = 2
        const all_passes = []
        for (const group of root.children) {
            for (const patch of group.children) {
                all_passes.push(...patch.getLevelPasses(2))
            }
        }
        if (all_passes.length !== 0)
            this.renderer.compute(all_passes);

        // Create a Patch. Enable contorl mesh and normals if this.materials is null
        // Assign material based on patch group. Only do that if this.materials is null otherwise use this.materials

        return root;
    }

    parseNew(text) {
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
                    case 3: {
                        const deg = Number(input[idx++]);
                        const numControlPoints = ((deg+2) * (deg+1)) / 2;
                        const cpBuffer = new Float32BufferAttribute(readPoints(numControlPoints, false), 4);
                        currentGroup.patches.push(new BvTri(patchType, deg, cpBuffer));
                        // currentGroup.patches.push({
                        //     patchType: "TRI",
                        //     deg: deg,
                        //     cpBuffer: cpBuffer,
                        // });

                        break;
                    }
                    case 4:
                    case 5: {
                        let uDeg, vDeg;
                        if (patchType === 4) {
                            uDeg = vDeg = Number(input[idx++]);
                        } else {
                            uDeg = Number(input[idx++]);
                            vDeg = Number(input[idx++]);
                        }

                        const numControlPoints = (uDeg + 1) * (vDeg + 1);
                        const cpBuffer = new Float32BufferAttribute(readPoints(numControlPoints, false), 4);
                        currentGroup.patches.push(new BvQuad(patchType, uDeg, vDeg, cpBuffer));
                        // currentGroup.patches.push({
                        //     patchType: "QUAD",
                        //     uDeg: uDeg,
                        //     vDeg: vDeg,
                        //     cpBuffer: cpBuffer,
                        // });

                        break;
                    }
                    case 8: {
                        throw new Error("TODO");
                    }
                    case 9: {
                        throw new Error("TODO");
                    }
                    case 10: {
                        throw new Error("TODO");
                    }
                    case 11: {
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

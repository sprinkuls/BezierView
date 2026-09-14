import { GUI } from 'lil-gui';
import * as THREE from 'three/webgpu';
import {BvLoader} from './BvLoader.js';
import {rsrc} from './rsrc.js';

const gui = new GUI();
const controllers = [];

const DEFAULT_SUBDIV_LEVEL = 4;

let bvFile;
let bvLoader;
let scene_;
let device;

let groupHandle;

const state = {

    // common/global settings
    activeGroup: "All Groups",

    // File
    file: 'Logo',

    // Other
    mouseMode: "Rotate",
    position: "Position 1",

    // Display
    light1: true,
    light2: false,
    light3: false,
    boundingBox: false,
    patchDetail: 2,

    // Help
    helpDummy: 0,

    // group-specific settings
    groupSettings: {
        "All Groups": {
            // Shader Settings
            flipNormals: false,
            flatShading: false,
            // can only have one of the following three active at a time
            highlightLines: false,
            highlightReflection: false,
            wireframe: false,
            visualizeNormals: false,
            highlightDensity: 0.50,
            highlightResolution: 10.0,
            visualizeNormalSize: 1.0,

            // Curvature
            showCurvature: false,
            curvatureType: "Gaussian",
            // the following two shouldn't be editable, just display numbers
            curvatureMin: 0,
            curvatureMax: 0,
            clampMin: 0,
            clampMax: 0,

            // Display
            controlMesh: false,
            showPatches: true,

            // Groups
            groupColor: "Yellow",
        }
    },
}

function setupGui({scene, renderer, light1, light2, light3, initBvFile}) {
    device = renderer.backend.device;
    scene_ = scene;
    loadPresetBvFile(initBvFile);

    ////////////////////////////// 'Load File'
    ////////////////////////////////////////////////////
    const loadFileFolder = gui.addFolder('Load File').close();
    let fileOptions = ['Logo', 'Teapot', 'Hexoid', 'Cube, G2', 'Male Head'];
    const presetModels = loadFileFolder.add(state, 'file', fileOptions,).name('Choose Model');
    loadFileFolder.add( {func() {const fileElem = document.getElementById('fileInput');fileElem.click()}}, 'func' ).name('Load File');

    bvLoader = new BvLoader(THREE.DefaultLoadingManager, renderer);
    document.getElementById('fileInput').addEventListener('change', (event) => {
        loadUserBvFile(event);
    });

    ////////////////////////////// 'Shader Settings'
    //////////////////////////////////////////////////

    const shaderSettingsFolder = gui.addFolder('Shader Settings')
    // .close();
    const flipNormals = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'flipNormals').name('Flip Normals').onChange(propogateState('flipNormals'));
    const flatShading = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'flatShading').name('Flat Shading').onChange(propogateState('flatShading'));
    const highlightLines = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'highlightLines').name('Highlight Lines').onChange(propogateState('highlightLines'));
    const highlightReflection = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'highlightReflection').name('Highlight Reflection').onChange(propogateState('highlightReflection'));
    const wireframe = shaderSettingsFolder
        .add(state.groupSettings["All Groups"], 'wireframe')
        .name('Wireframe')
        .onChange((newValue) => {
            // TODO: this should apply BASED ON GROUP! right now, it's
            // being set for ALL groups NO MATTER WHAT!
            if (newValue === true) {
                for (const bvGroup of bvFile.children) {
                    for (const chunk of bvGroup.chunks) {
                        chunk.mesh.material = chunk.wireframeMaterial;
                    }
                }
            } else {
                for (const bvGroup of bvFile.children) {
                    for (const chunk of bvGroup.chunks) {
                        chunk.mesh.material = chunk.material;
                    }
                }
            }
            propogateState('wireframe')();
        });

    const visualizeNormals = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'visualizeNormals').name('Visualize Normals').onChange(propogateState('visualizeNormals'));


    const highlightDensity = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'highlightDensity', 0, 1, 0.05).name('Highlight Density').onChange(propogateState('highlightDensity'));
    const highlightResolution = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'highlightResolution', 0, 30, 0.1).name('Highlight Resolution').onChange(propogateState('highlightResolution'));
    const visualizeNormalSize = shaderSettingsFolder.add(state.groupSettings["All Groups"], 'visualizeNormalSize', 0.01, 10, 0.01).name('Visualize Normals Size').onChange(propogateState('visualizeNormalSize'));

    controllers.push(flipNormals);
    controllers.push(flatShading);
    controllers.push(highlightLines);
    controllers.push(highlightReflection);
    controllers.push(wireframe);
    controllers.push(visualizeNormals);
    controllers.push(highlightDensity);
    controllers.push(highlightResolution);
    controllers.push(visualizeNormalSize);



    ////////////////////////////// 'Curvature'
    //////////////////////////////////////////////////
    const curvatureFolder = gui.addFolder('Curvature').close();
    const showCurvature = curvatureFolder.add(state.groupSettings["All Groups"], 'showCurvature').name('Show Curvature').onChange(propogateState('showCurvature'));
    controllers.push(showCurvature);

    const curvatureOptions = ['Gaussian', 'Mean', 'Max', 'Min'];
    const curvatureType = curvatureFolder.add(state.groupSettings["All Groups"], 'curvatureType', curvatureOptions).name('Curvature Types').onChange(propogateState('curvatureType'));
    controllers.push(curvatureType);

    // next two disabled since they're information readouts, and thus not editable
    const curvatureMin = curvatureFolder.add(state.groupSettings["All Groups"], 'curvatureMin').name('Curvature Min').disable().onChange(propogateState('curvatureMin'));
    const curvatureMax = curvatureFolder.add(state.groupSettings["All Groups"], 'curvatureMax').name('Curvature Max').disable().onChange(propogateState('curvatureMax'));
    const clampMin = curvatureFolder.add(state.groupSettings["All Groups"], 'clampMin').name('Clamp Min').onChange(propogateState('clampMin'));
    const clampMax = curvatureFolder.add(state.groupSettings["All Groups"], 'clampMax').name('Clamp Max').onChange(propogateState('clampMax'));

    controllers.push(curvatureMin);
    controllers.push(curvatureMax);
    controllers.push(clampMin);
    controllers.push(clampMax);

    curvatureFolder.add({func() {}}, 'func').name('Reset Clamp');



    ////////////////////////////// 'Display'
    //////////////////////////////////////////////////
    // const displayFolder = gui.addFolder('Display').close();
    const displayFolder = gui.addFolder('Display');
    const controlMesh = displayFolder.add(state.groupSettings["All Groups"], 'controlMesh').name('Show Control Mesh').onChange(propogateState('controlMesh'));
    displayFolder.add(state, 'boundingBox').name('Show Bounding Box');
    const showPatches = displayFolder.add(state.groupSettings["All Groups"], 'showPatches').name('Show Patches').onChange(propogateState('showPatches'));

    controllers.push(controlMesh);
    controllers.push(showPatches);

    displayFolder.add(state, 'light1').name('Light 1').onChange((newValue) => {
        if (newValue === true) {
            light1.intensity = 5;
        } else {
            light1.intensity = 0;
        }
    });
    displayFolder.add(state, 'light2').name('Light 2').onChange((newValue) => {
        if (newValue === true) {
            light2.intensity = 5;
        } else {
            light2.intensity = 0;
        }
    });
    displayFolder.add(state, 'light3').name('Light 3').onChange((newValue) => {
        if (newValue === true) {
            light3.intensity = 5;
        } else {
            light3.intensity = 0;
        }
    });

    let detailOptions = {
        '1 × 1'  : 0,
        '2 × 2'  : 1,
        '4 × 4'  : 2,
        '8 × 8'  : 3,
        '16 × 16': 4,
        '32 × 32': 5,
        '64 × 64': 6
    };
    displayFolder.add(state, 'patchDetail', detailOptions).name('Patch Detail')
        .setValue(DEFAULT_SUBDIV_LEVEL)
        .onChange((newLevel) => setSubdivLevel(newLevel));



    ////////////////////////////// 'Groups'
    //////////////////////////////////////////////////
    const groupsFolder = gui.addFolder('Groups').close();

    groupHandle = groupsFolder.add(state, 'activeGroup', Object.keys(state.groupSettings)).onChange(changeGroup).name('Group');

    let groupColorOptions = ['Yellow', 'Green', 'Silver', 'Red', 'Cyan', 'Magenta', 'Orange', 'Purple', 'Blue'];
    const groupColor = groupsFolder.add(state.groupSettings["All Groups"], 'groupColor', groupColorOptions).name('Group Color').onChange(propogateState('groupColor'));
    controllers.push(groupColor);

    //groupsFolder.add({func() {addGroup();}}, 'func').name('Add Group');
    groupsFolder.add({func() {}}, 'func').name('Delete Group');


    // 'Other' Section
    const otherFolder = gui.addFolder('Other').close();
    otherFolder.add(state, 'mouseMode', ['Rotate', 'Clip']).name('Mouse Mode');
    otherFolder.add({func() {}}, 'func').name('Save Position');
    otherFolder.add({func() {}}, 'func').name('Remove Position');

    let positionOptions = ['Position 1', 'Position 2', 'Position 3'];
    otherFolder.add(state, 'position', positionOptions).name('Load Position');

    otherFolder.add({func() {}}, 'func').name('Reload Current');
    otherFolder.add({func() {}}, 'func').name('Reset Projection');


    ////////////////////////////// 'Help'
    //////////////////////////////////////////////////
    const helpFolder = gui.addFolder('Help').close();
    const helpText =
            "Controls\n" +
            "Left Mouse: Rotate or Clip\n" +
            "Right Mouse: Pan\n" +
            "Scroll Wheel: Zoom\n" +
            "Scroll Wheel + Alt: Scale\n" +
            "Q and E: Z-axis rotation\n" +
            "ESC: Reset Projection\n";
    helpFolder.add(state, 'helpDummy').name(helpText).$widget.remove();

    return gui;
}

// effectively, we just loop through all of the controllers (the lil-gui
// objects that hold the input values from the user)
// and say "hey, we're now changing to *this* group, so make the
// values in all the settings be what we set them to on *this* group"
function changeGroup(groupName) {
    for (const controller of controllers) {
        controller.object = state.groupSettings[groupName]
        controller.updateDisplay()
    }
}

// idea is just that, when we update certain settings for "All Groups", those changes need
// to also be propogated to the other groups in the file. so this function is basically just
// a factory that returns a function to propogate certain settings to child 
function propogateState(settingsKey){
    function callback(value) {
        if (state.activeGroup == "All Groups") {
            for(const groupName in state.groupSettings) {
                if (groupName !== "All Groups") {
                    state.groupSettings[groupName][settingsKey] = value;
                }
            }
        }
    }
    return callback;
}


/*
//////////////////// BEFORE PARSE:
// local copy of group settings, so that global state isn't changed prematurely
let groupSettingsLocal = {};

// only copy in 'All Groups' since we if we're loading a new model, we don't
// want to keep around groups from another file
groupSettingsLocal["All Groups"] = structuredClone(state.groupSettings["All Groups"]);

//////////////////// AFTER PARSE:

// split the input into whitespace-delimited tokens
// make 'All Groups' the currently set group (since the user might
// have a group selected that doesn't exist in this new model)
state.activeGroup = 'All Groups';
groupHandle.setValue('All Groups');
changeGroup('All Groups'); // necessary as setValue() doesn't trigger onChange()

// copy in the groups and update the group options
state.groupSettings = structuredClone(groupSettingsLocal);
groupHandle.options(Object.keys(state.groupSettings));
*/

function loadUserBvFile(event) {
    const file = event.target.files[0];

    if (!file) {
        console.log("Error loading file");
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        console.log("user file loaded");
        const text = event.target.result;
        addBvModel(text);
    }
    reader.readAsText(file);
}

// TODO: use this to load the other presets (e.g. g2 cube)
async function loadPresetBvFile(path) {
    const response = await fetch(path);
    const text = await response.text();
    addBvModel(text);
}

// TODO: update the global state and GUI now that a new model has been loaded
// (groups section, etc)
function addBvModel(text) {
    if (bvFile) {
        scene_.remove(bvFile);
        for (const group of bvFile.children) {
            for (const chunk of group.children) {
                chunk.geometry.dispose();
                chunk.material.dispose();
            }
        }
    }

    bvFile = bvLoader.parse(text);
    // TODO: TESTING: UNCOMMENT THIS
    // setSubdivLevel(DEFAULT_SUBDIV_LEVEL);
    scene_.add(bvFile);
}

function setSubdivLevel(newLevel) {
    // TODO: we don't need to create the encoder/pass
    // if we're going to a level we've already been at before
    const encoder = rsrc.device.createCommandEncoder({
        label: `setLevel encoder (lvl ${newLevel}) `,
    });
    const pass = encoder.beginComputePass({
        label: `setLevel pass (lvl ${newLevel}) `,
    });

    // subdiv / normals
    for (const bvGroup of bvFile.children) {
        bvGroup.setLevel(newLevel, pass);
    }

    pass.end();
    const commandBuffer = encoder.finish();

    const t0 = performance.now();
    rsrc.device.queue.submit([commandBuffer]);

    rsrc.device.queue.onSubmittedWorkDone().then(() => {
      const t1 = performance.now();
      console.log(`GPU side finished in ${(t1 - t0).toFixed(2)} ms (set level to ${newLevel})`);
    });
}

export { setupGui, state };

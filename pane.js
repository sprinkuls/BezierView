import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const GROUP_COLORS = [
    '#3aa99f', // cyan
    '#4385be', // blue
    '#8b7ec8', // purple
    '#ce5d97', // magenta
    '#d14d41', // red
    '#da702c', // orange
    '#d0a215', // yellow
    '#879a39', // green
];


export class Pane {
    element;
    canvasElement;

    canvasTarget;
    scene;
    controls;
    ambient;
    sun;

    constructor(app, model, level) {
        this.app = app;
        this.model = model;


        this.chunksByLevel = {};

        // ---------- create html skeleton ------------------------------ //
        {
            const element = (this.element = document.createElement("div"));
            element.className = "pane";

            const titlebar = document.createElement("div");
            titlebar.className = "titlebar";
            element.append(titlebar);

            const canvasContainer = (this.canvasElement = document.createElement("div"));
            canvasContainer.className = "canvas-container";
            element.append(canvasContainer);

            const sceneCanvas = document.createElement("canvas");
            canvasContainer.append(sceneCanvas);

            this.canvasTarget = new THREE.CanvasTarget(sceneCanvas);
            this.canvasTarget.setPixelRatio(window.devicePixelRatio);
            this.canvasTarget.setSize(1, 1, false);

            document.getElementById("pane-container").appendChild(element);
        }
        // ---------- create three scene and related stuff -------------------- //
        {
            this.scene = new THREE.Scene();

            this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
            // this.camera.up.set(0, 1, 0);
            // this.camera = new THREE.PerspectiveCamera( 50, 1, 1, 10 );
            this.camera.position.z = 2;

            // TODO: remove
            const gridHelper = new THREE.PolarGridHelper(2, 16);
            this.scene.add(gridHelper);

            this.controls = new OrbitControls(this.camera, this.canvasTarget.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.12;
            this.controls.addEventListener('start', () => {
                // only set this pane as the last-used one if there are no other panes
                if (!this.app.leader) {
                    this.app.leader = this;
                }
            });
            this.controls.addEventListener('change', () => {
                console.log("HELP ME!")
            });

            this.ambient = new THREE.AmbientLight(0xffffff, 1);
            this.sun = new THREE.DirectionalLight(0xffffff, 3);
            this.scene.add(this.ambient, this.sun, this.sun.target);

        }

        // ---------- actually load the model in ------------------------------ //
        // TODO: we need to loop through model.groupNames like we do above and
        // create materials of each kind for each group
        this.#addModel(level);

        // ---------- done! ------------------------------ //
    }

    async #addModel(level) {
        this.chunksByLevel[level] = await this.model.getChunksByGroup(level);
        console.log(this.chunksByLevel[level]);

        // good good good yes good....
        const regular = new THREE.MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0 });
        for (const [name, chunks] of Object.entries(this.chunksByLevel[level])) {
            // so, idea now is we just need to go group by group
            // and create a mesh for each chunk.
            for (const c of chunks) {
                c.mesh = new THREE.Mesh(c.geometry, regular);
                this.scene.add(c.mesh);
            }
        }
    }

}

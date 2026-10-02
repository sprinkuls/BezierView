struct patchInfo {
    u_deg : u32,
    v_deg : u32,
}

struct passInfo {
    nr_invocations      : u32,
    curr_computed_level : u32,
    target_level        : u32,
}

@group(0) @binding(0) var<storage, read_write> honkin_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cp_buffer : array<vec4<f32>>;

@group(0) @binding(2) var<uniform> p       : patchInfo;
@group(0) @binding(3) var<uniform> cur_pass: passInfo;



@compute @workgroup_size(64, 1, 1) fn v_pass(
    @builtin( global_invocation_id ) globalId : vec3<u32>,
) {
    let instance_index : u32 = globalId.x;

    if (instance_index > cur_pass.nr_invocations - 1) {
        return;
    }

    // "highest (subdivision) level"
    let hl : u32 = cur_pass.target_level;

    // (2 ^ hl), so for hl=6 this is 64
    let max_quads : u32 = u32(1) << hl;

    let u_deg = p.u_deg;
    let v_deg = p.v_deg;

    // (2 ^ curr_level), so at curr_level=2 this is 4 quads per dimension
    let num_quads_per_dim : u32 = u32(1) << cur_pass.curr_computed_level;

    let stride : u32 = u32(1) << (hl - cur_pass.curr_computed_level);
    let halfstride : u32 = stride >> 1;

    // these lines are equivalent to division and modulo by a power of 2:
    // let grid_x : u32 = instance_index / num_quads_per_dim;
    // let grid_y : u32 = instance_index % num_quads_per_dim;
    let grid_x : u32 = instance_index >> cur_pass.curr_computed_level;
    let grid_y : u32 = instance_index & ((num_quads_per_dim) - 1);

    let x = grid_x * stride;
    let y = grid_y * stride * v_deg;

    // "buffer width", how many indices you need to move to increase the x coordinate by 1
    let bw : u32 = (v_deg * max_quads) + 1;

    // we're just multiplying the x coordinate by the buffer width (since the
    // data is stored in row-major order), then adding the y coordinate.
    //
    // this is hoisted out because it gets used thrice every loop,
    // and doesn't change for the entire run of this shader.
    //let base_index : u32 = p.index_offset + (bw * x);
    let base_index : u32 = (bw * x) + y;

    // when this is the first level of subdiv to run, we need to read the control points
    // directly from the CP buffer, since there's nothing in the scratch buffer right now
    if (cur_pass.curr_computed_level == 0) {
        for (var j : u32 = 0; j < (v_deg + 1); j++) {
            let src : u32 = ((v_deg + 1) * instance_index) + j;
            let dst : u32 = base_index + (stride * j);

            honkin_buffer[dst] = cp_buffer[src];
        }
    }

    for (var i : u32 = 0; i < v_deg; i++) {
        for (var j : u32 = i; j < v_deg; j++) {
            let y_temp = (halfstride * i) + (stride * (j-i));

            let p1  = base_index + y_temp;
            let mid = base_index + y_temp + halfstride;
            let p2  = base_index + y_temp + stride;

            honkin_buffer[mid] = (honkin_buffer[p1] + honkin_buffer[p2]) * 0.5;
        }
    }
}

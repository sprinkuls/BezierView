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

@group(0) @binding(1) var<uniform> p       : patchInfo;
@group(0) @binding(2) var<uniform> cur_pass: passInfo;



@compute @workgroup_size(64, 1, 1) fn u_pass(
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

    // NOTE: the mod/div are swapped from v_pass!
    // let grid_x : u32 = instance_index % num_quads_per_dim;
    // let grid_y : u32 = instance_index / num_quads_per_dim;
    let grid_x : u32 = instance_index & ((num_quads_per_dim) - 1);
    let grid_y : u32 = instance_index >> cur_pass.curr_computed_level;

    let x = grid_x * stride * u_deg;
    let y = grid_y * halfstride;

    let bw : u32 = (v_deg * max_quads) + 1;

    // since the data is stored in row-major order, and the row being accessed changes
    // during runs of the inner loop, we can't pre-calculate the index as much as
    // we can in the v shader
    let base_index : u32 = (x * bw) + y;

    for (var i : u32 = 0; i < u_deg; i++) {
        for (var j : u32 = i; j < u_deg; j++) {
            let x_temp : u32 = ((halfstride * i) + (stride * (j-i)));

            let p1  = base_index + (bw * (x_temp));
            let mid = base_index + (bw * (x_temp + halfstride));
            let p2  = base_index + (bw * (x_temp + stride));

            honkin_buffer[mid] = (honkin_buffer[p1] + honkin_buffer[p2]) * 0.5;
        }
    }
}

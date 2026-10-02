struct patchInfo {
    u_deg : u32,
    v_deg : u32,
}

struct passInfo {
    nr_invocations      : u32,
    curr_computed_level : u32,
    target_level        : u32,
    pnc_index_offset    : u32,
}

@group(0) @binding(0) var<storage, read> honkin_buffer: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> position_buffer : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> normal_buffer   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> curvature_buffer: array<vec4<f32>>;

@group(0) @binding(4) var<uniform> p       : patchInfo;
@group(0) @binding(5) var<uniform> cur_pass: passInfo;

@group(0) @binding(6) var<storage, read> cp_buffer : array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1) fn evaluate(
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

    // "please do not dead-code eliminate my stuff while i am testing"
    // let TODO = honkin_buffer[0] +
    //     position_buffer[0] +
    //     normal_buffer[0] +
    //     curvature_buffer[0] +
    //     cp_buffer[0];
    curvature_buffer[0] = vec4(0.0);

    let u_deg = p.u_deg;
    let v_deg = p.v_deg;

    // ((2 ^ level) + 1)
    // so at level 2, this is (2^2)+1 = 5 vertices per dimension
    let num_verts_per_dim : u32 = (u32(1) << cur_pass.curr_computed_level) + 1;

    let stride : u32 = u32(1) << (hl - cur_pass.curr_computed_level);

    let grid_x : u32 = instance_index / num_verts_per_dim ;
    let grid_y : u32 = instance_index % num_verts_per_dim ; 

    // "buffer width", how many indices you need to move to increase the x coordinate by 1
    let bw : u32 = (v_deg * max_quads) + 1;


    let x = grid_x * stride * u_deg * bw;
    let y = grid_y * stride * v_deg;
    let base_index = x + y;

    // these are specifically for when we want to write to P/N/C (as name suggests), since
    // unlike the points in the scratch buffer, these don't depend on u / v degree
    let pnc_bw : u32 =  (u32(1) << cur_pass.curr_computed_level) + 1;
    let pnc_offset = cur_pass.pnc_index_offset;
    let pnc_dst = (grid_x * pnc_bw) + grid_y + pnc_offset;

    // write to position_buffer
    if (cur_pass.target_level == 0) {
        let src = (grid_x * (u_deg*(v_deg+1))) + (grid_y * v_deg);
        position_buffer[pnc_offset + instance_index] = cp_buffer[src];
    } else {
        position_buffer[pnc_dst] = honkin_buffer[base_index];
    }

   var v00 : vec3<f32>;
   var v01 : vec3<f32>;
   var v10 : vec3<f32>;
   var v11 : vec3<f32>;

   if (grid_x == num_verts_per_dim - 1 && grid_y == num_verts_per_dim - 1) {
       // "top right" corner

       // indices for the points we will use for normals
       let v00_index = base_index;
       let v01_index = base_index - (1 * stride);
       let v10_index = base_index - (1 * stride * bw);
       let v11_index = base_index - (1 * stride * bw) - (1 * stride);

       // normalize rational point
       v00 = honkin_buffer[v00_index].xyz * (1 / honkin_buffer[v00_index].w);
       v01 = honkin_buffer[v01_index].xyz * (1 / honkin_buffer[v01_index].w);
       v10 = honkin_buffer[v10_index].xyz * (1 / honkin_buffer[v10_index].w);
       v11 = honkin_buffer[v11_index].xyz * (1 / honkin_buffer[v11_index].w);

   } else if (grid_x == num_verts_per_dim - 1) {
       // "right" edge

       // indices for the points we will use for normals
       let v00_index = base_index;
       let v01_index = base_index - (1 * stride * bw);
       let v10_index = base_index + (1 * stride);
       let v11_index = base_index - (1 * stride * bw) + (1 * stride);

       // normalize rational point
       v00 = honkin_buffer[v00_index].xyz * (1 / honkin_buffer[v00_index].w);
       v01 = honkin_buffer[v01_index].xyz * (1 / honkin_buffer[v01_index].w);
       v10 = (honkin_buffer[v10_index].xyz * (1 / honkin_buffer[v10_index].w));
       v11 = (honkin_buffer[v11_index].xyz * (1 / honkin_buffer[v11_index].w));


   } else if (grid_y == num_verts_per_dim - 1) {
       // "top" edge

       // indices for the points we will use for normals
       let v00_index = base_index;
       let v01_index = base_index + (1 * stride * bw);
       let v10_index = base_index - (1 * stride);
       let v11_index = base_index + (1 * stride * bw) - (1 * stride);

       // normalize rational point
       v00 = honkin_buffer[v00_index].xyz * (1 / honkin_buffer[v00_index].w);
       v01 = honkin_buffer[v01_index].xyz * (1 / honkin_buffer[v01_index].w);
       v10 = honkin_buffer[v10_index].xyz * (1 / honkin_buffer[v10_index].w);
       v11 = honkin_buffer[v11_index].xyz * (1 / honkin_buffer[v11_index].w);

   } else {
       // "bottom left" corner; what gets used for most of the calculations

       // indices for the points we will use for normals
       let v00_index = base_index;
       let v01_index = base_index + (1 * stride); // +1 y
       let v10_index = base_index + (1 * stride * bw); // +1 x
       let v11_index = base_index + (1 * stride * bw) + (1 * stride); // +1 x, +1 y

       // normalize rational point
       v00 = honkin_buffer[v00_index].xyz * (1 / honkin_buffer[v00_index].w);
       v01 = honkin_buffer[v01_index].xyz * (1 / honkin_buffer[v01_index].w);
       v10 = honkin_buffer[v10_index].xyz * (1 / honkin_buffer[v10_index].w);
       v11 = honkin_buffer[v11_index].xyz * (1 / honkin_buffer[v11_index].w);

   }

   var r_v : vec3<f32> = v01 - v00;
   if (length(r_v) < 1e-6) {
       r_v = v11 - v00;
   }

   var r_u : vec3<f32> = v10 - v00;
   if (length(r_u) < 1e-6) {
       r_u = v11 - v00;
   }

   let normal = normalize(cross(r_u, r_v));

   normal_buffer[pnc_dst] = vec4<f32>(normal, 1);
}

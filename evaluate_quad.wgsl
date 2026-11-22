// helper for normals
fn tangentDir(p: vec4f, d: vec4f) -> vec3f {
    return d.xyz * p.w - p.xyz * d.w;
}

// returns (gaussian, mean, max, min).
fn surfaceCurvature(
    P  : vec4f,
    ru : vec4f,
    rv : vec4f,
    ruu: vec4f,
    ruv: vec4f,
    rvv: vec4f,
) -> vec4f {
    // second fundamental form (unnormalized)
    let L = determinant(mat4x4f(ruu, ru, rv, P));
    let N = determinant(mat4x4f(rvv, ru, rv, P));
    let M = determinant(mat4x4f(ruv, ru, rv, P));

    // first fundamental form (unnormalized)
    let w = P.w;
    let a = ru.xyz * w - P.xyz * ru.w;
    let b = rv.xyz * w - P.xyz * rv.w;
    let E = dot(a, a);
    let G = dot(b, b);
    let F = dot(a, b);

    // squared length of the (unnormalized) normal
    let kesv = vec3f(
        determinant(mat3x3f(P.yzw, ru.yzw, rv.yzw)),
        determinant(mat3x3f(P.xzw, ru.xzw, rv.xzw)),
        determinant(mat3x3f(P.xyw, ru.xyw, rv.xyw)),
    );
    let kes = dot(kesv, kesv);

    var K = 0.0; // Gaussian
    var H = 0.0; // mean
    const MIN_SIN2_ANGLE: f32 = 1e-10;
    if (w * w * kes > MIN_SIN2_ANGLE * E * G) {
        let w2 = w * w;
        K = ((w2 * w2 * (L * N - M * M)) / kes) / kes;
        H = -((-0.5 * w * (L * G - 2.0 * M * F + N * E)) / kes) / sqrt(kes);
    }

    let disc = sqrt(max(H * H - K, 0.0));
    return vec4f(K, H, H + disc, H - disc);
}

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

@group(0) @binding(1) var<storage, read_write> position_buffer : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> normal_buffer   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> curvature_buffer: array<vec4<f32>>;

@group(0) @binding(4) var<uniform> p       : patchInfo;
@group(0) @binding(5) var<uniform> cur_pass: passInfo;

@group(0) @binding(6) var<storage, read> cp_buffer : array<vec4<f32>>;

// TODO: maybe just compile a string-templated version of this code with exactly the
// degree of patches that a model is actually using (as in, after parsing the model,
// compile a shader module with MAX_DEG set to the greater of u/v degree, so we don't
// allocate so much space)
// could also easily, on startup, compile a few for common sizes
// e.g. MAX_DEG = 3 for bicubic, or MAX_DEG = 6 for bi-6, that kind of thing.
// i mean, at a certain point, if you're passing in e.g. degree 20 patches there's
// probably no saving you.
const MAX_DEG: u32 = 7u;
const MAX_PTS: u32 = MAX_DEG + 1u;

struct CurveJet {
    p:  vec4f,
    d1: vec4f,
    d2: vec4f,
}

struct PatchJet {
    p:   vec4f,
    pu:  vec4f,
    pv:  vec4f,
    puu: vec4f,
    puv: vec4f,
    pvv: vec4f,
}

// standard 2D de Casteljau on (deg+1) points with parameter t
fn decasteljauJet(pts: array<vec4f, MAX_PTS>, deg: u32, t: f32) -> CurveJet {
    var b = pts;
    let n = f32(deg);
    var jet: CurveJet; // zero-initialized, so d2 = 0 when deg < 2

    for (var count = deg + 1u; count > 1u; count--) {
        if (count == 3u) {
            jet.d2 = (n * (n - 1.0)) * (b[2] - 2.0 * b[1] + b[0]);
        }
        if (count == 2u) {
            jet.d1 = n * (b[1] - b[0]);
        }
        for (var i = 0u; i + 1u < count; i++) {
            b[i] = mix(b[i], b[i + 1u], t);
        }
    }
    jet.p = b[0];
    return jet;
}

fn evalPatchJet(u: f32, v: f32, u_deg: u32, v_deg: u32) -> PatchJet {
    // idea here is that it takes fewer significant digits to store each control point
    // as an offset from c0, compared to storing each control point as an offset from
    // the origin
    // TODO: should transform the control points ahead of time, that way every compute
    //  shader working on a patch doesn't have to do this same set of calculations
    // TODO: centroid of patch instead of c0?
    let c0 = cp_buffer[0];
    let origin : vec3f = c0.xyz / c0.w;

    // each v-direction "row" of the patch evaluated with parameter v
    var rowP:   array<vec4f, MAX_PTS>;
    var rowPv:  array<vec4f, MAX_PTS>; // first v-derivatives
    var rowPvv: array<vec4f, MAX_PTS>; // second v-derivatives

    for (var i = 0u; i <= u_deg; i++) {
        var row: array<vec4f, MAX_PTS>;
        for (var j = 0u; j <= v_deg; j++) {
            // cp_buffer has rows of (v_deg + 1) points
            let q = cp_buffer[i * (v_deg + 1u) + j];

            row[j] = vec4f(q.xyz - (q.w * origin), q.w);
        }

        let rj = decasteljauJet(row, v_deg, v);
        rowP[i]   = rj.p;
        rowPv[i]  = rj.d1;
        rowPvv[i] = rj.d2;
    }

    // so in total, we actually get the coordinate-degree 2-jet,
    // but we only need the total-degree 2-jet for finding the curvature
    // (like, the final entry of c is "Puuvv" (if you can stand the name))
    let a = decasteljauJet(rowP, u_deg, u);   // P,  Pu, Puu
    let b = decasteljauJet(rowPv, u_deg, u);  // Pv, Puv
    let c = decasteljauJet(rowPvv, u_deg, u); // Pvv
    return PatchJet(a.p, a.d1, b.p, a.d2, b.d1, c.p);
}

@compute @workgroup_size(64, 1, 1) fn evaluate(
    @builtin( global_invocation_id ) globalId : vec3<u32>,
) {
    let instance_index : u32 = globalId.x;
    if (instance_index > cur_pass.nr_invocations - 1) {
        return;
    }

    let dst = instance_index + cur_pass.pnc_index_offset;

    // the idea is this:
    // convert instance_index to (row, col).
    // then we divide the row/col to get, respectively, the u/v location in the patch

    // ((2^level) + 1), so at level 2, this is ((2^2) + 1) = 5 vertices per dimension
    let num_verts: u32 = (1u << cur_pass.curr_computed_level) + 1;
    let grid_x   : u32 = instance_index / num_verts;
    let grid_y   : u32 = instance_index % num_verts;

    let num_segments : f32 = f32(num_verts) - 1;
    let u_fractional : f32 = f32(grid_x) / num_segments;
    let v_fractional : f32 = f32(grid_y) / num_segments;

    let jet : PatchJet = evalPatchJet(
        u_fractional, v_fractional,
        p.u_deg,      p.v_deg
    );

    // Curvature
    curvature_buffer[dst] = surfaceCurvature(
        jet.p,
        jet.pv,  jet.pu,
        jet.pvv, jet.puv, jet.puu
    );

    // Position
    let c0 = cp_buffer[0];
    let origin = c0.xyz / c0.w;
    position_buffer[dst] = vec4f(jet.p.xyz + jet.p.w * origin, jet.p.w);

    // Normals
    var tu  = tangentDir(jet.p, jet.pu);
    var tv  = tangentDir(jet.p, jet.pv);
    let tuv = tangentDir(jet.p, jet.puv);
    // fall back to tuv if we have a degenerate tu or tv
    let eps = 1e-8 * max(dot(tu, tu), dot(tv, tv));
    if (dot(tv, tv) <= eps) {
        tv = select(-tuv, tuv, u_fractional < 0.5);
    }
    if (dot(tu, tu) <= eps) {
        tu = select(-tuv, tuv, v_fractional < 0.5);
    }
    normal_buffer[dst] = vec4f(normalize(cross(tu, tv)), 1.0);
}

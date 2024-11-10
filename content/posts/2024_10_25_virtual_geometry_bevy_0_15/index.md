+++
title = "Virtual Geometry in Bevy 0.15"
date = "2024-10-25"

[taxonomies]
tags = ["bevy", "virtual geometry"]
+++

## Introduction

It's been a little over 4 months [since my last post](@/posts/2024_06_09_virtual_geometry_bevy_0_14/index.md) when I talked about the very early prototype of virtual geometry I wrote for Bevy 0.14.

While it's still not production ready, the version of virtual geometry that will ship in Bevy 0.15 (which is releasing soon) is a very large step in the right direction!

In this blog post I'll be going over all the PRs merged since my last post, in chronological order. At the end, I'll do a performance comparison of Bevy 0.15 vs 0.14, and finally discuss my roadmap for what I'm planning to work on in Bevy 0.16 and beyond.

Like last time, a lot of the larger architectural changes are copied from Nanite based on the SIGGRAPH presentation, which you should watch if you want to learn more.

It's going to be another super long read, so grab some snacks and strap in!

## Arseny Kapoulkine's Contributions
PRs [#13904](https://github.com/bevyengine/bevy/pull/13904), [#13913](https://github.com/bevyengine/bevy/pull/13913), and [#14038](https://github.com/bevyengine/bevy/pull/14038) improve the performance of the Mesh to MeshletMesh converter, and makes it more deterministic. These were written by Arseny Kapoulkine (author of meshoptimizer, the library I use for mesh simplification and meshlet building). Thanks for the contributions!

PR [#14042](https://github.com/bevyengine/bevy/pull/14042), also by Arseny Kapoulkine, fixed a bug with how we calculate the depth pyramid mip level to sample at for occlusion culling.

These PRs were actually shipped in Bevy 0.14, but were opened after I published my last post, hence why I'm covering them now.

## Faster MeshletMesh Loading
PR [#14193](https://github.com/bevyengine/bevy/pull/14193) improves performance when loading MeshletMesh assets from disk. This also shipped in Bevy 0.14, but was written after my last post.

Previously I was using the `bincode` and `serde` crates to serialize and deserialize MeshletMeshes. All I had to do was slap `#[derive(Serialize, Deserialize)]` on the type, and then I could use `bincode::serialize_into()` to turn my asset into a slice of bytes for writing to disk, and `bincode::deserialize_from()` in order to turn a slice of bytes loaded from disk back into my asset type. Easy.

Unfortunately, that ease of use came with a good bit of performance overhead. Specifically in the deserializing step, where bytes get turned into the asset type. Deserializing the 5mb Stanford Bunny asset I was using for testing took a depressingly long 77ms on my Ryzen 5 2600 CPU.

Thinking about the code flow more, we _already_ have an asset -> bytes step. After the asset is loaded into CPU memory, we serialize it _back_ into bytes so that we can upload it to GPU memory. For this, we use the `bytemuck` crate which provides functions for casting slices of data that are `Pod` (plain-old-data, i.e. just numbers, which all of our asset data is) to slices of bytes, without any real overhead.

Why not simply use bytemuck to cast our asset data to slices of bytes, and write that? Similarly for reading from disk, we can simply cast the slice of bytes back to our asset type.

```rust
fn write_slice<T: Pod>(
    field: &[T],
    writer: &mut dyn Write,
) -> Result<(), MeshletMeshSaveOrLoadError> {
    writer.write_all(&(field.len() as u64).to_le_bytes())?;
    writer.write_all(bytemuck::cast_slice(field))?;
    Ok(())
}

fn read_slice<T: Pod>(reader: &mut dyn Read) -> Result<Arc<[T]>, std::io::Error> {
    let len = read_u64(reader)? as usize;

    let mut data: Arc<[T]> = std::iter::repeat_with(T::zeroed).take(len).collect();
    let slice = Arc::get_mut(&mut data).unwrap();
    reader.read_exact(bytemuck::cast_slice_mut(slice))?;

    Ok(data)
}
```

These two functions are all we need to read and write asset data. `write_slice()` takes a slice of asset data, writes the length of the slice, and then casts the slice to bytes and writes it to disk. `read_slice()` reads the length of the slice from disk, allocates an atomically reference counted buffer of that size, and then reads from disk to fill the buffer, casting it back into the asset data type.

Writing the entire asset to disk now looks like this:
```rust
write_slice(&asset.vertex_data, &mut writer)?;
write_slice(&asset.vertex_ids, &mut writer)?;
write_slice(&asset.indices, &mut writer)?;
write_slice(&asset.meshlets, &mut writer)?;
write_slice(&asset.bounding_spheres, &mut writer)?;
```

And reading it back from disk looks like this:
```rust
let vertex_data = read_slice(reader)?;
let vertex_ids = read_slice(reader)?;
let indices = read_slice(reader)?;
let meshlets = read_slice(reader)?;
let bounding_spheres = read_slice(reader)?;

Ok(MeshletMesh {
    vertex_data,
    vertex_ids,
    indices,
    meshlets,
    bounding_spheres,
})
```

Total load time from disk to CPU memory for our 5mb MeshletMesh went from 102ms down to 12ms, an 8.5x speedup.

## Software Rasterization
PR [#14623](https://github.com/bevyengine/bevy/pull/14623) improves our visbuffer rasterization performance for clusters that appear small on screen (i.e. almost all of them). I rewrote pretty much the entire virtual geometry codebase in this PR, so this is going to be a really long section.

### Motivation

If you remember the frame breakdown from the last post, visbuffer rasterization took the largest chunk of our frame time. Writing out a buffer of cluster + triangle IDs to render in the culling pass, and then doing a single indirect draw over the total count of triangles does not scale very well.

The buffer used a lot of memory (4 bytes per non-culled triangle). The GPU's primitive assembler can't keep up with the sheer number of vertices we're sending it as we're not using indexed triangles (to save extra memory and time spent writing out an index buffer), and therefore lack a vertex cache. And finally the GPU's rasterizer just performs poorly with small triangles, and we have a _lot_ of small triangles.

Current GPU rasterizers expect comparatively few triangles that each cover many pixels. They have performance optimizations aimed at that kind of workload like shading 2x2 quads of pixels at a time and tile binning of triangles. Meanwhile, our virtual geometry renderer is aimed at millions of tiny triangles that only cover a pixel each. We need a rasterizer aimed at being efficient over the number of triangles; not the number of covered pixels per triangle.

We need a custom rasterizer algorithm, written in a compute shader, that does everything the GPU's hardware rasterizer does, but with the extra optimizations stripped out.

### Preparation

Before we get to the actual software rasterizer, there's a bunch of prep work we need to do first. Namely, redoing our entire hardware rasterizer setup.

In Bevy 0.14, we were writing out a buffer of triangles from the culling pass, and issuing a single indirect draw to rasterize every triangle in the buffer. We're going to throw all that out, and go with a completely new scheme.

First, we need a buffer for to store a bunch of cluster IDs (the ones we want to rasterize). We'll have users give a fixed size for this buffer on startup, based on the maximum number of clusters they expect to have visible in a frame in any given scene (not the amount pre-culling and LOD selection).

```rust
MeshletPlugin { cluster_buffer_slots: 8192 }

render_device.create_buffer(&BufferDescriptor {
    label: Some("meshlet_raster_clusters"),
    size: cluster_buffer_slots as u64 * size_of::<u32>() as u64,
    usage: BufferUsages::STORAGE,
    mapped_at_creation: false,
});
```

Next, we'll setup two indirect commands in some buffers. One for hardware raster, one for software raster. For hardware raster, we're going to hardcode the vertex count to 64 (the maximum number of triangles per meshlet) times 3 (vertices per triangle) total vertices. We'll also initialize the instance count to zero.

This was a sceme I described in my last post, but purposefully avoided due to the lackluster performance. However, now that we're adding a software rasterizer, I expect that almost all clusters will be software rasterized. Therefore some performance loss for the hardware raster is acceptable, as it should be rarely used. In return, we'll get to use a nice trick in the next step.

```rust
render_device.create_buffer_with_data(&BufferInitDescriptor {
    label: Some("meshlet_hardware_raster_indirect_args"),
    contents: DrawIndirectArgs {
        vertex_count: 64 * 3,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    }
    .as_bytes(),
    usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
});

render_device.create_buffer_with_data(&BufferInitDescriptor {
    label: Some("meshlet_software_raster_indirect_args"),
    contents: DispatchIndirectArgs { x: 0, y: 1, z: 1 }.as_bytes(),
    usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
});
```

In the culling pass, after LOD selection and culling, we're going to replace the the triangle buffer writeout code with something new.

First we need to decide if the cluster is going to be software rasterized, or hardware rasterized. For this, my current heuristic is to take the cluster's screen-space AABB size we already calculated for occlusion culling, and check how big it is. If it's small (currently < 64 pixels on both axis), then it should be software rasterized. If it's large, then it gets hardware rasterized.

At some point, when I have some better test scenes setup, I'll need to experiment with this parameter and see if I get better results with a different number.

```rust
let cluster_is_small = all(vec2(aabb_width_pixels, aabb_height_pixels) < vec2(64.0));
```

Finally, the culling pass needs to output a list of clusters for both software and hardware rasterization. For this, I'm going to borrow a trick from Unreal's Nanite I learned from this [frame breakdown](https://www.elopezr.com/a-macro-view-of-nanite).

Instead of allocating two buffers (one for SW raster, one for HW raster), we have the one `meshlet_raster_clusters` buffer that we'll share between them, saving memory. Software rasterized clusters will be added starting from the left side of the buffer, while hardware rasterized clusters will be added from the right side of the buffer. As long as the buffer is big enough, they'll never overlap.

Software rasterized clusters will increment the previously created indirect dispatch (1 workgroup per cluster), while hardware rasterized clusters will increment the previously created indirect draw (one draw instance per cluster).

```rust
var buffer_slot: u32;
if cluster_is_small && not_intersects_near_plane {
    // Append this cluster to the list for software rasterization
    buffer_slot = atomicAdd(&meshlet_software_raster_indirect_args.x, 1u);
} else {
    // Append this cluster to the list for hardware rasterization
    buffer_slot = atomicAdd(&meshlet_hardware_raster_indirect_args.instance_count, 1u);
    buffer_slot = constants.meshlet_raster_cluster_rightmost_slot - buffer_slot;
}
meshlet_raster_clusters[buffer_slot] = cluster_id;
```

### Hardware Rasterization and atomicMax

We can now perform the indirect draw for hardware rasterization, and an indirect dispatch for software rasterization.

In the hardware rasterization pass, since we're now spawning `MESHLET_MAX_TRIANGLES * 3` vertices per cluster, we now need extra vertex shader invocations to write NaN triangle positions to ensure the extra triangles gets discarded.

```rust
@vertex
fn vertex(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let cluster_id = meshlet_raster_clusters[meshlet_raster_cluster_rightmost_slot - instance_index];
    let meshlet_id = meshlet_cluster_meshlet_ids[cluster_id];
    var meshlet = meshlets[meshlet_id];

    let triangle_id = vertex_index / 3u;
    if triangle_id >= get_meshlet_triangle_count(&meshlet) { return dummy_vertex(); }

    // ...
}
```

In the fragment shader, instead of writing to a bound render target, we're now going to do an atomicMax() on a storage buffer to store the rasterized visbuffer result. The reason is that we'll need to do the same for the software rasterization pass (because compute shaders don't have access to render targets), so to keep things simple and reuse the same bind group and underlying texture state between the rasterization passes, we're going to stick to using the atomicMax() trick for the hardware rasterization pass as well. The Nanite slides describe this in more detail if you want to learn more.

```rust
@fragment
fn fragment(vertex_output: VertexOutput) {
    let frag_coord_1d = u32(vertex_output.position.y) * u32(view.viewport.z) + u32(vertex_output.position.x);

    let depth = bitcast<u32>(vertex_output.position.z);
    let visibility = (u64(depth) << 32u) | u64(vertex_output.packed_ids);
    atomicMax(&meshlet_visibility_buffer[frag_coord_1d], visibility);
}
```

Special thanks to [@atlv24](https://github.com/atlv24) for adding 64-bit integers and atomic u64 support in wgpu 22, specifically so that I could use it here.

Note that there's a couple of improvements we could make here still, pending on support in wgpu and naga for some missing features:
* R64Uint texture atomics would both be faster than using buffers, and a bit more ergonomic to sample from and debug. This is hopefully coming in wgpu 24, again thanks to @atlv24.
* Async compute would let us overlap the hardware and software rasterization passes, which would be safe since they're both writing to the same texture/buffer using atomics, which is another reason to stick with atomics for hardware raster.
* Wgpu currently requires us to bind an empty render target for the hardware raster, even though we don't ever write to it, which is a waste of VRAM. Ideally we wouldn't need any bound render target.
* And of course if we had mesh shaders, I wouldn't use a regular draw at all.

### Rewriting the Indirect Dispatch

Before we get to software rasterization (soon, I promise!), we first have to deal with one final problem.

We're expecting to deal with a _lot_ of visible clusters. For each software rasterized cluster, we're going to increment the X dimension of an indirect dispatch, with 1 workgroup per cluster. On some GPUs (mainly AMD), you're limited to 65536 workgroups per dispatch dimension, which is too low. We need to do the same trick we've done in the past of turning a 1d dispatch into a higher dimension dispatch (in this case 2d), and then reinterpreting it back as a 1d dispatch ID in the shader.

Since this is an indirect dispatch, we'll need to run a single-thread shader after the culling pass and before software rasterization, to do the 1d -> 2d remap of the indirect dispatch arguments on the GPU.

```rust
@compute
@workgroup_size(1, 1, 1)
fn remap_dispatch() {
    meshlet_software_raster_cluster_count = meshlet_software_raster_indirect_args.x;

    if meshlet_software_raster_cluster_count > max_compute_workgroups_per_dimension {
        let n = u32(ceil(sqrt(f32(meshlet_software_raster_cluster_count))));
        meshlet_software_raster_indirect_args.x = n;
        meshlet_software_raster_indirect_args.y = n;
    }
}
```

### The Software Rasterizer

Finally, we can do software rasterization.

The basic idea is to have a compute shader workgroup with size equal to the max triangles per meshlet.

Each thread within the workgroup will load 1 vertex of the meshlet, transform it to screen-space, and then write it to workgroup shared memory and issue a barrier.

After the barrier, the workgroup will switch to handling triangles, with one thread per triangle. First each thread will load the 3 indices for its triangle, and then load the 3 vertices from workgroup shared memory based on the indices.

Once each thread has the 3 vertices for its triangle, it can compute the position/depth gradients across the triangle, and screen-space bounding box around the triangle.

Each thread can then iterate the bounding box (Like Nanite does, choosing to either iterate each pixel or iterate scanlines, based on the bounding box sizes across the subgroup), writing pixels to the visbuffer as it goes using the same atomicMax() method that we used for hardware rasterization.

One notable difference to the Nanite slides is that for the scanline variant, I needed to check if the pixel center was within the triangle for each pixel in the scanline, which the slides don't show. Not sure if the slides just omitted it for brevity or what, but I got artifacts if I left the check out.

There's also some slight differences between my shader and the GPU rasterizer - I didn't implement absolutely every detail. Notably I skipped fixed-point math and the top-left rule. I should implement these in the future, but for now I haven't seen any issues from skipping them.

### Material and Depth Resolve

In Bevy 0.15, after the visbuffer rasterization, we have two final steps.

The resolve depth pass reads the visbuffer (which contains packed depth), and writes the depth to the actual depth texture of the camera.

```rust
/// This pass writes out the depth texture.
@fragment
fn resolve_depth(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    let frag_coord_1d = u32(in.position.y) * view_width + u32(in.position.x);
    let visibility = meshlet_visibility_buffer[frag_coord_1d];
    return bitcast<f32>(u32(visibility >> 32u));
}
```

The resolve material depth pass has the same role in Bevy 0.15 that it did in Bevy 0.14, where it writes the material ID of each pixel to a depth texture, so that we can later abuse depth testing to ensure we shade the correct pixels during each material draw in the material shading pass.

However, you may have noticed that unlike the rasterization pass in Bevy 0.14, the new rasterization passes write only depth and cluster + triangle IDs, and not material IDs. During the rasterization pass, where we want to write only the absolute minimum amount of information per pixel (cluster ID, triangle ID, and depth) that we have to.

Because of this, the resolve material depth pass can no longer read the material ID texture and copy it directly to the material depth texture. There's now a new step at the start to first load the material ID based on the visbuffer.

```rust
/// This pass writes out the material depth texture.
@fragment
fn resolve_material_depth(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    let frag_coord_1d = u32(in.position.y) * view_width + u32(in.position.x);
    let visibility = meshlet_visibility_buffer[frag_coord_1d];

    let depth = visibility >> 32u;
    if depth == 0lu { return 0.0; }

    let cluster_id = u32(visibility) >> 7u;
    let instance_id = meshlet_cluster_instance_ids[cluster_id];
    let material_id = meshlet_instance_material_ids[instance_id];

    // Everything above this line is new - the shader used to just load the
    // material_id from another texture

    return f32(material_id) / 65535.0;
}
```

### Retrospect

Software rasterization is a lot of complexity, learning, and work - I spent a lot of time researching how the GPU rasterizer works, rewrote a _lot_ of code, and just writing the software rasterization shader itself and getting it bug-free took a week or two of effort. As you'll see later, I missed a couple of (severe) bugs, which will need correcting.

The upside is that performance is a _lot_ better than my previous method (I'll show some numbers at the end of this post), and we can have thousands of tiny triangles on screen without hurting performance.

My advice to others working on virtual geometry is to skip software raster until close to the end. If you have mesh shaders, stick with those. From what I've heard from other projects, software raster is only a 10-20% performance improvement over mesh shaders in most scenes, unless you really crank the tiny triangle count (which is admittedly a goal, but not an immediate priority).

If like me, you don't have mesh shaders, then I would still probably stick with only hardware rasterization until you've exhausted other, more fundamental areas to work on, like culling and DAG building. However, I would learn from my mistakes, and not spend so much time trying to get hardware rasterization to be fast. Just stick to writing out a list of visible cluster IDs in the culling shader and have the vertex shader ignore extra triangles, instead of trying to get clever with writing out a buffer of visible triangles and drawing the minimum number of vertices. You'll eventually add software rasterization, and then the hardware rasterization performance won't be so important.

If you do want to implement a rasterizer in software (for virtual geometry, or otherwise), check out the below resources that were a big help for me in learning rasterization and the related math.

* <https://kristoffer-dyrkorn.github.io/triangle-rasterizer>
* <https://fgiesen.wordpress.com/2013/02/06/the-barycentric-conspirac>
* <https://www.youtube.com/watch?v=k5wtuKWmV48>

## Larger Meshlet Sizes
PR [#15023](https://github.com/bevyengine/bevy/pull/15023) has a bunch of small improvements to virtual geometry.

The main change is switching from a maximum 64 vertices and 64 triangles (`64v:64t`) to 255 vertices and 128 triangles per meshlet (`255v:128t`). I found that having a less than or equal `v:t` ratio leads to most meshlets having less than `t` triangles, which we don't want. Having a `2v:t` ratio leads to more fully-filled meshlets, and I went with `255v:128t` (which is nearly the same as Nanite, minus the fact that meshoptimizer only supports meshlets with up to 255 vertices) over `128v:64t` after some performance testing.

Note that this change involved some other work, such as adjusting software and hardware raster to work with more triangles, software rasterization looping if needed to load 2 vertices per thread instead of 1, using another bit per triangle ID when packing cluster + triangle IDs to accomodate triangles up to 127, etc.

The other changes I made were:
* Setting the target error when simplifying triangles to `f32::MAX` (no point in capping it for continuous LOD, gives better simplification results)
* Adjusting the threshold to allow less-simplified meshes to still count as having been simplified enough (gets us closer to `log2(lod_0_meshlet_count)` total LOD levels)
* Setting `group_error = max(group_error, all_child_errors)` instead of `group_error += max(all_child_errors)` (not really sure if this is more or less correct)

## Screenspace-derived Tangents
PR [#15084](https://github.com/bevyengine/bevy/pull/15084) calculates tangents at runtime, instead of precomputing them and storing them as part of the MeshletMesh asset.

Virtual geometry isin't just about rasterizing huge amounts of high-poly meshes - asset size is also a _big_ factor. GPUs only have so much memory, disks only have so much space, and transfer speeds from disk to RAM and RAM to VRAM is only so fast (as we discovered in the last post).

Looking at our asset data, right now we're storing 48 bytes per vertex, with a single set of vertices shared across all meshlets in a meshlet mesh.

```rust
struct MeshletVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}
```

An easy way to reduce the amount of data per asset is to just remove the explicitly-stored tangents, and instead calculate them at runtime. In the visbuffer resolve shader function, rather then loading 3 vertex tangents and interpolating across the triangle, we can instead calculate the tangent based on UV derivatives across the triangle.

The tangent derivation I used was ["Surface Gradient–Based Bump Mapping Framework"](https://jcgt.org/published/0009/03/04) from Morten S. Mikkelsen (author of the [mikktspace](http://www.metalliandy.com/mikktspace/tangent_space_normal_maps.html) standard). It's a really cool paper that provides a framework for using normal maps in many more scenarios than just screen-space based tangents. Definitely give it a further read.

I used the code from this <https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping> blog post, which also does a great job motivating and explaining the paper.

The only issue I ran into is that the tangent.w always came out with the wrong sign compared to the existing mikktspace-tangents I had as a reference. I double checked my math and coordinate space handiness a couple of times, but could never figure out what was wrong. I ended up just inverting the sign after calculating the tangent. If anyone knows what I did wrong, please open an [issue](https://github.com/bevyengine/bevy/issues)!

```rust
// https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping/#surface-gradient-from-a-tangent-space-normal-vector-without-an-explicit-tangent-basis
fn calculate_world_tangent(
    world_normal: vec3<f32>,
    ddx_world_position: vec3<f32>,
    ddy_world_position: vec3<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
) -> vec4<f32> {
    // Project the position gradients onto the tangent plane
    let ddx_world_position_s = ddx_world_position - dot(ddx_world_position, world_normal) * world_normal;
    let ddy_world_position_s = ddy_world_position - dot(ddy_world_position, world_normal) * world_normal;

    // Compute the jacobian matrix to leverage the chain rule
    let jacobian_sign = sign(ddx_uv.x * ddy_uv.y - ddx_uv.y * ddy_uv.x);

    var world_tangent = jacobian_sign * (ddy_uv.y * ddx_world_position_s - ddx_uv.y * ddy_world_position_s);

    // The sign intrinsic returns 0 if the argument is 0
    if jacobian_sign != 0.0 {
        world_tangent = normalize(world_tangent);
    }

    // The second factor here ensures a consistent handedness between
    // the tangent frame and surface basis w.r.t. screenspace.
    let w = jacobian_sign * sign(dot(ddy_world_position, cross(world_normal, ddx_world_position)));

    return vec4(world_tangent, -w); // TODO: Unclear why we need to negate this to match mikktspace generated tangents
}
```

At the cost of a few extra calculations in the material shading pass, and some slight inaccuracies compared to explicit tangents, mostly on curved surfaces, we save 16 bytes per vertex, both on disk (although LZ4 compression means we might be saving less in practice), and in memory.

TODO: Image comparison

Also of note is that while trying to debug the sign issue, I found that The Forge had published an [updated version](https://github.com/ConfettiFX/The-Forge/blob/9d43e69141a9cd0ce2ce2d2db5122234d3a2d5b5/Common_3/Renderer/VisibilityBuffer2/Shaders/FSL/vb_shading_utilities.h.fsl#L90-L150) of their partial derivatives calculations, fixing a small bug. I updated my WGSL port to match.

## Compressed Per-Meshlet Vertex Data
PR [#15643](https://github.com/bevyengine/bevy/pull/15643) stores copies of the overall mesh's vertex attribute data per-meshlet, and then heavily compresses it.

### Motivation

The whole idea behind virtual geometry is that you only pay (as much as possible, it's of course not perfect) for the geometry currently needed on screen. Zoomed out? You pay the rasterization cost for only a few triangles at a higher LOD, and not for the entire mesh. Part of the mesh occluded? It gets culled. But continuing on with the theme from the last PR, memory usage is also a big cost. We might be able to render a large scene of high poly meshes with clever usage of LODs and culling, but can we afford to _store_ all that mesh data to begin with in our GPU's measily 8-12gb of VRAM? (not even accounting for space taken up by material textures which will reduce our budget even further).

The way we fix this is with streaming. Rather than keep everything in memory all the time, you have the GPU write requests of what data it needs to a buffer, read that back onto the CPU, and then load the requested data from disk into a fixed-size GPU buffer. If the GPU no longer needs a piece of data, you mark that section of the buffer as free space, and can write new data to it as new requests come in.

Typical implementations stream entire LODs at a time, but our goal is to be much more fine-grained. We want to stream individual meshlets, not prebuilt LODs (in practice, Nanite streams fixed-size pages of meshlet data, and not individual meshlets). This presents a problem with our current implementation: since all meshlets reference the same set of vertex data, we have no simple way of unloading or loading vertex data for a single meshlet. While I'm not going to tackle streaming in Bevy 0.15, in this PR I'll be changing the way we store vertex data to solve this problem and unblock streaming in the future.

Up until now, each MeshletMesh has had one set of vertex data shared between all meshlets within the mesh. Each meshlet has a local index buffer, mapping triangles to meshlet-local vertex IDs, and then a global index buffer mapping meshlet-local vetex IDs to actual vertex data from the mesh. E.g. triangle corner X within a meshlet points to vertex ID Y within a meshlet which points to vertex Z within the mesh.

In order to support streaming, we're going to move to a new scheme. We will store a copy of vertex data for each meshlet, concatenated together into one slice. All the vertex data for meshlet 0 will be stored as one contiguous slice, with all the vertex data for meshlet 1 stored contiguously after it, and all the vertex data for meshlet 2 after _that_, etc.

Each meshlet's local index buffer will point directly into vertices within the meshlet's vertex data slice, stored as an offset relative to the starting index of the meshlet's vertex data slice within the overall buffer. E.g. triangle corner X within a meshlet points to vertex Y within the meshlet directly.

Besides unblocking streaming, this scheme is also much simpler to reason about, uses less dependent memory reads, and works much nicer with our software rasterization pass where each thread in the workgroup is loading a single meshlet vertex into workgroup shared memory.

That was a lot of background and explanation for what's really a rather simple change, so let me finally get to the main topic of this PR: the problem with duplicating vertex data per meshlet is that we've just increased the size of our MeshletMesh asset by a thousandfold.

The solution is quantization and compression.

### Position Compression

Meshlets compress pretty well. Starting with vertex positions, there's no reason we need to store a full `vec3<f32>` per vertex. Most meshlets tend to enclose a fairly small amount of space. Instead of storing vertex positions as coordinates relative to the mesh center origin, we can instead store them in some coordinate space relative to the meshlet bounds.

For each meshlet, we'll iterate over all of its vertex positions, and calculate the min and max value for each of the X/Y/Z axis. Then, we can remap each position relative to those bounds by doing `p -= min`. The positions initially range from `[min, max]`, and then range from `[0, max - min]` after remapping. We can store the `min` values for each of the X/Y/Z axis (as a full `f32` each) in the meshlet metadata, and in the shader reverse the remapping by doing `p += min`.

Our first (albeit small) saving become apparent: at the cost of 12 extra bytes in the meshlet metadata, we save 3 bits per vertex position due to no longer needing a bit for the sign for each of the X/Y/Z values, as `[0, max - min]` is never going to contain any negative numbers. We technically now only need a hypothetical `f31` per axis.

However, there's a another trick we can perform. If take the ceiling of the log2 of a range of floating point values `ceil(log2(max - min + 1))`, we get the minimum number of bits we need to store any value in that range. Rather than storing meshlet vertex positions as a list of `vec3<f32>`s, we could instead store them as a packed list of bits (a bitstream).

E.g. if we determine that we need 4/7/3 bits for the X/Y/Z ranges of the meshlet, we could store a list of bits where bits 0..4 are for vertex 0 axis X, bits 4..11 are for vertex 0 axis Y, bits 11..14 are for vertex 0 axis Z, bits 14..18 are for vertex 1 axis X, bits 18..25 are for vertex 1 axis Y, etc.

Again we can store the bit size (as a `u8`) for each of the X/Y/Z axis within the meshlet's metadata, at a cost of 3 extra bytes. We'll use this later in our shaders to figure out how many bits to read from the bistream for each of the meshlet's vertices.

Of course, if you try this out, you're probably going to end up with fairly large bit sizes per axis, and not actually save any space vs using `vec3<f32>`. This is due to the large amount of precision we have in our vertex positions (a full `f32`), which leads to a lot of precision needed in the range, and therefore a large bit size.

The final trick up our sleeves is that we don't actually _need_ all this precision. If we know that our meshlet's vertices range from 10.2041313123 to 84.382543538, do we really need to know that a vertex happens to be stored at _exactly_ 57.594392822? We could pick some arbitrary amount of precision to round each of our vertices to, say four decimal places, resulting in 57.5944. Less precision means a less precise range, which means our bit size will be smaller.

Better yet, lets pick some factor `q = 2^p`, where `p` is some arbitrary `u8` integer. Now, lets snap each vertex to the nearest point on the grid that's a multiple of `1/q`, and then store the vertex as the number of "steps" of size `1/q` that we took from the origin to reach the snapped vertex position (a fixed-point representation). E.g. if we say `p = 4`, then we're quantizing to a grid with a resolution of `1/16`, so `v = 57.594392822` would snap to `v = 57.625` (throwing away some unnecessary precision) and we would store that as `v = round(57.594392822 / (1/16)) = i32(57.594392822 * 16 + 0.5) = 922`. This is once again easily reversible in our shader so long as we have our factor `p`: `922 / 2^4 = 57.625`.

The factor `p` we choose is not particularly important. I set it to 4 by default (with an additional factor to convert from Bevy's meters to the more appropriate-for-this-use-case unit of centimeters), but users can choose a good value themselves if 4 is too high (unnecessary precision = larger bit sizes and therefore larger asset sizes), or too low (visible mesh deformity from snapping the vertices too-coarsely). Nanite has an automatic heuristic that I assume is based on some kind of triangle surface area to mesh size ratio, but also lets users choose `p` manually. The important thing to note is that you should _not_ choose `p` per-meshlet, i.e. `p` should be the same for every meshlet within the mesh. Otherwise, you'll end up with cracks between meshlets.

Finally, we can combine all three of these tricks. We can quantize our meshlet's vertices, find the per-axis min/max values and remap to a better range, and then store as a packed bitstream using the minimum number of bits for the range. The final code to compress a meshlet's vertex positions is below.

```rust
let quantization_factor =
    (1 << vertex_position_quantization_factor) as f32 * CENTIMETERS_PER_METER;

let mut min_quantized_position_channels = IVec3::MAX;
let mut max_quantized_position_channels = IVec3::MIN;

// Lossy vertex compression
let mut quantized_positions = [IVec3::ZERO; 255];
for (i, vertex_id) in meshlet_vertex_ids.iter().enumerate() {
    let position = ...;

    // Quantize position to a fixed-point IVec3
    let quantized_position = (position * quantization_factor + 0.5).as_ivec3();
    quantized_positions[i] = quantized_position;

    // Compute per X/Y/Z-channel quantized position min/max for this meshlet
    min_quantized_position_channels = min_quantized_position_channels.min(quantized_position);
    max_quantized_position_channels = max_quantized_position_channels.max(quantized_position);
}

// Calculate bits needed to encode each quantized vertex position channel based on the range of each channel
let range = max_quantized_position_channels - min_quantized_position_channels + 1;
let bits_per_vertex_position_channel_x = log2(range.x as f32).ceil() as u8;
let bits_per_vertex_position_channel_y = log2(range.y as f32).ceil() as u8;
let bits_per_vertex_position_channel_z = log2(range.z as f32).ceil() as u8;

// Lossless encoding of vertex positions in the minimum number of bits per channel
for quantized_position in quantized_positions.iter().take(meshlet_vertex_ids.len()) {
    // Remap [range_min, range_max] IVec3 to [0, range_max - range_min] UVec3
    let position = (quantized_position - min_quantized_position_channels).as_uvec3();

    // Store as a packed bitstream
    vertex_positions.extend_from_bitslice(
        &position.x.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_x as usize],
    );
    vertex_positions.extend_from_bitslice(
        &position.y.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_y as usize],
    );
    vertex_positions.extend_from_bitslice(
        &position.z.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_z as usize],
    );
}
```

### Position Decoding

Before this PR, our meshlet metadata was this 16-byte type:

```rust
pub struct Meshlet {
    /// The offset within the parent mesh's [`MeshletMesh::vertex_ids`] buffer where the indices for this meshlet begin.
    pub start_vertex_id: u32,
    /// The offset within the parent mesh's [`MeshletMesh::indices`] buffer where the indices for this meshlet begin.
    pub start_index_id: u32,
    /// The amount of vertices in this meshlet.
    pub vertex_count: u32,
    /// The amount of triangles in this meshlet.
    pub triangle_count: u32,
}
```

With all the custom compression, we need to store some more info, giving us this carefully-packed 32-byte type (a little bit bigger, but reducing size for vertices is much more important than reducing the size of the meshlet metadata):

```rust
pub struct Meshlet {
    /// The bit offset within the parent mesh's [`MeshletMesh::vertex_positions`] buffer where the vertex positions for this meshlet begin.
    pub start_vertex_position_bit: u32,
    /// The offset within the parent mesh's [`MeshletMesh::vertex_normals`] and [`MeshletMesh::vertex_uvs`] buffers
    /// where non-position vertex attributes for this meshlet begin.
    pub start_vertex_attribute_id: u32,
    /// The offset within the parent mesh's [`MeshletMesh::indices`] buffer where the indices for this meshlet begin.
    pub start_index_id: u32,
    /// The amount of vertices in this meshlet.
    pub vertex_count: u8,
    /// The amount of triangles in this meshlet.
    pub triangle_count: u8,
    /// Unused.
    pub padding: u16,
    /// Number of bits used to to store the X channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_x: u8,
    /// Number of bits used to to store the Y channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_y: u8,
    /// Number of bits used to to store the Z channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_z: u8,
    /// Power of 2 factor used to quantize vertex positions within this meshlet.
    pub vertex_position_quantization_factor: u8,
    /// Minimum quantized X channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_x: f32,
    /// Minimum quantized Y channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_y: f32,
    /// Minimum quantized Z channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_z: f32,
}
```

To fetch a single vertex from the bitstream (we we bind as an array of `u32`s), we can use this function:

```rust
fn get_meshlet_vertex_position(meshlet: ptr<function, Meshlet>, vertex_id: u32) -> vec3<f32> {
    // Get bitstream start for the vertex
    let unpacked = unpack4xU8((*meshlet).packed_b);
    let bits_per_channel = unpacked.xyz;
    let bits_per_vertex = bits_per_channel.x + bits_per_channel.y + bits_per_channel.z;
    var start_bit = (*meshlet).start_vertex_position_bit + (vertex_id * bits_per_vertex);

    // Read each vertex channel from the bitstream
    var vertex_position_packed = vec3(0u);
    for (var i = 0u; i < 3u; i++) {
        let lower_word_index = start_bit / 32u;
        let lower_word_bit_offset = start_bit & 31u;
        var next_32_bits = meshlet_vertex_positions[lower_word_index] >> lower_word_bit_offset;
        if lower_word_bit_offset + bits_per_channel[i] > 32u {
            next_32_bits |= meshlet_vertex_positions[lower_word_index + 1u] << (32u - lower_word_bit_offset);
        }
        vertex_position_packed[i] = extractBits(next_32_bits, 0u, bits_per_channel[i]);
        start_bit += bits_per_channel[i];
    }

    // Remap [0, range_max - range_min] vec3<u32> to [range_min, range_max] vec3<f32>
    var vertex_position = vec3<f32>(vertex_position_packed) + vec3(
        (*meshlet).min_vertex_position_channel_x,
        (*meshlet).min_vertex_position_channel_y,
        (*meshlet).min_vertex_position_channel_z,
    );

    // Reverse vertex quantization
    let vertex_position_quantization_factor = unpacked.w;
    vertex_position /= f32(1u << vertex_position_quantization_factor) * CENTIMETERS_PER_METER;

    return vertex_position;
}
```

This could probably be written better - right now we're doing a minimum of 3 `u32` reads (1 per channel), but there's a good chance that a single `u32` read will contain the data for all 3 channels of the vertex. Something to optimize in the future.

### Other Attributes

Now that we've done positions, lets talk about how to handle other vertex attributes.

Tangents we already removed in the last PR.

For UVs, I currently store them uncompressed. I could have maybe used half-precision floating point values, but I am wary of artifacts resulting from the reduced precision, so for right now it's a full `vec2<f32>`. This is a big opportunity for future improvement.

Normals are a bit more interesting. They start as `vec3<f32>`. I first perform an octahedral encoding on them, bringing them down to a `vec2<f32>` near-losessly. I then give up some precision to reduce the size even further by using `pack2x16snorm()`, bringing it down to a `vec2<f16>`, or a packed `u32`. These operations are easily reversed in the shader using the built-in `unpack2x16snorm()` function, and then the simple octahedral decode step.

I _did_ try a bitstream encoding similiar to what I did for positions, but couldn't get any smaller sizes than a simple `pack2x16snorm()`. I think with more time and motivation (I was getting burnt out by the end of this), I could have probably figured out a good variable-size octahedral encoding for normals as well. Something else to investigate in the future.

### Results

After all this, how much memory savings did we get?

Disk space is practically unchanged (maybe 2% smaller at best), but memory savings on a test mesh went from `109.972084mb` before this PR (without duplicating the vertex data per-meshlet at all), to `63.614636mb` after this PR (copying and compressing vertex data per-meshlet). Huge savings, with room for future improvements! I'll definitely be coming back to this at some point in the future.

Additional references:
* <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf#page=128>
* <https://arxiv.org/abs/2404.06359> (also compresses the index buffer, not just vertices!)
* <https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html>
* <https://gpuopen.com/download/publications/DGF.pdf> (more focused on raytracing than rasterization)

## Improved LOD Selection Heuristic
PR [#15846](https://github.com/bevyengine/bevy/pull/15846) changes how we select the LOD cut.

Previously, I was building a bounding sphere around each group with radius based on the group error, and then projecting that to screen space to get the visible error in pixels.

That method worked, but isin't entirely watertight. Where you place the bounding sphere center in the group is kind of arbitrary, right? And how do you ensure that the error projection is perfectly monotonic, if you have these random bounding spheres in each group?

Arseny Kapoulkine once again helped me out here. As part of meshoptimizer, they started experimenting with their [nanite.cpp](https://github.com/zeux/meshoptimizer/blob/d93419ced5956307f41333c500c8037c8b861d59/demo/nanite.cpp) demo. In this PR, I copied his code for LOD cut selection.

To determine the group bounding sphere, you simply build a new bounding sphere enclosing all of the group's childrens' bounding spheres. The first group you build out of LOD 0 uses the LOD 0 culling bounding spheres around each meshlet. This way, you ensure that both the error (using the existing method of taking the max error among the group and group children), _and_ the bounding sphere are monotonic. Error is no longer stored in the radius of the bounding sphere, and is instead stored as a seperate f16 (lets us pack both group and parent group error into a single u32, and the lost precision is irrelevant). This also gave me the opportunity to clean up the code now that I understand the theory better, and clarify the difference between meshlets and meshlet groups better.

For projecting the error at runtime, we now use the below function. I can't claim to understand how it works that well (and it's been a few weeks since I last looked at it), but it does work. The end result is that we get more seamless LOD changes, and our mesh to meshlet mesh converter is more robust (it used to crash on larger meshes, due to a limitation in the code for how I calculated group bounding spheres).

```rust
// https://github.com/zeux/meshoptimizer/blob/1e48e96c7e8059321de492865165e9ef071bffba/demo/nanite.cpp#L115
fn lod_error_is_imperceptible(lod_sphere: MeshletBoundingSphere, simplification_error: f32, world_from_local: mat4x4<f32>, world_scale: f32) -> bool {
    let sphere_world_space = (world_from_local * vec4(lod_sphere.center, 1.0)).xyz;
    let radius_world_space = world_scale * lod_sphere.radius;
    let error_world_space = world_scale * simplification_error;

    var projected_error = error_world_space;
    if view.clip_from_view[3][3] != 1.0 {
        // Perspective
        let distance_to_closest_point_on_sphere = distance(sphere_world_space, view.world_position) - radius_world_space;
        let distance_to_closest_point_on_sphere_clamped_to_znear = max(distance_to_closest_point_on_sphere, view.clip_from_view[3][2]);
        projected_error /= distance_to_closest_point_on_sphere_clamped_to_znear;
    }
    projected_error *= view.clip_from_view[1][1] * 0.5;
    projected_error *= view.viewport.w;

    return projected_error < 1.0;
}
```

An interesting side note, finding the minimal bounding sphere around a set of other bounding sphere turns out to be a very difficult problem. Kaspar Fischer's thesis ["The smallest enclosing balls of balls"](http://www.inf.ethz.ch/personal/emo/DoctThesisFiles/fischer05.pdf) covers the math, and it's very complex. I copied Kapoulkine's approximate, much simpler method.

## Improved Mesh to MeshletMesh Conversion
PR [#15886](https://github.com/bevyengine/bevy/pull/15886) brings more improvements to the mesh to meshlet mesh converter.

Following on from the last PR, I again took a bunch of improvements from the meshoptimizer nanite.cpp demo:

* Consider only the vertex position (and ignore things like UV seams) when determining meshlet groups
* Add back stuck meshlets that either failed to simplify, or failed to group, to the processing queue to try again at a later LOD. Dosen't seem to be much of an improvement though.
* Provide a seed to METIS to make the meshlet mesh conversion fully deterministic. I didn't realize METIS even had options before now.
* Target groups of 8 meshlets instead of 4. This improved simplification quality a lot! Nanite does groups of size 8-32, probably based on some kind of heuristic, which is probably worth experimenting with in the future.
* Manually lock only vertices belonging to meshlet group borders, instead of the full toplogical group border that meshoptimizer's `LOCK_BORDER` flag does.

With all of these changes combined, we can finally reliably get down to a single meshlet (or at least 1-3 meshlets for larger meshes) at the highest LOD!

The last item on the list in particular is a _huge_ improvement. Consider this scenario: TODO

## Faster Fill Cluster Buffers Pass
PR [#15955](https://github.com/bevyengine/bevy/pull/15955) improves the speed of the fill cluster buffers pass.

### Speeding Up

At this point, I improved rasterization performance, meshlet mesh building, and asset storage and loading. The Bevy 0.15 release was coming up, people were winding down features in favor of testing the release candidates, and I wasn't going to have the time (or, the motivation) to do another huge PR.

While looking at some small things I could improve, I ended up talking with Kirill Bazhenov about how he manages per-instance (entity) GPU data in his [Esoterica](https://www.youtube.com/watch?v=8gwPw1fySMU) renderer.

To recap the problem we had in the last post, uploading 8 bytes (instance ID + meshlet ID) per cluster to the GPU was way too expensive. The solution I came up with was to dispatch a compute shader thread per cluster, have it perform a binary search on an array of per-instance data to find the instance and meshlet it belongs to, and then write out the instance and meshlet IDs. This way, we only had to upload 8 bytes per _instance_ to the GPU, and then the cluster -> instance ID + meshlet ID write outs would be VRAM -> VRAM writes, which are much faster than RAM -> VRAM uploads. This was the fill cluster buffers pass in Bevy 0.14.

It's not _super_ fast, but it's also not the bottleneck, and so for a while I was fine leaving it as-is. Kirill, however, showed me a much better way.

Instead of having our compute shader operate on a list of clusters, and write out the two IDs per cluster, we can turn the scheme on its head. We can instead have the shader operate on a list of _instances_, and write out the two IDs for each cluster within the instance. After all, each instance already has the list of meshlets it has, so writing out the cluster (an instance of a meshlet) is easy!

Instead of dispatching one thread per cluster, now we're going to dispatch one workgroup per instance, with each workgroup having 1024 threads (the maximum allowed). Instead of uploading a prefix-sum of meshlet counts per instance, now we're going to upload just a straight count of meshlets per instance (we're still only uploading 8 bytes per instance total).

In the shader, each workgroup can load the 8 bytes of data we uploaded for the instance it's processing.

```rust
let instance_id = workgroup_id.x;
let instance_meshlet_count = meshlet_instance_meshlet_counts[instance_id];
let instance_meshlet_slice_start = meshlet_instance_meshlet_slice_starts[instance_id];
```

Then, the first thread in each workgroup can reserve space in the output buffers for its instance's clusters via an atomic counter, and broadcast the start index to the rest of the workgroup.

```rust
var<workgroup> cluster_slice_start_workgroup: u32;

// Reserve cluster slots for the instance and broadcast to the workgroup
if local_invocation_index == 0u {
    cluster_slice_start_workgroup = atomicAdd(&meshlet_global_cluster_count, instance_meshlet_count);
}
let cluster_slice_start = workgroupUniformLoad(&cluster_slice_start_workgroup);
```

Finally, we can have the workgroup loop over its instance's clusters, and for each one, write out its instance ID (which we already have, since it's just the workgroup ID) and meshlet ID (the instance's first meshlet ID, plus the loop counter). Each thread will handle 1 cluster, and the workgroup as a whole will loop enough times to write out all of the instance's clusters.

```rust
// Loop enough times to write out all the meshlets for the instance given that each thread writes 1 meshlet in each iteration
for (var clusters_written = 0u; clusters_written < instance_meshlet_count; clusters_written += 1024u) {
    // Calculate meshlet ID within this instance's MeshletMesh to process for this thread
    let meshlet_id_local = clusters_written + local_invocation_index;
    if meshlet_id_local >= instance_meshlet_count { return; }

    // Find the overall cluster ID in the global cluster buffer
    let cluster_id = cluster_slice_start + meshlet_id_local;

    // Find the overall meshlet ID in the global meshlet buffer
    let meshlet_id = instance_meshlet_slice_start + meshlet_id_local;

    // Write results to buffers
    meshlet_cluster_instance_ids[cluster_id] = instance_id;
    meshlet_cluster_meshlet_ids[cluster_id] = meshlet_id;
}
```

The shader is now very efficient - the workgroup as a whole, once it reserves space for its clusters, is just repeatedly performing contiguous reads from and writes to global GPU memory.

Overall, in a test scene with 1041 instances with 32217 meshlets per instance, we went from 0.55ms to 0.40ms, a small 0.15ms savings. NSight now shows that we're at 95% VRAM throughput, and that we're bound by global memory operations. The speed of this pass is now basically dependent on our GPU's bandwidth - there's not much I could do better, short of reading and writing less data entirely.

### Hitting a Speed Bump

In the process of testing this PR, I ran into a rather confusing bug. The new fill cluster buffers pass worked on some smaller test scenes, but spawning 1042 instances with 32217 meshlets per instance (cliff mesh) lead to the below glitch. It was really puzzling - only some instances would be affected (concentrated in the same region of space), and the clusters themselves appeared to be glitching and changing each frame.

Debugging the issue was complicated by the fact that the rewritten fill cluster buffers code is no longer deterministic. Clusters get written in different orders depending on how the scheduler schedules workgroups, and the order of the atomic writes. That meant that every time I clicked on a pass in RenderDoc to check it's output, the output order would completely change as RenderDoc replayed the entire command stream up until that point.

Since using a debugger wasn't stable enough to be useful, I tried to think the logic through. My first thought was that my rewritten code was subtly broken, but testing on mainline showed something alarming - the issue persisted. Testing several old PRs showed that it went back for several PRs. It couldn't have been due to any recent code changes.

It took me a week or so of trial and error, and debugging on mainline (which did have a stable output order since it used the old fill cluster buffers shader), but I eventually made the following observations:

* 1041 cliffs: rendered correctly
* 1042 cliffs: did _not_ render correctly, with 1 glitched instance
* 1041 + N cliffs: the last N being spawned glitched out
* 1042+ instances of a different mesh with much less meshlets than the cliff: _did_ render correctly
* 1042+ cliffs on the PR before I increased meshlet size to 255v/128t: rendered correctly

The issue turned out to be overflow of cluster ID. The output of the culling pass, and the data we store in the visbuffer, is cluster ID + triangle ID packed together in a single u32. After increasing the meshlet size, it was 25 bits for the cluster ID, and 7 bits for the triangle ID (2^7 = 128 triangles max).

Doing the math, 1042 instances * 32217 meshlets = 33570114 clusters. 2^25 - 33570114 = -15682. We had overflowed the cluster limit by 15682 clusters. This meant that the cluster IDs we were passing around were garbage values, leading to glitchy rendering on any instances we spawned after the first 1041.

Obviously this is a problem - the whole point of virtual geometry is to make rendering independent of scene complexity, yet now we have a rather low limit of 2^25 clusters in the scene.

The solution is to never store data per cluster in the scene, and only store data per _visible_ cluster in the scene, i.e. clusters post LOD selection and culling. Not necessarily visible on screen, but visible in the sense that we're going to rasterize them. Doing so would require a large amount of architectural changes, however, and is not going to be a simple and easy fix. For now, I've documented the limitation, and merged this PR confident that it's not a regression.

## Software Rasterization Bugfixes
PR [#16049](https://github.com/bevyengine/bevy/pull/16049) fixes some glitches in the software rasterizer.

While testing out some scenes to prepare for the release, I discovered some previously-missed bugs with software rasterization. When zooming in to the scene, sometimes triangles would randomly glitch and cover the whole screen, leading to massive slowdowns (remember the software rasterizer is meant to operate on small triangles only). Similarly, when zooming out, sometimes there would be single stray pixels rendered that didn't belong. These issues didn't occur with only hardware rasterization enabled.

The stray pixels turned out to be due to two issues. The first bug is in how I calculated the bounding box around each triangle. I wasn't properly accounting for triangles that would be partially on-screen, and partially off-screen. I changed my bounding box calculations to stick to floating point, and clamped negative bounds to 0 to fix. The second bug is that I didn't perform any backface culling in the software rasterizer, and ignoring it does not lead to valid results. If you want a double-sided mesh, then you need to explicitly check for backfacing triangles and invert them. If you want backface culling (I do), then you need to reject the triangle if it's backfacing. Ignoring it turned out to not be an option - skipping backface culling earlier turned out to have bitten me :).

The fullscreen triangles was trickier to figure out, but I ended up narrowing it down to near plane clipping. Rasterization math, specifically the homogenous divide, has a [singularity](https://en.wikipedia.org/wiki/Singularity_(mathematics)) when z = 0. Normally, the way you solve this is by clipping to the near plane, which is a frustum plane positioned slightly in front of z = 0. As long as you provide the plane, GPU rasterizers handle near plane clipping for you automatically. In my software rasterizer, however, I had of course not accounted for near plane clipping. That meant that we were getting Nan/Infinity vertex positions due to the singularity during the homogenous divide, which led to the garbage triangles we were seeing.

Proper near plane clipping is somewhat complicated (slow), and should not be needed for most clusters. Rather than have our software rasterizer handle near plane clipping, we're instead going to have the culling pass detect which clusters intersect the near plane, and put them in the hardware rasterization queue regardless of size. The fix for this is just two extra lines.

```rust
// Before
if cluster_is_small {
    // Software raster
} else {
    // Hardware raster
}

// After
let not_intersects_near_plane = dot(view.frustum[4u], culling_bounding_sphere_center) > culling_bounding_sphere_radius;
if cluster_is_small && not_intersects_near_plane {
    // Software raster
 } else {
    // Hardware raster
}
```

With these changes, software raster is now visibly bug-free.

TODO: Show bugged images from PR

## Normal-aware LOD Selection
PR [#16111](https://github.com/bevyengine/bevy/pull/16111) improves how we calculate the LOD cut to account for vertex normals.

At the end of the Bevy 0.15's development cycle, meshoptimizer 0.22 was released, bringing some simplification improvements. Crucially, it greatly improves `meshopt_simplifyWithAttributes()`.

I now use this function to pass vertex normals into the simplifier, meaning that the deformation error the simplifier outputs (which we feed directly into the LOD cut selection shader) accounts for not only position deformation, but also normal deformation.

Without this change, before this PR, visualizing the pixel positions was near-seamless as the LOD cut changed when you zoomed in or out. Pixel normals, however, had visible differences between LOD cuts. After this PR, normals are now near-seamless too.

There's still work to be done in this area - I'm not currently account for UV coordinate deformation, and the weights I chose for position vs normal influence are completely arbitrary. The Nanite presentation talks about this problem a lot - pre-calculating an error amount that perfectly accounts for every aspect of human perception, for meshes with arbitrary materials, is a _really_ hard problem. The best we can do is spend time tweaking heuristics, which I'll leave for a future PR.

## Performance of Bevy 0.14 vs 0.15

(Compare perf but also memory usage, use tables)

## Roadmap
I got a lot done in Bevy 0.15, but there's still a _ton_ left to do for Bevy 0.16 and beyond.

The major, immediate priority (once I'm rested and ready to work on virtual geometry again) will be improving the culling/LOD selection pass. While cluster selection (I should rename the pass to that, that's a good name now that I think of it) is an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem in theory, in practice, having to dispatch a thread per cluster in the scene is an enormous waste of time. There can be million of clusters in the scene, and divergence and register usage on top of the sheer number of threads needed means that this pass is currently the biggest bottleneck.

The fix is to (like Nanite does) traverse a BVH (tree) of clusters, where we only need to process clusters up until they would be the wrong LOD, and then can immediately stop processing their children. Doing tree traversal on a GPU is very tricky, and doing it maximally efficient depends on [undefined behavior](https://arxiv.org/pdf/2109.06132v1) of GPU schedulers that not all GPUs have.

Once I switch to tree traversal for cluster selection, I think (I'm not fully clear on the details yet) I can also get rid of the need for the fill cluster buffers pass entirely, which would let us reclaim even more performance. More crucially, we could do away with the need to allocate buffers to hold instance ID + cluster ID per cluster in the scene, instead letting us store this data per _visible_ (post LOD selection/culling) cluster in the scene. Besides the obvious memory savings, it also saves us from running into the cluster ID limit issue that was limiting our scene size before. We would no longer need a unique ID for each cluster in the scene - just a unique ID for visible clusters only, post tree traversal, which is a much smaller amount.

Besides cluster selection improvements, and improving on existing stuff, other big areas I could work on include:

* Streaming of meshlet vertex data (memory savings)
* Disk-oriented asset compression (disk and load time savings)
* Rendering clusters for all views at once (performance savings for shadow views)
* Material shader optimizations (I haven't spent any time at all on this yet)
* Occlusion culling fixes (I plan to port Hans-Kristian Arntzen's Granite renderer's [HiZ shader](https://github.com/Themaister/Granite/blob/7543863d2a101faf45f897d164b72037ae98ff74/assets/shaders/post/hiz.comp) to WGSL)
* Tooling to make working with MeshletMeshes easier
* Testing and improving CPU performance for large amounts of instances

With any luck, in another few months I'll be writing about some of these topics in the post for Bevy 0.16. See you then!

## Appendix
Further resources on Nanite-style virtual geometry:

* <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf>
* <https://github.com/jglrxavpok/Carrot>
* <https://github.com/LVSTRI/IrisVk>
* <https://github.com/pettett/multires>
* <https://github.com/Scthe/nanite-webgpu>
* <https://github.com/ShawnTSH1229/SimNanite>
* <https://github.com/SparkyPotato/radiance>
* <https://github.com/zeux/meshoptimizer/blob/master/demo/nanite.cpp>

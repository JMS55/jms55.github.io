+++
title = "Virtual Geometry in Bevy 0.15"
date = "2024-10-25"

[taxonomies]
tags = ["bevy", "virtual geometry"]
+++

## Introduction

It's been a little over 4 months [since my last post](@/posts/2024_06_09_virtual_geometry_bevy_0_14/index.md) where I talked about the very early prototype of virtual geometry I wrote for Bevy 0.14.

In the time since, I've written 10 PRs that greatly improve the feature. While it's still not production ready, the version of virtual geometry that will ship in Bevy 0.15 (which is releasing soon) is a very large step in the right direction!

In this blog post I'll be going over the PRs I made in chronological order. At the end, I'll do a performance comparison vs Bevy 0.14, and finally discuss my roadmap for what I'm planning to work on in Bevy 0.16 and beyond.

It's going to be another super long read, so grab some snacks and strap in!

## Arseny Kapoulkine's Contributions
PRs [#13904](https://github.com/bevyengine/bevy/pull/13904), [#13913](https://github.com/bevyengine/bevy/pull/13913), and [#14038](https://github.com/bevyengine/bevy/pull/14038) improve the performance of the Mesh to MeshletMesh converter, and makes it more deterministic. These were written by Arseny Kapoulkine (author of meshoptimizer, the library I use for mesh simplification and meshlet building). Thanks for the contributions!

PR [#14042](https://github.com/bevyengine/bevy/pull/14042), also by Arseny Kapoulkine, fixed a bug with how we calculate the depth pyramid mip level to sample at for occlusion culling.

These PRs were actually shipped in Bevy 0.14, but were opened after I published my last post, hence why I'm covering them now.

## Faster MeshletMesh Loading
PR [#14193](https://github.com/bevyengine/bevy/pull/14193) improves performance when loading MeshletMesh assets from disk. This also shipped in Bevy 0.14, but was written after my last post.

Previously I was using the `bincode` and `serde` crates to serialize and deserialize MeshletMeshes. All I had to do was slap `#[derive(Serialize, Deserialize)]` on the type, and then I could use `bincode::serialize_into()` to turn my asset into a slice of bytes for writing to disk, and `bincode::deserialize_from()` in order to turn a slice of bytes loaded from disk back into my asset type. Easy.

Unfortunately, that ease of use came with a good bit of performance overhead. Specifically in the deserializing step, where bytes get turned into the asset type. Deserializing the 5mb Stanford Bunny asset I was using for testing took a depressingly long 77ms on my Ryzen 5 2600 CPU.

Thinking about the code flow more, we _already_ have an asset -> bytes step. After the asset is loaded into CPU memory, we deserialize it _back_ into bytes so that we can upload it to GPU memory. For this, we use the `bytemuck` crate which provides functions for casting slices of data that are `Pod` (plain-old-data, i.e. just numbers, which all of our asset data is) to slices of bytes, without any real overhead.

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

These two functions are all we need to read and write asset data. `write_slice()` takes a slice of asset data, writes the length of the slice, and then casts the slice to bytes and writes it to disk. `read_slice()` reads the length of the slice from disk, allocates an atomically reference counted buffer of that size, and then reads that many bytes from disk into the buffer, casting it back into the asset data type.

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

This was a sceme I described in my last post, but purposefully avoided due to the performance lost. However, now that we're adding a software rasterizer, I expect that almost all clusters will be software rasterized. Therefore some performance loss for the hardware raster is acceptable. In return, we'll get to use a nice trick in the next step.

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

In the fragment shader, instead of writing to a bound render target, we're now going to do an atomicMax() on a storage buffer to store the rasterized visbuffer result. The reason is that we'll need to do the same for the software rasterization pass (because compute shaders don't have access to render targets), so to keep things simple and reuse the same texture binding and underlying texture state between the rasterization passes, we're going stick to the atomicMax() trick for the hardware rasterization too. The Nanite slides describe this in more detail if you want to learn more.

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

We're expecting to deal with a _lot_ of visible clusters. For each software rasterized cluster, we're going to increment the X dimension of an indirect dispatch, with 1 workgroup per cluster. On some GPUs (mainly AMD), you're limited to 65536 workgroups per dispatch dimension, which is too low. We need to do the same trick we've done in the past of turning a 1d dispatch into a higher dimension dispatch (in this case 2d), and then reformulating it back to a 1d dispatch ID in the shader.

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

Each thread can then iterate the bounding box (either iterating each pixel, or iterating scanlines like Nanite does), writing pixels to the visbuffer as it goes using the same atomicMax() method that we used for hardware rasterization.

One notable difference to the Nanite slides is that for the scanline variant, I needed to check if the pixel center was within the triangle after each pixel in the scanline, which the slides don't show. Not sure if the slides just omitted it for brevity or what, but I got artifacts if I left the check outss.

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

My advice to others working on virtual geometry is to skip software raster until close to the end. If you have mesh shaders, stick with those. From what I've heard from other projects, software raster is only a 10-20% performance improvement over mesh shaders in most scenes, unless you really crank the tiny triangle count (which is, admittedly, a goal, but not an immediate priority).

If like me, you don't have mesh shaders, then I would still probably stick with only hardware rasterization until you've exhausted other, more fundamental areas to work on, like culling and DAG building. However, I would learn from my mistakes, and not spend so much time trying to get hardware rasterization to be fast. Just stick to writing out a list of visible cluster IDs in the culling shader and have the vertex shader ignore extra triangles, instead of trying to get clever with writing out a buffer of visible triangles and drawing the minimal number of vertices. You'll eventually add software rasterization, and then the hardware rasterization performance won't be so important.

If you do want to implement a rasterizer in software (for virtual geometry, or otherwise), check out the below resources that were a big help for me in learning rasterization and the related math.

* <https://kristoffer-dyrkorn.github.io/triangle-rasterizer>
* <https://fgiesen.wordpress.com/2013/02/06/the-barycentric-conspirac>
* <https://www.youtube.com/watch?v=k5wtuKWmV48>

## Larger Meshlet Sizes
PR [#15023](https://github.com/bevyengine/bevy/pull/15023) has a bunch of small improvements to virtual geometry.

The main change is switching from a maximum 64 vertices and 64 triangles to 255 vertices (`64v:64t`) and 128 triangles per meshlet (`255v:128t`). I found that having an equal `v:t` ratio leads to most meshlets having less than `t` triangles, which we don't want. Having a `2v:t` ratio leads to more fully-filled meshlets, and I went with `255v:128t` (which is nearly the same as Nanite, minus the fact that meshoptimizer only supports meshlets with up to 255 vertices) over `128v:64t` after some performance testing.

Note that this change involved some other work, such as adjusting software and hardware raster to work with more triangles, software rasterization looping if needed to load 2 vertices per thread instead of 1, using another bit per triangle ID when packing cluster + triangle IDs to accomodate triangles up to 127, etc.

The other changes I made were:
* Setting the target error to f32::MAX when simplifying a mesh (no point in capping it for continuous LOD, gives better simplification results)
* Adjusting the threshold to allow less-simplified meshes to still count as having been simplified enough (gets us closer to log2(lod_0_meshlet_count) LOD levels)
* Setting `group_error = max(group_error, all_child_errors)` instead of `group_error += max(all_child_errors)` (not really sure if this is more or less correct)

## Screenspace-derived Tangents
PR [#15084](https://github.com/bevyengine/bevy/pull/15084)

https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping
https://jcgt.org/published/0009/03/04

## Vertex Attribute Compression
PR [#15643](https://github.com/bevyengine/bevy/pull/15643)

https://arxiv.org/abs/2404.06359
https://gpuopen.com/download/publications/DGF.pdf
https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html

## Improved LOD Selection Heuristic
PR [#15846](https://github.com/bevyengine/bevy/pull/15846)

## Improved Mesh to MeshletMesh Conversion
PR [#15886](https://github.com/bevyengine/bevy/pull/15886)

## Improved Fill Cluster Buffers Pass
PR [#15955](https://github.com/bevyengine/bevy/pull/15955)

Reference baz

## Software Rasterization Bugfixes
PR [#16049](https://github.com/bevyengine/bevy/pull/16049)

## Normal-aware LOD Selection
PR [#16111](https://github.com/bevyengine/bevy/pull/16111)

## Performance of Bevy 0.14 vs 0.15

(Compare perf but also memory usage)

## Roadmap

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

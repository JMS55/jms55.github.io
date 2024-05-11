+++
title = "Virtual Geometry in Bevy 0.14"
date = "2024-05-10"

[taxonomies]
tags = ["bevy", "virtual geometry"]
+++

# Introduction
Bevy 0.14 releases soon, and with it, the release of an experimental virtual geometry feature that I've been working on for several months.

(TODO: Meshlet image)

In this blog post, I'm going to give a technical deep dive into Bevy's new "meshlet" feature, what improvements it bring, techniques I tried that did or did not work out, and what I'm looking to improve on in the future. There's a lot that I've learned (and a _lot_ of code I've written and rewritten multiple times), and I'd like to share what I learned in the hope that it will help others.

I'd also like to take a moment to thank [LVSTRI](https://github.com/LVSTRI) and [jglrxavpok](https://jglrxavpok.github.io) for their experience with virtual geometry, [atlv24](https://github.com/atlv24) for their help in several areas, especially for their work on wgpu/naga for some missing features I needed, other Bevy developers for testing and reviewing my PRs, Unreal Engine(Brian Karis, Rune Stubbe, Graham Wihlidal) for their _excellent_ and highly detailed [SIGGRAPH presentation](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf), and many more people than I can name who provided advice on the project.

Code for this feature can be found [on my github](https://github.com/JMS55/bevy/tree/cecd7647c8631d2fdbd3eef9c2ce937deb28dec4/crates/bevy_pbr/src/meshlet).

If you're already familiar with Nanite, feel free to skip the next few sections of background info until you get to the Bevy-specific parts.

## Why Virtual Geometry?

Before talking about what virtual geometry _is_, I think it's worth looking at what problems it is trying to _solve_.

Lets go over the high level steps your typical [pre-2015 renderer](https://www.advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf) would perform to render some basic geometry. I've omitted some steps that aren't relevant to this post such as uploading mesh and texture data, shadow map rendering, lighting, and other shading details.

First, on the CPU:
* Frustum culling of instances outside of the camera's view
* Choosing the appropriate level of detail (LOD) for each instance
* Sorting and batching instances into multiple draw lists
* Recording draw calls into command buffers for each draw list

Then, on the GPU:
* Setting up GPU state according to the command buffers
* Transforming vertices and rasterizing triangles
* Depth testing triangle fragments
* Shading visible fragments

Now lets try taking this renderer, and feeding it a dense cityscape made of 500 million triangles, and 150 thousand instances of different meshes.

It's going to be slow. Why? Lets look at some of the problems:
* Frustum culling lets us skip preparing or drawing instances that are outside the camera's frustum, but what if you have an instance that's only partially visible? The GPU still needs to transform, clip, and process all vertices in the mesh. Or, what if the entire scene is in the camera's frustum?
* If one instance is in front of another, it's a complete waste to draw an instance to the screen that will later be completely drawn over by another (overdraw).
* Sorting, batching, and encoding the command buffers for all those instances are going to be slow. Each instance likely has a different vertex and index buffer, different set of textures to bind, different shader (pipeline) for vertex and fragment processing, etc.
* The GPU will spend time spinning down and spinning back up as it switches state between each draw call.

Now, it's no longer 2015, there are a variety of techniques (some from before 2015, that I purposefully left out) to alleviate a lot of these issues. Deferred shading or a depth-only prepass means overdraw is less costly, bindless techniques and ubershaders reduce state switching, multi-draw can reduce draw count, etc.

However, there are some more subtle issues that come up:
* Storing all that mesh data in memory takes too much VRAM. Modern mid-tier desktop GPUs tend to have 8-12 GB of VRAM, which means all your mesh data and 4k textures need to be able to fit in that amount of storage.
* LODs were one of the steps that were meant to help reduce the amount of geometry you were feeding a GPU. However, they come with some downsides: 1) The transition between LODs tends to be noticable, even with a crossfade effect, 2) Artists need to spend time producing and tweaking LODs from their initial high-poly meshes, and 3) Like frustum culling, they don't help with the worst case of simply being close to a lot of high-poly geometry, unless you're willing to cap out at a lower resolution than the artist's original mesh.

There's also another issue I've saved for last. Despite all the culling and batching and LODs, we still have _too much_ geometry to draw every frame. We need a better way to deal with it than simple LODs.

## What is Virtual Geometry?

With the introduction of Unreal Engine 5 in 2021 came the introduction of a new technique called [Nanite](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine). Nanite is a system where you can pre-process your non-deforming opaque meshes, and at runtime be able to very efficiently render them, largely solving the above problems with draw counts, memory limits, high-poly mesh rasterization, and the deficiencies of traditional LODs.

Nanite works by first splitting your base mesh into a series of meshlets - small, independent clusters of triangles. Nanite then takes those clusters, simplifies them to use less triangles, and then groups the clusters themselves into a set of _new_ clusters. By repeating this process, you get a tree of clusters where the leaves of the tree form the base mesh, and the root of the tree forms a simplfied approximation of the base mesh.

Now at runtime, we don't just have to render one level (LOD) of the tree. We can choose specific clusters from different levels of the tree so that if you're close to one part of the mesh, it'll render many high resolution clusters. If you're far from a different part of the mesh, however, then that part will use a couple low resolution clusters that are cheaper to render. Unlike traditional LODs, which are all or nothing, part of the mesh can be low resolution, part of the mesh can be high resolution, and a third part can be somewhere in between, all at the time same time, all on a very granular level.

(TODO: Slide from Nanite showing the cluster tree)

Combine this LOD technique with some per-cluster culling, a visibility buffer, streaming in and out of individual cluster data to prevent high memory usage, a custom rasterizer, and a whole bunch of others parts, and you end up with a renderer that _can_ deal with a scene made of 500 million triangles.

I mentioned before that meshes have to be opaque, and can't deform or animate (for the initial release of Nanite in Unreal Engine 5.0 this is true, but it's an area Unreal is working to improve). Nanite isin't perfect - there are still limitations. But the ceiling of what's feasible is a lot higher.

## Virtual Geometry in Bevy

Now that the background is out of the way, lets talk about Bevy. For Bevy 0.14, I've written an initial implementation that largely copies the basic ideas of how Nanite works, without implementing every single optimization and technique. Currently, the feature is called meshlets. In a minute, I'll get into the actual frame breakdown and code for meshlets, but first lets start with the user-facing API.

Users wanting to use meshlets should compile with the `meshlet` cargo feature at runtime, and `meshlet_processor` cargo feature for pre-processing meshes (again, more on how that works later) into the special meshlet-specific format the meshlet renderer uses.

Enabling the meshlet feature unlocks a new module: `bevy::pbr::experimental::meshlet`.

Users should first call `MeshletMesh::from_mesh()` (again, you need the `meshlet_processor` feature enabled) to pre-process their mesh. This step is _very_ slow, and should be done once ahead of time, and then saved to a meshlet_mesh file. Unfortunately, due to [missing capabilities](https://github.com/bevyengine/bevy/discussions/13078) in Bevy's asset system, there's currently no easy way to just automatically convert a whole GLTF using Bevy's built-in asset processing feature.

Next, add `MeshletPlugin` to your app:

```rust
app.add_plugins(MeshletPlugin);
```

Now, spawn your entities. In the same vein as `MeshMaterialBundle`, there's a `MeshletMeshMaterialBundle`, which uses a `MeshletMesh` instead of the typical `Mesh`.

```rust
commands.spawn(MaterialMeshletMeshBundle {
    meshlet_mesh: meshlet_mesh_handle.clone(),
    material: material_handle.clone(),
    transform: Transform::default().with_translation(Vec3::new(x as f32 / 2.0, 0.0, 0.3)),
    ..default()
});
```

Lastly, a note on materials. Meshlet entities use the same `Material` trait as regular mesh entities. There are 3 new methods that meshlet entities use however: `meshlet_mesh_fragment_shader`, `meshlet_mesh_prepass_fragment_shader`, and `meshlet_mesh_deferred_fragment_shader`. Notice that there is no access to vertex shaders. Meshlet rendering uses a hardcoded vertex shader that cannot be changed.

Fragment shaders for meshlets are mostly the same as fragment shaders for regular mesh entities. The key difference is that instead of this:

```rust
@fragment
fn fragment(vertex_output: VertexOutput) -> @location(0) vec4<f32> {
    // ...
}
```

You should use this:

```rust
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output

@fragment
fn fragment(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let vertex_output = resolve_vertex_output(frag_coord);
    // ...
}
```

# Frame Breakdown

We've finally arrived at the portion of this post where we'll dive into the code and see how everything works under the hood.

The frame capture we'll be looking at is this scene with ~3100 Stanford Bunnies. Five of the bunnies are using unique PBR materials, while the rest use the same debug material that visualizes the clusters/triangles of the mesh. Each bunny is made of (TODO) triangles, with (TODO) clusters at LOD 0.

GPU timings were measured on a RTX 3080 _locked to base clock speeds_, rendering at 1080p.

(TODO: Picture of frame with triangles)

(TODO: Picture of frame with clusters)

(TODO: Picture of NSight screen capture)

## Mesh Conversion

## Fill Cluster Buffers

## Culling (First Pass)

## Raster (First Pass)

## Downsample Depth

## Culling (Second Pass)

## Raster (Second Pass)

## Copy Material Depth

## Downsample Depth (Again)

## Material Draws

# Future Work

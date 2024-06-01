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

I'd also like to take a moment to thank [LVSTRI](https://github.com/LVSTRI) and [jglrxavpok](https://jglrxavpok.github.io) for their experience with virtual geometry, [atlv24](https://github.com/atlv24) for their help in several areas, especially for their work on wgpu/naga for some missing features I needed, other Bevy developers for testing and reviewing my PRs, Unreal Engine (Brian Karis, Rune Stubbe, Graham Wihlidal) for their _excellent_ and highly detailed [SIGGRAPH presentation](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf), and many more people than I can name who provided advice on the project.

Code for this feature can be found [on my github](https://github.com/JMS55/bevy/tree/cecd7647c8631d2fdbd3eef9c2ce937deb28dec4/crates/bevy_pbr/src/meshlet) in the `crates/bevy_pbr/src/meshlet` folder.

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

Nanite works by first splitting your base mesh into a series of meshlets - small, independent clusters of triangles. Nanite then takes those clusters, simplifies them to use less triangles, and then groups the clusters themselves into a set of _new_ clusters. By repeating this process, you get a tree of clusters where the leaves of the tree form the base mesh, and the root of the tree forms a simplified approximation of the base mesh.

Now at runtime, we don't just have to render one level (LOD) of the tree. We can choose specific clusters from different levels of the tree so that if you're close to one part of the mesh, it'll render many high resolution clusters. If you're far from a different part of the mesh, however, then that part will use a couple low resolution clusters that are cheaper to render. Unlike traditional LODs, which are all or nothing, part of the mesh can be low resolution, part of the mesh can be high resolution, and a third part can be somewhere in between, all at the time same time, all on a very granular level.

(TODO: Slide from Nanite showing the cluster tree)

Combine this LOD technique with some per-cluster culling, a visibility buffer, streaming in and out of individual cluster data to prevent high memory usage, a custom rasterizer, and a whole bunch of others parts, and you end up with a renderer that _can_ deal with a scene made of 500 million triangles.

I mentioned before that meshes have to be opaque, and can't deform or animate (for the initial release of Nanite in Unreal Engine 5.0 this is true, but it's an area Unreal is working to improve). Nanite isin't perfect - there are still limitations. But the ceiling of what's feasible is a lot higher.

## Virtual Geometry in Bevy

Now that the background is out of the way, lets talk about Bevy. For Bevy 0.14, I've written an initial implementation that largely copies the basic ideas of how Nanite works, without implementing every single optimization and technique. Currently, the feature is called meshlets (likely to change to virtual_geometry or something else in the future). In a minute, I'll get into the actual frame breakdown and code for meshlets, but first lets start with the user-facing API.

Users wanting to use meshlets should compile with the `meshlet` cargo feature at runtime, and `meshlet_processor` cargo feature for pre-processing meshes (again, more on how that works later) into the special meshlet-specific format the meshlet renderer uses.

Enabling the `meshlet` feature unlocks a new module: `bevy::pbr::experimental::meshlet`.

First step, add `MeshletPlugin` to your app:

```rust
app.add_plugins(MeshletPlugin);
```

Next, preprocess your `Mesh` into a `MeshletMesh`. Currently, this needs to be done manually via  `MeshletMesh::from_mesh()` (again, you need the `meshlet_processor` feature enabled). This step is _very_ slow, and should be done once ahead of time, and then saved to an asset file. Note that there are limitations on the types of meshes and materials supported, make sure to read the docs.

I'm also in the [middle of working on](https://github.com/bevyengine/bevy/pull/13431) an asset processor system to automatically convert entire glTF scenes, but it's not quite ready yet.

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

# Mesh Conversion

We're now going to start the portion of this blog post going into how everything is implemented.

The first step, before we can render anything, is to convert all meshes to meshlet meshes. I talked about the user-facing API earlier on, but in this section we'll dive into what `MeshletMesh::from_mesh()` is doing under the hood in `from_mesh.rs`.

This section will be a bit dry, lacking commentary on why I did things, in favor of just describing the algorithm itself. The reason is that I don't have many unique insights into the conversion process. The steps taken are pretty much just copied from Nanite. If you're interested in understanding it more, definitely check out the original Nanite presentation.

Feel free to skip ahead to the frame breakdown section.

The high level steps for converting a mesh are as follows:
1. Build LOD 0 meshlets
2. For each meshlet, find the set of all edges making up the triangles within the meshlet
3. For each meshlet, find the set of connected meshlets (share an edge)
4. Divide meshlets into groups of roughly 4
5. For each group of meshlets, build a new list of triangles approximating the original group
6. For each simplified group, break them apart into new meshlets
7. Repeat steps 3-7 using the set of new meshlets, unless we ran out of meshlets to simplify

(TODO: Picture of algorithm)

## Build LOD 0 Meshlets

We're starting with a generic triangle mesh, so the first step is to group its triangles into an initial set of meshlets. No simplification or modification of the mesh is involved - we're simply splitting up the original mesh into a set meshlets that would render exactly the same.

The crate `meshopt-rs` provides Rust bindings to the excellent `meshoptimizer` library, which provides a nice `build_meshlets()` function for us.

```rust
// Split the mesh into an initial list of meshlets (LOD 0)
let vertex_buffer = mesh.get_vertex_buffer_data();
let vertex_stride = mesh.get_vertex_size() as usize;
let vertices = VertexDataAdapter::new(&vertex_buffer, vertex_stride, 0).unwrap();
let mut meshlets = compute_meshlets(&indices, &vertices);
```

We also need some bounding spheres for each meshlet. The culling bounding sphere is straightforward - `compute_meshlet_bounds()`, again from `meshopt-rs`, will give us a bounding sphere encompassing the meshlet that we can use for frustum and occlusion culling later on.

The `self_lod` and `parent_lod` bounding spheres need a lot more explanation.

As we simplify each group of meshlets into new meshlets, we will deform the mesh slightly. That deformity adds up over time, eventually giving a very visibly different mesh from the original. However, when viewing the very simplified mesh from far away, due to perspective the difference will be much less noticable. While we would want to view the original (or close to the original) mesh close-up, at longer distances we can get away with rendering a much simpler version of the mesh.

So, how to choose the right LOD level, or in our case, the right LOD tree cut? The LOD cut will be based on the simplification error of each meshlet along the cut, with the goal being to select a cut that is impercetibly different from the original mesh at the distance we're viewing the mesh at.

For reasons I'll get into later, we're going to treat the error as a bounding sphere around the meshlet, with the radius being the error. We're also going to want two of these: one for the current meshlet itself, and one for the less-simplified version of the meshlet that we simplified into the current meshlet (the current meshlet's parent in the LOD tree).

LOD 0 meshlets, being the original representation of the mesh, have no error (0.0). They also have no parent meshlet, which we will represent with an infinite amount of error (f32::MAX), again for reasons I will get into later.

```rust
let mut bounding_spheres = meshlets
    .iter()
    .map(|meshlet| compute_meshlet_bounds(meshlet, &vertices))
    .map(convert_meshlet_bounds)
    .map(|bounding_sphere| MeshletBoundingSpheres {
        self_culling: bounding_sphere,
        self_lod: MeshletBoundingSphere {
            center: bounding_sphere.center,
            radius: 0.0,
        },
        parent_lod: MeshletBoundingSphere {
            center: bounding_sphere.center,
            radius: f32::MAX,
        },
    })
    .collect::<Vec<_>>();
```

## Find Meshlet Edges

Now that we have our initial set of meshlets, we can start simplifying.

The first step is to find the set of triangle edges that make up each meshlet. This can be done with a simple loop over triangles, building a hashset of edges where each edge is ordered such that the smaller numbered vertex comes before the larger number vertex. This ensures that we don't accidently add both (v1, v2) and (v2, v1), which conceptually are the same edge. Each triangle has 3 vertices and 3 edges.

```rust
let mut meshlet_triangle_edges = HashMap::new();
for i in meshlet.triangles.chunks(3) {
    let v0 = meshlet.vertices[i[0] as usize];
    let v1 = meshlet.vertices[i[1] as usize];
    let v2 = meshlet.vertices[i[2] as usize];

    meshlet_triangle_edges.insert((v0.min(v1), v0.max(v1)));
    meshlet_triangle_edges.insert((v0.min(v2), v0.max(v2)));
    meshlet_triangle_edges.insert((v1.min(v2), v1.max(v2)));
}
```

## Find Connected Meshlets

Next, we need to find the meshlets that connect to each other.

A meshlet will be considered as connected to another meshlet if both meshlets share at least one edge.

In the previous step, we built a set of edges for each meshlet. Finding if two meshlets share any edges can be done by simply taking the intersection of their two edge sets, and checking if the resulting set is not empty.

We will also store the _amount_ of shared edges between two meshlets, giving a heuristic for how "connected" each meshlet is to another. This is simply the size of the intersection set.

Overally, we will build a list per meshlet, containing tuples of (meshlet_id, shared_edge_count) for each meshlet connected to the current meshlet.

```rust
for (meshlet_id1, meshlet_id2) in simplification_queue.tuple_combinations() {
    let shared_edge_count = triangle_edges_per_meshlet[&meshlet_id1]
        .intersection(&triangle_edges_per_meshlet[&meshlet_id2])
        .count();

    if shared_edge_count != 0 {
        connected_meshlets_per_meshlet
            .get_mut(&meshlet_id1)
            .unwrap()
            .push((meshlet_id2, shared_edge_count));
        connected_meshlets_per_meshlet
            .get_mut(&meshlet_id2)
            .unwrap()
            .push((meshlet_id1, shared_edge_count));
    }
}
```

## Partition Meshlets Into Groups

Now that we know which meshlets are connected, the next step is to group them together. We're going to aim for 4 meshlets per group, although there's no way of guaranteeing that.

How should we determinewhich meshlets go in which group?

You can view the connected meshlet sets as a graph. Each meshlet is a node, and bidirectional edges connect one meshlet to another in the graph if we determined that they were connected earlier. The weight of each edge is the amount of shared edges between the two meshlet nodes.

Partitioning the meshlets into groups is now a matter of partitioning the graph. I use the `metis-rs` crate which provides Rust bindings to the `METIS` library. The edge weights will be used so that meshlets with a high shared edge count are more likely to be group together.

The code to format this data for metis is a bit complicated, but in the end we have a list of groups, where each group is a list of meshlets.

## Simplify Groups

Now for an important step, and the most tricky.

We take each group, and merge the triangles of the underlying meshlets together into one large list of triangles, forming a new mesh.

Now, we can simplify this new mesh into a lower-resolution (faster to render) version. Meshopt again provides a helpful `simplify()` function for us. Finally, less triangles to render!

In addition to the new mesh, we get an "error" value, describing how much the mesh deformed by when simplifying.

The quadratic error metric (QEM) returned from simplifying is a somewhat meaningless value, but we can use `simplify_scale()` to get an object-space value. This value is _still_ fairly meaningless, but we can treat it as the maximum amount of object-space distance a vertex displaced by during simplification.

The error represents displacement from the meshlets we simplified, but we want the displacement from the original (LOD 0) meshlets. We can add the max error of the meshlets that went into building the current meshlet group (child nodes of the parent node that we're currently building in the LOD tree) to make the error relative to LOD 0.

If this all feels handwavy to you, that's because it is. And this is vertex positions only; we haven't even considered UV error during simplification, or how the mesh's eventual material influences perceptual differences between LOD levels. Perceptual simplification is an unsolved problem in computer graphics, and for now Bevy only uses positions for simplification.

You'll have to take my word for it that using the error like this works. You'll see how it gets used to pick the LOD level during runtime in a later section. For now, we'll take the group error and build a bounding sphere out of it, and assign it as the parent LOD bounding sphere for the group's (parent node, higher LOD) underlying meshlets (child nodes, lower LOD).

```rust
// Simplify the group to ~50% triangle count
let Some((simplified_group_indices, mut group_error)) =
    simplify_meshlet_groups(group_meshlets, &meshlets, &vertices, lod_level)
else {
    continue;
};

// Add the maximum child error to the parent error to make parent error cumulative from LOD 0
// (we're currently building the parent from its children)
group_error += group_meshlets.iter().fold(group_error, |acc, meshlet_id| {
    acc.max(bounding_spheres[*meshlet_id].self_lod.radius)
});

// Build a new LOD bounding sphere for the simplified group as a whole
let mut group_bounding_sphere = convert_meshlet_bounds(compute_cluster_bounds(
    &simplified_group_indices,
    &vertices,
));
group_bounding_sphere.radius = group_error;

// For each meshlet in the group set their parent LOD bounding sphere to that of the simplified group
for meshlet_id in group_meshlets {
    bounding_spheres[*meshlet_id].parent_lod = group_bounding_sphere;
}
```

## Split Groups

Finally, the last step is to take the large mesh formed from simplifying the entire meshlet group, and split it into a set of brand new meshlets.

This is in fact the same process as splitting the original mesh into meshlets.

If everything went optimally, we should have gone from the original 4 meshlets per group, to 2 new meshlets per group with 50% less triangles overall.

For each new meshlet, we'll calculate a bounding sphere for culling, assign the self_lod bounding sphere as that of the group,and the parent_lod bounding sphere again as uninitialized.

```rust
// Build new meshlets using the simplified group
let new_meshlets_count = split_simplified_groups_into_new_meshlets(
    &simplified_group_indices,
    &vertices,
    &mut meshlets,
);

// Calculate the culling bounding sphere for the new meshlets and set their LOD bounding spheres
let new_meshlet_ids = (meshlets.len() - new_meshlets_count)..meshlets.len();
bounding_spheres.extend(
    new_meshlet_ids
        .map(|meshlet_id| {
            compute_meshlet_bounds(meshlets.get(meshlet_id), &vertices)
        })
        .map(convert_meshlet_bounds)
        .map(|bounding_sphere| MeshletBoundingSpheres {
            self_culling: bounding_sphere,
            self_lod: group_bounding_sphere,
            parent_lod: MeshletBoundingSphere {
                center: group_bounding_sphere.center,
                radius: f32::MAX,
            },
        }),
);
```

We can repeat this whole process several times, ideally getting down to a single meshlet forming the root of the LOD tree. In practice, we can't currently get to that point for most meshes.

# Frame Breakdown

With the asset processing part out of the way, we can finally move onto the more interesting runtime code section.

The frame capture we'll be looking at is this scene with (TODO) Stanford Bunnies. Five of the bunnies are using unique PBR materials, while the rest use the same debug material that visualizes the clusters/triangles of the mesh. Each bunny is made of (TODO) triangles, with (TODO) meshlets at LOD 0, and (TODO) meshlets total in the LOD tree.

GPU timings were measured on a RTX 3080 locked to base clock speeds, rendering at 1080p.

(TODO: Picture of frame with triangles)

(TODO: Picture of frame with clusters)

(TODO: Picture of NSight screen capture)

## Terminology

First, some terminology:
* `asset buffers` - When a new MeshletMesh asset is loaded, we copy the buffers it's made of into large suballocated buffers. All the vertex data, meshlet data, bounding spheres, etc for multiple MeshletMesh assets are packed together into one large buffer per type.
* `instance` - A single Bevy entity with a MeshletMesh and Material.
* `instance uniform` - A transform matrix and mesh flags for an instance.
* `material` - A combination of pipeline and bind group used for shading fragments.
* `meshlet` - A single meshlet from within a MeshletMesh asset, pointing to data within the asset buffers (more or less).
* `cluster` - A single render-able piece of an entity. Each cluster is associated with an instance and a meshlet.
    * All of our shaders will operate on clusters, and _not_ on meshlets. You can think of these like an instance of a meshlet for a specific entity, in the same way you can have an instance of a class in object-oriented programming languages.
    * Up to this point I've been using meshlet and cluster interchangeably. From now on, they have seperate, defined meanings.
* `view` - A perspective or orthographic camera with an associated depth buffer and optional color output. The main camera is a view, and additional views are dynamically generated for rendering shadowmaps.
* `id` - A u32 index into a buffer.

## Fill Cluster Buffers

Now the first pass we're going to look at might be surprising.

Over the course of the frame, for each cluster we will need its instance (giving us a transform and material), along with its meshlet (giving us vertex data and bounding spheres).

While the cluster itself is implicit (each thread or workgroup of a shader will handle one cluster, with the global thread/workgroup ID being the cluster ID), we need some method of telling the GPU what the instance and meshlet for each cluster is.

I.e., we need an array of instance IDs and meshlet IDs such that we can do `let cluster_instance = instances[cluster_instance_ids[cluster_id]]` and `let cluster_meshlet = meshlets[cluster_meshlet_ids[cluster_id]]`.

The naive method would be to simply write out these two buffers from the CPU and transfer them to the GPU. This how things initially worked, and it worked fine for my simple initial test scene with a single bunny, but I very quickly ran into performance problems when trying to scale up to render 3000 bunnies.

Each ID is a 4-byte u32, and it's two IDs per cluster. That's 8 bytes per cluster.

With (TODO) bunnies in the scene, and (TODO) meshlets per bunny, that's 8 * (TODO) * (TODO) bytes total = (TODO) MB/GB total.

For dedicated GPUs, uploading data from the system's RAM to the GPU's VRAM is done over PCIe. PCIe x16 Gen3 max bandwidth is 16 GB/s.

Ignoring data copying costs and other overhead, and assuming max PCIe bandwidth, that would mean it would take (TODO)ms to upload cluster data. That's (TODO) / 16.6 = (TODO)% of our frame budget gone, before we've even rendered anything! Obviously, we need a better method.

---

Instead of uploading per-cluster data, we're going to stick to uploading only per-instance data. Specifically, two buffers called `instance_meshlet_counts_prefix_sum`, and `instance_meshlet_slice_starts`.

The former will contain a prefix sum (calculated on the CPU while writing out the buffer) of how many meshlets each instance is made of. The latter will contain the index of where in the meshlet asset buffer each instance's list of meshlets begin.

(TODO: Example scene and buffer contents)

We're uploading only 8 bytes per _instance_, and not per _cluster_, which is much, much cheaper. (TODO: Math for how much data we uploaded for the X instances in the frame we're looking at)

Now that the GPU has this data, we can have the GPU write out the `cluster_instance_ids` and `cluster_meshlet_ids` buffers from a compute shader. Max VRAM<->VRAM bandwidth on my RTX 3080 is a whopping 760.3 GB/s; ~47.5x faster than the 16 GB/s of bandwidth we had over PCIe.

Each thread of the compute shader will handle one cluster, and do a binary search over the prefix sum array to find to what instance it belongs to.

Binary search might seem surprising - it's multiple dependent divergent memory accesses within a thread. However, it's very coherent _across_ threads within the subgroup, and scales extremely well (O log n) with the number of instances in the scene. In practice, while it could be improved, the performance of this pass has not been a bottleneck.

Now that we know what instance the cluster belongs to, it's trivial to calculate the meshlet index of the cluster within the instance's meshlet mesh asset. Adding that to the instance's meshlet_slice_start using the other buffer we uploaded gives us the global meshlet index within the overall meshlet asset buffer. The thread can then write out the two calculated IDs for the cluster.

This is the only pass that runs once per-frame. The rest of the passes all run once per-view.

```rust
/// Writes out instance_id and meshlet_id to the global buffers for each cluster in the scene.

@compute
@workgroup_size(128, 1, 1) // 128 threads per workgroup, 1 cluster per thread
fn fill_cluster_buffers(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>
) {
    // Calculate the cluster ID for this thread
    let cluster_id = local_invocation_id.x + 128u * dot(workgroup_id, vec3(num_workgroups.x * num_workgroups.x, num_workgroups.x, 1u));
    if cluster_id >= cluster_count { return; }

    // Binary search to find the instance this cluster belongs to
    var left = 0u;
    var right = arrayLength(&meshlet_instance_meshlet_counts_prefix_sum) - 1u;
    while left <= right {
        let mid = (left + right) / 2u;
        if meshlet_instance_meshlet_counts_prefix_sum[mid] <= cluster_id {
            left = mid + 1u;
        } else {
            right = mid - 1u;
        }
    }
    let instance_id = right;

    // Find the meshlet ID for this cluster within the instance's MeshletMesh
    let meshlet_id_local = cluster_id - meshlet_instance_meshlet_counts_prefix_sum[instance_id];

    // Find the overall meshlet ID in the global meshlet buffer
    let meshlet_id = meshlet_id_local + meshlet_instance_meshlet_slice_starts[instance_id];

    // Write results to buffers
    meshlet_cluster_instance_ids[cluster_id] = instance_id;
    meshlet_cluster_meshlet_ids[cluster_id] = meshlet_id;
}
```

## Culling (First Pass)

TODO

## Raster (First Pass)

TODO

## Downsample Depth

TODO

## Culling (Second Pass)

TODO

## Raster (Second Pass)

TODO

## Copy Material Depth

TODO

## Downsample Depth (Again)

TODO

## Material Shading

TODO

# Future Work

TODO
* Conversion improvements (vertex welding, distance heuristic, attribute-aware simplification, performance improvements)
* Per-instance LOD process / persistent culling to eliminate the brute-force dispatch over every possible cluster, and the fill clusters step
* Implicit tangents and octahedral-encoded normals
* Software raster and mesh shaders
* Compute-based material shading with VRS (link to Nanite's new presentation)
* Streaming

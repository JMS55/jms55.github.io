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

## Zeux (TODO)
* https://github.com/bevyengine/bevy/pull/13904
* https://github.com/bevyengine/bevy/pull/13913
* https://github.com/bevyengine/bevy/pull/14038
* https://github.com/bevyengine/bevy/pull/14042

## Faster MeshletMesh Loading
PR [#14193](https://github.com/bevyengine/bevy/pull/14193) improves performance when loading MeshletMesh assets from disk. This actually shipped in Bevy 0.14, but was written after publishing my last post, so I'm going to cover it now.

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

The buffer used a lot of memory (4 bytes per non-culled triangle). The GPU's primitive assembler can't keep up with the sheer number of triangles as we're not using indexed triangles (to save 8 bytes of memory and time spent on writing out an index buffer), and therefore lack a vertex cache. And finally the GPU's rasterizer just performs poorly with small triangles, and we have a _lot_ of small triangles.

Current GPU rasterizers expect comparatively few triangles that each cover many pixels. They have performance optimizations aimed at that kind of workload like shading 2x2 quads of pixels at a time and tile binning of triangles. Meanwhile, our virtual geometry renderer is aimed at millions of tiny triangles that only cover a pixel each. We need a rasterizer aimed at being efficient over the number of triangles; not the number of covered pixels per triangle.

### The Software Rasterizer

TODO

https://kristoffer-dyrkorn.github.io/triangle-rasterizer
https://fgiesen.wordpress.com/2013/02/06/the-barycentric-conspirac
https://www.youtube.com/watch?v=k5wtuKWmV48

## Larger Meshlet Sizes
PR [#15023](https://github.com/bevyengine/bevy/pull/15023)

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

## Roadmap

## Appendix
Here are some other resources on nanite-style virtual geometry:

* <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf>
* <https://github.com/jglrxavpok/Carrot>
* <https://github.com/LVSTRI/IrisVk>
* <https://github.com/pettett/multires>
* <https://github.com/Scthe/nanite-webgpu>
* <https://github.com/ShawnTSH1229/SimNanite>
* <https://github.com/SparkyPotato/radiance>
* <https://github.com/zeux/meshoptimizer/blob/master/demo/nanite.cpp>

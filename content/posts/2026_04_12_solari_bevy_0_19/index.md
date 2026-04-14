+++
title = "Realtime Raytracing in Bevy 0.19 (Solari)"
date = "2026-04-12"

[taxonomies]
tags = ["bevy", "raytracing"]
+++

## Introduction

Welcome back to the third installment of [my series](/tags/raytracing) on the development of Solari, Bevy's realtime pathtraced renderer! Bevy 0.19 is fast approaching, so let's talk about what I've been working on.

Compared to 0.18, this cycle was lighter on big features. Most of my time went into polishing existing aspects of the renderer, rather than writing new techniques. I spent a lot of time fixing Solari's material BRDF, improving specular quality, and yet again improving the world cache.

I also spent a significant chunk of this cycle researching light sampling techniques to improve NEE, though that work is still ongoing, and hasn't produced any concrete PRs yet. More on that later.

{{ figure(src="headline_many_lights.png", caption="A scene lit purely by 441 emissive sphere meshes, stress testing light sampling.") }}

Here's a summary of what's changed in Solari for Bevy 0.19:

* **Specular improvements**: dedicated mirror BRDF, primary surface replacement, better path termination
* **BRDF correctness**: proper layering for non-metals, Fresnel-based lobe selection, invalid sample rejection
* **World cache**: light leak reduction and stochastic updates
* **ReSTIR**: better spatial sample finding, skip GI for smooth metals

Before getting into the details, let's recap where we left off.

## Recap of 0.18

The major feature of Solari 0.18 was added support for specular materials. We added a separate specular GI pass that does 0-3 bounce pathtracing by sampling the GGX lobe of the BRDF via bounded VNDF sampling. We also fixed a major energy loss bug in light tile packing, reduced ReSTIR resampling bias, and made the world cache more reactive and much more performant on large scenes like Bistro.

However, the initial specular implementation was rough around the edges. During the 0.19 cycle I noticed that mirrors had artifacts, and our BRDF had correctness issues with non-metals. The world cache also still had light leak problems.

This cycle, I spent a lot of time fixing or mitigating these issues.

## Specular Improvements

### Mirror BRDF

In Bevy 0.18, perfect mirrors were handled by clamping roughness to 0.001, and then special-casing the GGX VNDF sampling to reflect perfectly at that threshold. This worked for sampling, but the microfacet BRDF evaluation was never designed to handle near-zero roughness — it produced black line artifacts on mirror surfaces.

{{ figure(src="mirror_artifacts_before.png", caption="Black line artifacts (bottom right) from the hacked microfacet BRDF") }}
{{ figure(src="mirror_artifacts_after.png", caption="Fixed mirrors with the dedicated mirror BRDF") }}

The fix was to stop trying to hack the microfacet BRDF, and instead introduce a proper mirror BRDF. Roughness is no longer clamped to a minimum — true zero-roughness materials are now allowed.

The mirror BRDF is simple in theory, but a little tricky to implement correctly: it evaluates to the Fresnel term when the half-vector is aligned with the normal, and zero otherwise. Its PDF is INF (a delta distribution) when aligned, and zero otherwise.

When sampling the mirror BRDF for importance sampling a direction for the next bounce, you need to avoid dividing by the INF PDF. But when doing MIS for NEE/emissive hits, you _do_ want to use the INF PDF. I also had to change my balance heuristic code to handle infinite values correctly:

```rust
// Old balance heuristic
fn balance_heuristic(f: f32, g: f32) -> f32 {
    let sum = f + g;
    if sum == 0.0 {
        return 0.0;
    }
    return max(0.0, f / sum);
}

// New balance heuristic - handles INF
fn balance_heuristic(f: f32, g: f32) -> f32 {
    // ReSTIR reservoirs can have UCW=0, but you can remove this if you're not doing ReSTIR
    if f == 0.0 {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}
```

As part of this change, cos_theta (the `NdotL` term from the rendering equation) was also moved into the BRDF evaluation functions themselves. The mirror BRDF doesn't use cos_theta (it cancels out for a perfect reflection), so having it outside the BRDF was causing incorrect results. The diffuse and specular BRDF functions now each include their own cos_theta handling internally.

### Primary Surface Replacement

One of the open issues from 0.18 was that mirror reflections had denoising artifacts — cross-hatching patterns when the camera was static, and ghosting when objects in reflections moved.

{{ figure(src="mirror_denoising_before.png", caption="Cross-hatching artifacts without PSR") }}

The root cause is that DLSS Ray Reconstruction uses guide buffers (normals, depth, motion vectors) from the primary surface to denoise. But for a mirror, the primary surface is just a flat reflective plane — the denoiser has no information about the geometry visible _in_ the reflection.

To fix this, I implemented primary surface replacement (PSR), a technique described in NVIDIA's ["Rendering Perfect Reflections and Refractions in Path-Traced Games"](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games).

The idea is to trace through mirror bounces until we hit a non-mirror surface, then replace the primary surface's guide buffer data with that surface's data. The denoiser then sees the actual reflected geometry instead of the flat mirror.

{% note() %}
Currently, I only apply PSR to smooth metals. I didn't get great results from applying it to smooth non-metals.
{% end %}

The tricky part is computing correct motion vectors for the reflected surface. We need to account for the mirror transformations, which means chaining [Householder reflection matrices](https://en.wikipedia.org/wiki/Householder_transformation) across each mirror bounce:

```rust
fn reflection_matrix(plane_normal: vec3f) -> mat3x3<f32> {
    let n_nt = mat3x3<f32>(
        plane_normal * plane_normal.x,
        plane_normal * plane_normal.y,
        plane_normal * plane_normal.z,
    );
    let identity_matrix = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    return identity_matrix - n_nt * 2.0;
}
```

For each mirror bounce, we multiply the accumulated rotation matrix by the reflection matrix of that surface's normal. When we finally hit a non-mirror surface, we use this accumulated matrix to transform the hit position into a "virtual" position as seen through the mirror chain, and compute motion vectors from that.

{% note() %}
This technique assumes a flat surface, and is incorrect for curved mirrors like a sphere.
{% end %}

{{ figure(src="mirror_denoising_after.png", caption="Improved mirror reflections with PSR") }}

In practice, while PSR itself gave massive improvements, I haven't really noticed any benefit from implementing specular motion vectors. More testing is needed; DLSS-RR is a little picky, and expects guide buffers to be calculated in the same way it was trained on — I could just be implementing things in a way DLSS-RR does not like.

### Path Termination

In 0.18, the specular GI pass decided when to terminate a path in the world cache based on the roughness of the current surface. If the surface was rough enough, we'd query the cache instead of tracing further.

While this works, it's a little naive. A series of semi-rough bounces is basically equivalent to one very rough bounce.

The fix was to switch to a path termination heuristic inspired by [RTXGI](https://github.com/NVIDIAGameWorks/RTXGI), which accounts for the cumulative spread of the full path. Path spread can be approximated by a cone based on the roughness of the surface a ray is coming from, and the distance the ray traveled. When the base of the cone is larger than the size of the world cache cell we're sampling from, it's safe to terminate into the world cache.

{{ figure(src="path_spread.svg", caption="Cumulative path spread grows with each bounce until it exceeds the world cache cell size, triggering termination") }}

This improves performance by terminating paths sooner when appropriate. However, there are still visible artifacts when transitioning between LOD levels of the world cache — the lighting can pop or shift as the path termination crosses a LOD boundry. Unlike ReSTIR GI for diffuse, there's no temporal reservioirs to hide the transition. Smoothing out these transitions is still an open problem.

## BRDF Correctness

In 0.18, Solari's BRDF for non-metals simply added the diffuse and specular lobe contributions together. I assumed that's all there was to it — evaluate each lobe independently, sum the results, done.

This is what [Filament](https://google.github.io/filament/main/filament.html) does (which is what Bevy's BRDF is based on), but it's wrong. In reality, the specular lobe sits _on top of_ the diffuse lobe as a separate layer. Light has to pass through the specular layer to reach the diffuse layer, and then pass through it again on the way out. The Fresnel equations govern how much light is reflected vs transmitted at each crossing. Ignoring this means the BRDF produces too much energy — both lobes claim the full incoming light, when they should be sharing it.

{{ figure(src="brdf_before.png", caption="Incorrect BRDF code rendered via Solari's pathtracer") }}

Fixing this was the biggest correctness change of the cycle, and I owe a big thanks to [Matt Kielan](https://www.linkedin.com/in/matt-kielan-9b054a165) for guiding me through this (I spent a _lot_ of time trying to figure out what I was doing wrong before I learned about this), along with the INF-handling MIS code from earlier.

### Non-Metal Layering

The fix is two Fresnel weighings — one for light entering the specular layer, and one for light exiting back out:

```rust
fn evaluate_diffuse_brdf(
    wo: vec3<f32>,
    wi: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
) -> vec3<f32> {
    let diffuse_color = calculate_diffuse_color(
        material.base_color, material.metallic, 0.0, 0.0,
    ) / PI;

    let NdotL = dot(world_normal, wi);
    let NdotV = dot(world_normal, wo);
    if NdotL < 0.0001 || NdotV < 0.0001 { return vec3(0.0); }

    let F0 = calculate_F0(
        material.base_color, material.metallic, vec3(material.reflectance),
    );
    let layering = (1.0 - fresnel(F0, NdotL)) * (1.0 - fresnel(F0, NdotV));

    return diffuse_color * layering * NdotL;
}
```

The `(1 - F)` terms ensure that energy reflected by the specular layer doesn't also contribute to the diffuse layer. At grazing angles, where Fresnel reflection is strong, the diffuse contribution naturally darkens.

Note the `fresnel()` call — previously Solari imported this from `bevy_pbr`, but Bevy's standard PBR pipeline does some non-physical things with f90 for ambient occlusion purposes. For a pathtracer, we want the straightforward Schlick approximation, so Solari now has its own:

```rust
fn fresnel(f0: vec3<f32>, LdotH: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - LdotH, 5.0);
}
```

### Lobe Selection in the Pathtracer

The same Fresnel logic also applies to importance sampling. The reference pathtracer now picks between the diffuse and specular lobes based on how much energy Fresnel reflects at the current viewing angle:

```rust
let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
let df = 1.0 - luminance(fresnel(F0, NdotV));
let diffuse_weight = mix(df, 0.0, material.metallic);
let specular_weight = 1.0 - diffuse_weight;
```

At grazing angles, most energy goes to the specular lobe, so the pathtracer samples it more often. At normal incidence, more energy goes to the diffuse lobe.

{{ figure(src="brdf_after.png", caption="Correct BRDF code rendered via Solari's pathtracer") }}

### VNDF Sample Validation

I also discovered that the bounded VNDF sampling code could produce invalid samples — directions below the surface hemisphere. I hadn't read the paper thoroughly enough to realize this was even possible, and was blindly using whatever direction it gave back.

The fix is a simple validity check:

```rust
// Result is invalid when output.z <= 0.0 or is NaN, and must be discarded
fn ggx_vndf_sample_invalid(ray_tangent: vec3<f32>) -> bool {
    return !(ray_tangent.z > 0.0);
}
```

Invalid samples now get discarded at every call site — returning zero radiance in specular GI, breaking out of multi-bounce loops, and so on. NaN guards were also added to the PDF calculation (used for MIS).

After all these changes, we're still not passing the [white furnace test](https://lousodrome.net/blog/light/2023/10/21/the-white-furnace-test), but we're [getting closer](https://github.com/bevyengine/bevy/pull/23737)!

If you find BRDF math fun to read about, [OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/) and [Unreal's Substrate](https://advances.realtimerendering.com/s2023/2023%20Siggraph%20-%20Substrate.pdf) have a ton of detailed info on practical layered PBR models.

## World Cache Improvements

### Light Leak Reduction

Light leaks near corners of objects have been one of Solari's most visible artifacts. They occur when a world cache cell spans across a surface boundary — the cell contains irradiance from the other side of the wall, which bleeds through.

The key insight is that light leaks tend to happen when the ray querying the cache is shorter than the cache cell size. In those cases, the queried cell is likely spanning across geometry.

The fix is to force the finest LOD when `ray_t < cell_size`:

```rust
var cell_size = get_cell_size(world_position, view_position);

#ifdef WORLD_CACHE_FIRST_BOUNCE_LIGHT_LEAK_PREVENTION
    if ray_t < cell_size {
        cell_size = WORLD_CACHE_POSITION_BASE_CELL_SIZE;
    }
#endif
```

{% note() %}
As an optimization, I only apply this to the first bounce of ReSTIR GI. Queries as part of world cache update continue to use a coarser LOD.
{% end %}

Combined with making LOD transitions less aggressive and reducing the finest LOD cell size (from 0.25m to 0.15m), this greatly reduces light leaks near corners. It doesn't fix light leaks entirely (especially since I turned world cache jitter back on this cycle), but it helps.

### Stochastic Updates

The finer resolution helps with quality, but it also means more active cells, which hurts performance a fair amount.

{{ figure(src="bistro_all_cells_update.png", caption="2.65ms to update every world cache cell in Bistro") }}

To compensate, we now randomly skip updating cells once the count exceeds 40,000:

```rust
if rand_f(&rng) >= f32(WORLD_CACHE_CELL_UPDATES_SOFT_CAP) / f32(world_cache_active_cells_count) {
    return;
}
```

Stochastic world cache updates keeps the performance hit down in larger scenes, at the cost of less reactive lighting.

{{ figure(src="bistro_stochastic_cells_update.png", caption="0.47ms to update a random 40,000 world cache cells in Bistro") }}

I'm overall not super happy with the world cache, and plan to continue iterating on it, or explore alternative solutions.

As part of this change I also split the DI and GI world cache sampling steps into separate compute dispatches, which gave a small performance win.

## ReSTIR Improvements

### Improved Spatial Sampling

This cycle I also improved the algorithm for finding valid spatial samples from neighboring pixels during ReSTIR DI and GI spatial resampling. In areas with challenging geometry like curved surfaces or cracks between meshes, pixels were unable to find a valid neighbor, hurting quality.

{{ figure(src="old_spatial_resampling.png", caption="Pink pixels have no valid spatial sample (before)") }}

The fix was pretty simple: Although we only pick 1 spatial sample, we attempt to find a valid neighbor up to 5 times. Before we were using a fixed sampling radius, but now after every failed attempt, we halve the sampling radius.

{{ figure(src="new_spatial_resampling.png", caption="Far fewer pixels missing spatial samples (after)") }}

### Skip ReSTIR GI for Smooth Metals

Finally, a simple optimization: smooth metallic surfaces have zero diffuse contribution, so there's no point running ReSTIR GI for those pixels. We now skip these pixels entirely during the ReSTIR GI pass, saving performance on scenes with lots of metallic materials.

## Light Sampling Research

Besides the BRDF work, the other major time sink this cycle was researching better light sampling. Currently, Solari's light sampling is brute force — we take completely random samples from the scene's lights and use RIS to pick one that is likely to contribute the most (not counting visibility). This works, but it's very inefficient, especially for large scenes with many lights, where most of them don't contribute to a given pixel.

{{ figure(src="32_sample_ris.png", caption="Many lights scene, direct lighting only, no denoising, 32 random sample RIS") }}

The obvious question is: can we do better by precomputing which lights matter _where_?

### What Didn't Work

A bit ago, @SparkyPotato experimented with [light trees](https://gpuopen.com/download/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf) — hierarchical acceleration structures that let you importance sample lights based on their estimated contribution. While theoretically sound (light trees are used a lot in the film industry), the overhead of constructing and traversing the tree ended up being too high for our use case.

I explored building alias tables over light CDFs, built on top of Solari's existing world cache, which would give O(1) sampling from an theoretically high quality distribution. But building the alias table for every cell each frame turned out to be expensive, and the quality gains didn't justify the cost.

I also wanted to avoid methods like [ReGIR](https://research.nvidia.com/labs/rtr/publication/boksansky2021rendering), which requires a lot of storage and computation to get good results, and relies heavily on temporal reuse.

A key finding from my early experiments is that stochastic methods are best left to screen-space, where multiple pixels (samples) can contribute towards the same distribution (e.g. by splatting per-pixel contributions into world-space). Otherwise, each world-space cell requires quite a lot of computation per-frame to get usable results.

### Light Grids

The approach I've been exploring instead is world-space light grids — essentially the raytracing equivalent of the froxel light binning that traditional rasterized renderers use (including Bevy's standard renderer). The idea is straightforward: divide the world into cells, figure out which lights contribute to each cell, and then only sample from those lights in a given cell when shading.

This is the same general approach used by recently shipped titles like [DOOM: The Dark Ages](https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf) and [Resident Evil Requiem](https://www.youtube.com/watch?v=Gq1QdlQSzn0).

I have a working prototype where a fixed grid covers the world, with a compute shader that runs one thread per cell to iterate over every light, and determine if it contributes.

It's not quite as good as algorithms that store the importance of each light like ReGIR, but it's quite cheap to build, and does not rely on any temporal reuse or stochastic evaluation.

The results are pretty good!

{{ figure(src="32_sample_grid_ris.png", caption="Many lights scene, direct lighting only, no denoising, 32 sample RIS from the light grid") }}

However, there are still some unsolved problems:

1. Each cell has a fixed amount of storage for lights, so users need to be very careful not to exceed the limit — too many overlapping lights in one area, and things break.
2. The grid requires manual setup with a fixed world-space extent. Tying it to the camera doesn't work well, as it's very hard to maintain coverage of the full scene as the camera moves around. I haven't figured out the right solution here yet — maybe a spatial hashmap like we use for the GI world cache would work, but that brings its own complexity.

None of this has shipped yet, but I think light grids are the most promising path forward for Solari's light sampling. Hopefully I'll have more info to share next time.

## Recent Interesting Research

Speaking of research — to wrap up, here are some recent publications related to realtime raytracing that I've found interesting. If you enjoyed this post, give them a look!

* [Fast as Hell: idTech8 Global Illumination](https://www.youtube.com/watch?v=VTrdeqMMMK0)
* [Real-Time Path Tracing in RE ENGINE for Resident Evil Requiem and PRAGMATA](https://www.youtube.com/watch?v=Gq1QdlQSzn0)
* [The Future of Path Tracing | Best Practices, Optimizations & Future Standards](https://www.youtube.com/watch?v=0IrzX4LDIx8) (includes a great walkthrough of writing a pathtracer)
* [Path Tracing Nanite in NVIDIA Zorah](https://www.youtube.com/watch?v=8AnVpcIczyk)
* [Surfel-based global illumination on the web](https://juretriglav.si/surfel-based-global-illumination-on-the-web/)

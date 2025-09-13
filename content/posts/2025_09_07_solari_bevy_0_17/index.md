+++
title = "Hardware Raytracing in Bevy 0.17 (Solari)"
date = "2025-09-06"

[taxonomies]
tags = ["bevy", "raytracing"]
+++

# Hardware Raytracing in Bevy 0.17 (Solari)

## Introduction

Lighting is hard. Anyone who's tried to make a 3D scene look good knows the frustration of placing light probes, tweaking shadow cascades, and trying to figure out why their materials don't look quite right.

Over the past few years, real-time raytracing has gone from a research curiosity to a shipping feature in major game engines, promising to solve many of these problems by simulating how light actually behaves.

With the release of v0.17, [Bevy](https://bevy.org) now joins the club with experimental support for hardware raytracing!

<video style="max-width: 100%; margin: var(--gap) var(--gap) 0 var(--gap); border-radius: 6px;" controls>
  <source src="solari_recording.mp4" type="video/mp4">
</video>
<center>

*[PICA PICA scene by SEED](https://github.com/SEED-EA/pica-pica-assets)*

</center>

Back in early 2023, I [started](https://github.com/bevyengine/bevy/pull/10000) an ambitious project called Solari to integrate hardware raytracing into Bevy's rendering pipeline. I was experimenting with [Lumen](https://youtu.be/2GYXuM10riw)-style screen space probes for global illumination, and later extended it to use [radiance cascades](https://radiance-cascades.com).

These techniques, while theoretically sound, proved challenging to use in practice. Screen space probes were tricky to get good quality out of (reusing and reprojecting the same probe across multiple pixels is hard!), and radiance cascades brought its own set of artifacts and performance costs.

On top of the algorithmic challenges, the ecosystem simply wasn't ready. wgpu's raytracing support existed only as a work-in-progress PR that never got merged upstream. Maintaining a fork of wgpu (and by extension, Bevy) was time-consuming and unsustainable. After months of dealing with these challenges, I shelved the project.

In the 2 years since, I've learned a bunch more, raytracing has been upstreamed into wgpu, and raytracing algorithms have gotten much more developed. I've restarted the project with a new approach (ReSTIR, DLSS-RR), and soon it will be released as an official Bevy plugin!

In this post, I'll be doing a frame breakdown of how Solari works in Bevy 0.17, why I made certain choices, some of the challenges I faced, and some of the issues I've yet to solve.

## Why Raytracing for Bevy?

Before we start, I think it's fair to ask why an "indie" game engine needs high-end raytracing features that requires an expensive graphics card. The answer comes from my own experience learning 3D graphics.

Back when I was a teenager experimenting with small 3D games in Godot, I had a really hard time figuring out why my lighting looked so bad. Metallic objects didn't look reflective, scenes felt flat, and everything just looked wrong compared to the games I was playing.

I didn't understand that I was missing indirect light, proper reflections, and accurate shadows - I had no idea I was supposed to bake lighting (which wouldn't have worked for my dynamic game anyway).

This is the core problem that raytracing solves for indie developers. Even if most players won't have hardware capable of running ray-traced effects, having a reference implementation of what lighting is *supposed* to look like is incredibly valuable.

With fully dynamic global illumination, reflections, shadows, and direct lighting, developers can see how their scenes should be lit. Then they can work backwards to replicate those results with baked lighting, screen-space techniques, and other less performance-intensive approximations.

Without that reference, it's really hard to know what you're missing or how to improve your lighting setup. Raytracing provides the ground truth that other techniques are trying to approximate.

There's also the practical consideration that if Bevy ever wants to attract AAA game developers, we need these kinds of systems. Recent AAA games like DOOM: The Dark Ages and Cyberpunk 2077 rely heavily on raytracing, and artists working on these types of projects expect their tools to support similar techniques.

And honestly? It's just cool, and something I personally love working on :)

## Frame Breakdown

In its initial release, Solari supports raytraced diffuse direct (DI) and indirect lighting (GI). Light can come from either emissive triangle meshes, or analytic [directional lights](https://docs.rs/bevy/0.16.1/bevy/pbr/struct.DirectionalLight.html).

Direct lighting is handled via ReSTIR DI, while indirect lighting is handled by a combination of ReSTIR GI, and a world-space irradiance cache. Denoising is handled by DLSS Ray Reconstruction.

As opposed to coarse screen-space probes, per-pixel ReSTIR brings much better detail, along with being _considerably_ easier to get started with. I had my first prototype working in a weekend.

While I won't be covering ReSTIR from first principles, [A Gentle Introduction to ReSTIR:
Path Reuse in Real-time](https://intro-to-restir.cwyman.org) and [A gentler introduction to ReSTIR](https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir) are both really great resources. If you haven't played with ReSTIR before, I suggest giving them a skim before continuing with this post. Or continue anyways, and just admire the pretty pixels.

Onto the frame breakdown!

### GBuffer Raster

TODO: Images

The first step of Solari is also the most boring: rasterize a standard G-buffer.

#### Why Raster?

The G-buffer pass remains completely unchanged from standard Bevy (it's the same plugin). This might seem like a missed opportunity - after all, I could have used raytracing for primary visibility instead of rasterization - but there's an important reason I kept rasterization here.

By using raster for primary visibility, I maintain the option for people to use low-resolution proxy meshes in the raytracing scene, while still getting high quality meshes and textures in the primary view. The raster meshes can be full resolution with all their geometric detail, while the raytracing acceleration structure contains simplified versions that are cheaper to trace against.

#### Attachments

Bevy's G-buffer uses quite a bit of packing. The main attachment is a `Rgba32Uint` texture with each channel storing multiple values:

- **First channel**: sRGB base color and perceptual roughness packed as 4x8unorm
- **Second channel**: Emissive color stored as Rgb9e5
- **Third channel**: Reflectance, metallic, baked diffuse occlusion (unused by Solari), and an unused slot, again packed as 4x8unorm
- **Fourth channel**: World-space normal encoded into 24 bits via [octahedral encoding](https://www.jcgt.org/published/0003/02/01), plus 8 bits of flags meant for Bevy's default deferred shading (unused by Solari)

There's also a second `Rg16Float` attachment for motion vectors, and of course the depth attachment.

#### Drawing

The G-buffer rendering itself uses `multi_draw_indirect` to draw several meshes at once, using [sub-allocated](https://crates.io/crates/offset-allocator) buffers. Culling is done on the GPU using [two-pass occlusion culling](@/posts/2024_06_09_virtual_geometry_bevy_0_14/index.md#culling-first-pass) against a hierarchal depth buffer. Textures are handled bindlessly, and we try to minimize overall pipeline permutations.

These combined techniques keep draw call overhead and per-pixel overdraw fairly low, even for complex scenes.

### Light Tile Presampling

TODO: Image of light tile buffers

The next step in Solari is generating some presampled light buffers, following section 5 of [Rearchitecting Spatiotemporal Resampling for Production](https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf).

This presampling step greatly speeds up later passes that want to sample the scene's light sources, namely ReSTIR DI.

> Feel free to skip this section for now, and come back to it after reading the ReSTIR DI section.

#### Light Sampling APIs

Before I can explain this step, we first need to talk about Solari's shader API for working with light sources.

Bevy stores light sources as a big list of objects on the GPU. All emissive meshes and directional lights get collected by the CPU, and put in this list.

When calculating radiance emitted by a light source, Bevy works with specific light _samples_ - not the whole light at once. A `LightSample` uniquely identifies a specific subset of the light source, e.g. a specific point on an emissive mesh.

```rust
struct LightSample {
    light_id: u16,
    triangle_id: u16, // Unused for directional lights
    seed: u32,
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> LightSample {
    let light_count = arrayLength(&light_sources);
    let light_id = rand_range_u(light_count, rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);

    return LightSample(light_id, triangle_id, seed);
}
```

The light ID points to the overall light source object in the big list of lights.

The seed is used to initialize a random number generator (RNG). For directional lights, the RNG is used to choose a direction within a cone. For emissive meshes, the RNG is used to choose a specific point on the triangle identified by the triangle ID.

A `LightSample` can be resolved, giving some info on its properties:

```rust
struct ResolvedLightSample {
    world_position: vec4<f32>, // w component is 0.0 for directional lights, and 1.0 for emissive meshes
    world_normal: vec3<f32>,
    emitted_radiance: vec3<f32>,
    inverse_pdf: f32,
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

        let direction_to_light = sample_cone(directional_light);

        return ResolvedLightSample(
            vec4(direction_to_light, 0.0),
            -direction_to_light,
            directional_light.luminance,
            directional_light.inverse_pdf,
        );
    } else {
        let triangle_count = light_source.kind >> 1u;
        let triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);

        // Interpolates and transforms vertex positions, UVs, etc, and samples material textures
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
        );
    }
}
```

And finally a `ResolvedLightSample` can be used to calculate the received radiance at a point from the light sample, also known as the unshadowed light contribution:

```rust
struct LightContribution {
    received_radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    let light_distance = length(ray);
    let wi = ray / light_distance;

    let cos_theta_origin = saturate(dot(wi, origin_world_normal));
    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    let received_radiance = resolved_light_sample.emitted_radiance * cos_theta_origin * (cos_theta_light / light_distance_squared);

    return LightContribution(received_radiance, resolved_light_sample.inverse_pdf, wi);
}
```

Notably, only the first and second steps (generating a `LightSample`, resolving it into a `ResolvedLightSample`) involve branching based on the type of light (directional or emissive). Calculating the light contribution involves no branching.

#### Presampling Lights

Later steps in the rendering process are going to want to calculate `LightContributions`, e.g. in the ReSTIR DI shader. One thing we could do is perform the whole light sampling process (generate -> resolve  -> calculate contribution) all in one shader.

Indeed, for my first ReSTIR DI prototype, this iss what I did - but performance was terrible. In fact, even more than the actual raytracing - light sampling was by far the biggest performance bottleneck.

By generating the light sample, resolving it, and then calculating its contribution all in the same shader, we're introducing a lot of divergent branches and incoherent memory accesses. If there's one thing GPUs hate, it's divergence. GPUs perform better when all threads in a group are executing the same branch and don't need masking, and when the threads are all accessing similar memory locations that are likely in a nearby cache.

Instead, we can seperate out the steps. Generating a bunch of random light samples and resolving them can be performed ahead of time, by a seperate shader. We can then pack the resolved samples and store them in a buffer.

```rust
fn pack_resolved_light_sample(sample: ResolvedLightSample) -> ResolvedLightSamplePacked {
    return ResolvedLightSamplePacked(
        sample.world_position.x,
        sample.world_position.y,
        sample.world_position.z,
        pack2x16unorm(octahedral_encode(sample.world_normal)),
        vec3_to_rgb9e5_(sample.radiance * view.exposure),
        sample.inverse_pdf * select(1.0, -1.0, sample.world_position.w == 0.0),
    );
}

fn unpack_resolved_light_sample(packed: ResolvedLightSamplePacked, exposure: f32) -> ResolvedLightSample {
    return ResolvedLightSample(
        vec4(packed.world_position_x, packed.world_position_y, packed.world_position_z, select(1.0, 0.0, packed.inverse_pdf < 0.0)),
        octahedral_decode(unpack2x16unorm(packed.world_normal)),
        rgb9e5_to_vec3_(packed.radiance) / exposure,
        abs(packed.inverse_pdf),
    );
}
```

We call these presampled sets of lights "light tiles". Following the paper, we perform a compute dispatch to generate a fixed 128 tiles (these are not screen-space tiles), each with 1024 samples (`ResolvedLightSamplePacked`).

Samples are generated completely randomly, without any info about the scene - there is no spatial heuristic or any way of identifying "good" samples.

Later rendering steps that want to sample the scene's light sources can pick a random tile, and then random samples within the tile, and use `calculate_resolved_light_contribution()` to calculate radiance.

### World Cache

TODO: Image of the world cache

Whereas light tiles reshapes light sampling to accelerate direct lighting, the world cache accelerates sampling _indirect_ lighting for ReSTIR GI.

> Feel free to skip this section for now, and come back to it after reading the ReSTIR GI section.

The world cache voxelizes the world, storing accumulated irradiance (light hitting the surface) in each voxel. When sampling indirect lighting in ReSTIR GI, rather than having to trace additional rays towards light sources to estimate the irradiance, we can simply lookup the irradiance at the given voxel.

TODO: Diagram of sampling

The world cache both amortizes the cost of the GI pass, and reduces variance, especially for newly-disoccluded pixels for which the screen-space ReSTIR GI has no temporal history.

#### Cache Querying
The cache uses spatial hashing (TODO: link) to discretize the world. Unlike other options such as SDFs(?) (TODO: Godot), clipmaps (?) (TODO: Kajiya), cards (TODO: Lumen), or bricks (TODO: Brixelizer), spatial hashing requires no explicit build step, and automatically adapts to scene geometry.

With spatial hashing, a given descriptor (e.g., `{position, normal}`) hashes to a `u32` key. This key corresponds to an index within a fixed-size buffer, which holds whatever values you want to store in the hashmap - in our case, irradiance.

Either the entry that you're querying corresponds to some existing entry (same checksum), and you can return the value, or the entry does not exist (empty checksum), and you can initialize the entry by writing the checksum to it.

The checksum is the same descriptor, hashed to a different key via a different hash function, and is used to detect hash collisions.

The `query_world_cache()` function below is what ReSTIR GI will later use to lookup irradiance at the hit point for raytraces.

```rust
fn query_world_cache(world_position: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>) -> vec3<f32> {
    let cell_size = get_cell_size(world_position, view_position);

    let world_position_quantized = bitcast<vec3<u32>>(quantize_position(world_position, cell_size));
    let world_normal_quantized = bitcast<vec3<u32>>(quantize_normal(world_normal));

    var key = compute_key(world_position_quantized, world_normal_quantized);
    let checksum = compute_checksum(world_position_quantized, world_normal_quantized);

    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        let existing_checksum = atomicCompareExchangeWeak(&world_cache_checksums[key], WORLD_CACHE_EMPTY_CELL, checksum).old_value;
        if existing_checksum == checksum {
            // Cache entry already exists - get irradiance and reset cell lifetime
            atomicStore(&world_cache_life[key], WORLD_CACHE_CELL_LIFETIME);
            return world_cache_irradiance[key].rgb;
        } else if existing_checksum == WORLD_CACHE_EMPTY_CELL {
            // Cell is empty - reset cell lifetime so that it starts getting updated next frame
            atomicStore(&world_cache_life[key], WORLD_CACHE_CELL_LIFETIME);
            world_cache_geometry_data[key].world_position = world_position;
            world_cache_geometry_data[key].world_normal = world_normal;
            return vec3(0.0);
        } else {
            // Collision - jump to another entry
            key = wrap_key(pcg_hash(key));
        }
    }

    return vec3(0.0);
}
```

In Solari, the descriptor is a combination of the `world_position` of the query point, the `geometric_world_normal` (shading normal is too detailed) of the query point, and a LOD factor that's used to reduce cell count for far-away query points.

TODO: LOD showcase

```rust
fn quantize_position(world_position: vec3<f32>, quantization_factor: f32) -> vec3<f32> {
    return floor(world_position / quantization_factor + 0.0001);
}

fn quantize_normal(world_normal: vec3<f32>) -> vec3<f32> {
    return floor(world_normal + 0.0001);
}

fn compute_key(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = pcg_hash(world_position.x);
    key = pcg_hash(key + world_position.y);
    key = pcg_hash(key + world_position.z);
    key = pcg_hash(key + world_normal.x);
    key = pcg_hash(key + world_normal.y);
    key = pcg_hash(key + world_normal.z);
    return wrap_key(key);
}

fn compute_checksum(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = iqint_hash(world_position.x);
    key = iqint_hash(key + world_position.y);
    key = iqint_hash(key + world_position.z);
    key = iqint_hash(key + world_normal.x);
    key = iqint_hash(key + world_normal.y);
    key = iqint_hash(key + world_normal.z);
    return key;
}
```

#### Cache Decay

In order to maintain the world cache, we need a series of passes to decay and update active entries.

The first compute dispatch checks every entry in the hashmap, decaying their "life" count by 1. Each entry's life is initialized when the entry is created, and is reset when queried.

When an entry reaches 0 life, we clear out the entry, freeing up a space for future voxels to use.

```rust
@compute @workgroup_size(1024, 1, 1)
fn decay_world_cache(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var life = world_cache_life[global_id.x];
    if life > 0u {
        // Decay and write new life
        life -= 1u;
        world_cache_life[global_id.x] = life;

        // Clear cells that become dead
        if life == 0u {
            world_cache_checksums[global_id.x] = WORLD_CACHE_EMPTY_CELL;
            world_cache_irradiance[global_id.x] = vec4(0.0);
        }
    }
}
```

#### Cache Compact

The next three dispatches compact and count the total number of active entries in the world cache. This produces a dense array of indices of active entries, as well as indirect dispatch parameters for the next two passes.

The code is just a standard parallel prefix-sum, so I'm going to skip showing it.

#### Cache Update

Now that we know the list of active entries in the world cache (and can perform indirect dispatches to process each active entry), it's time to update the irradiance estimate for each voxel.

The first part of the update process is taking new samples of the scene's lighting.

Two rays are traced per voxel: a direct light sample, and an indirect light sample.

```rust
@compute @workgroup_size(1024, 1, 1)
fn sample_irradiance(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x < world_cache_active_cells_count {
        // Get voxel data
        let cell_index = world_cache_active_cell_indices[active_cell_id.x];
        let geometry_data = world_cache_geometry_data[cell_index];
        var rng = cell_index + constants.frame_index;

        // Sample direct lighting via RIS (1st ray)
        var new_irradiance = sample_random_light_ris(geometry_data.world_position, geometry_data.world_normal, workgroup_id.xy, &rng);

        // Sample indirect lighting via BRDF sampling + world cache querying (2nd ray)
        let ray_direction = sample_cosine_hemisphere(geometry_data.world_normal, &rng);
        let ray_hit = trace_ray(geometry_data.world_position, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray_hit.kind != RAY_QUERY_INTERSECTION_NONE {
            let ray_hit = resolve_ray_hit_full(ray_hit);
            new_irradiance += ray_hit.material.base_color * query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position);
        }

        world_cache_active_cells_new_irradiance[active_cell_id.x] = new_irradiance;
    }
}
```

The direct light sample is chosen via RIS, and uses the same presampled light tiles that we're going to use for ReSTIR DI. It's basically the same process as ReSTIR DI initial candidate sampling.

I've thought about using ReSTIR (well, ReTIR, without the spatial resampling part) for the world cache, but it's not something I've tried yet.

The indirect light sample is a little more interesting.

In order to estimate indirect lighting, we trace a ray using a cosine-hemisphere distribution. At the ray hit point, we query the world cache.

You might be thinking "Wait, aren't we _updating_ the cache? But we're also sampling from the same cache in order to... update it?"

TODO discuss multibounce...

#### Cache Blend

The second and final step of the world cache update process is to blend the new light samples with the existing irradiance samples, giving us an estimate of the overall irradiance via temporal accumulation. If you've ever seen code for temporal antialiasing, this should look pretty familiar.

The blending factor is based on the total sample count of voxel, capped at a max value. New voxels without any existing irradiance estimate use more of the new sample's contribution, while existing voxels with existing irradiance estimates use less of the new sample.

Choosing the max sample count is a tradeoff between having the cache be stable and low-variance, and having the cache be responsive to changes in the scene's lighting.

It's also important to note that this is a seperate compute dispatch from the previous dispatch we used for sampling lighting. If the passes were combined, we would have data races from voxels writing new irradiance estimates at the same time other voxels were querying them.

```rust
@compute @workgroup_size(1024, 1, 1)
fn blend_new_samples(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x < world_cache_active_cells_count {
        let cell_index = world_cache_active_cell_indices[active_cell_id.x];

        let old_irradiance = world_cache_irradiance[cell_index];
        let new_irradiance = world_cache_active_cells_new_irradiance[active_cell_id.x];
        let sample_count = min(old_irradiance.a + 1.0, WORLD_CACHE_MAX_TEMPORAL_SAMPLES);

        let blended_irradiance = mix(old_irradiance.rgb, new_irradiance, 1.0 / sample_count);

        world_cache_irradiance[cell_index] = vec4(blended_irradiance, sample_count);
    }
}
```

### ReSTIR DI

TODO

### ReSTIR GI

TODO

### DLSS Ray Reconstruction

TODO

## Results and Future Work

## Reference List
* https://blog.traverseresearch.nl/dynamic-diffuse-global-illumination-b56dc0525a0a
* https://intro-to-restir.cwyman.org
* https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir/
* https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf
* https://github.com/EmbarkStudios/kajiya/blob/main/docs/gi-overview.md

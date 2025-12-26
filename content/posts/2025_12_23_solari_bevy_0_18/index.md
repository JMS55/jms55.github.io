+++
title = "Realtime Raytracing in Bevy 0.18 (Solari)"
date = "2025-12-23"

[taxonomies]
tags = ["bevy", "raytracing"]
+++

## Introduction

It's been approximately three months since [my last post](@/posts/2025_09_20_solari_bevy_0_17/index.md), which means it's time to talk about all the work I've been doing for the upcoming release of Bevy 0.18!

Like last time, this cycle has seen me focused entirely on Solari - Bevy's next-gen, fully dynamic raytraced lighting system, allowing artists and developers to get high quality lighting - without having to spend time on static baking.

Before getting into what's changed in this release, let's take a quick look back at where Solari was in Bevy 0.17.

## Recap of 0.17

Bevy 0.17 saw the initial release of Solari, with the following components:

* Direct diffuse lighting via **ReSTIR DI**
* Indirect diffuse lighting final gather via **ReSTIR GI**
* Multi-bounce indirect diffuse lighting via a world-space **irradiance cache** (world cache)
* Denoising, anti-aliasing, and upscaling via **DLSS-RR**

ReSTIR DI handles the first bounce of lighting, ReSTIR GI handles the second bounce of lighting, and the world cache handles all subsequent bounces.

Summed together and denoised we get full, pathtraced lighting, close to the quality of a offline movie-quality pathtracer - but running much, much faster due to heavy temporal and spatial amortization.

Or at least, that's the theory.

In practice, all the amortization and shortcuts gives up some accuracy (making the result biased) in order to improve performance.

My goal with Solari is to get _as close as possible_ to the offline reference (zero bias), while getting "good enough" performance for realtime. Quality is currently the main priority.

To that end, Bevy 0.18 brings many quality (and some performance!) improvements to Solari:

* Specular material support
* Fixed the loss of brightness in the scene compared to the reference
* Eliminated bias from ReSTIR DI resampling
* Greatly reduced GI lag
* Greatly improved performance on larger scenes

## Specular Materials

In Bevy 0.17, Solari only supported diffuse materials. Diffuse materials were easier to get started with, because they don't depend on the incident light direction - they scatter the same no matter what direction the light is coming from.

Of course, games want more than just purely diffuse materials. Most PBR materials combine a diffuse lobe (Burley in Bevy's standard renderer, Lambert in Solari) with a specular lobe (usually GGX).

In Bevy 0.18, Solari now supports specular materials using a multiscattering GGX lobe, which gets added to the diffuse lobe.

### BRDF Evaluation

First, let's take a look at the material BRDF itself:

```rust
#import bevy_pbr::lighting::{F_AB, D_GGX, V_SmithGGXCorrelated, fresnel, specular_multiscatter}
#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_render::maths::PI
#import bevy_solari::scene_bindings::ResolvedMaterial

fn evaluate_brdf(
    world_normal: vec3<f32>,
    wo: vec3<f32>,
    wi: vec3<f32>,
    material: ResolvedMaterial,
) -> vec3<f32> {
    let diffuse_brdf = evaluate_diffuse_brdf(material.base_color, material.metallic);
    let specular_brdf = evaluate_specular_brdf(
        world_normal,
        wo,
        wi,
        material.base_color,
        material.metallic,
        material.reflectance,
        material.perceptual_roughness,
        material.roughness,
    );
    return diffuse_brdf + specular_brdf;
}

fn evaluate_diffuse_brdf(base_color: vec3<f32>, metallic: f32) -> vec3<f32> {
    let diffuse_color = calculate_diffuse_color(base_color, metallic, 0.0, 0.0);
    return diffuse_color / PI;
}

fn evaluate_specular_brdf(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    reflectance: vec3<f32>,
    perceptual_roughness: f32,
    roughness: f32,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NdotL = saturate(dot(N, L));
    let NdotH = saturate(dot(N, H));
    let LdotH = saturate(dot(L, H));
    let NdotV = max(dot(N, V), 0.0001);

    let F0 = calculate_F0(base_color, metallic, reflectance);
    let F_ab = F_AB(perceptual_roughness, NdotV);

    let D = D_GGX(roughness, NdotH);
    let Vs = V_SmithGGXCorrelated(roughness, NdotV, NdotL);
    let F = fresnel(F0, LdotH);
    return specular_multiscatter(D, Vs, F, F0, F_ab, 1.0);
}
```

Diffuse is nearly the same as in Solari 0.17, except the diffuse BRDF was changed so that it returns 0 for metallic materials, as metallic materials have no diffuse lobe.

For specular, a lot of the code can be reused from `bevy_pbr`, so the BRDF evaluation is only a couple of lines of function calls.

One thing to note is that special care must be taken to avoid NaNs.

In addition to clamping `NdotV` in the BRDF, we also limit roughness to 0.001 when loading materials, as zero roughness materials cause NaNs in the visibility function.

```rust
// Clamp roughness to prevent NaNs
m.perceptual_roughness = clamp(m.perceptual_roughness, 0.0316227766, 1.0); // Clamp roughness to 0.001
m.roughness = m.perceptual_roughness * m.perceptual_roughness;
```

### BRDF Sampling

Given this is a pathtracer, we don't just want to evaluate the BRDF; we also want to importance sample it to choose directions that would contribute a lot of outgoing light.

There are a couple of different methods to sample the overall BRDF for non-metallic materials that have both a diffuse and specular lobe, but let's skip that for now and just discuss sampling each individually.

Sampling the diffuse (Lambert) BRDF is pretty simple - it's just a cosine-weighted hemisphere (code from Solari 0.17):

```rust
// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec28%3A303
fn sample_cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = 1.0 - 2.0 * rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = normal.x + sin_theta * cos(phi);
    let y = normal.y + sin_theta * sin(phi);
    let z = normal.z + cos_theta;
    return vec3(x, y, z);
}
```

With the pdf being `cos_theta / PI`.

Sampling the specular (GGX) BRDF, however, is much more complicated.

The current state of the art paper for sampling a GGX distribution is "Bounded VNDF Sampling for Smith–GGX Reflections" by Kenta Eto and Yusuke Tokuyoshi:

```rust
// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    if roughness <= 0.001 {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        return ndf / (2.0 * (k * i.z + t));
    }
    return ndf * (t - i.z) / (2.0 * len2);
}
```

There are two tricky things to note with these functions:
* Inputs and outputs are in tangent space, and not world space
* `wo` and `wi` are defined from the BRDF's perspective, which is typically opposite to how you think about it in a pathtracer

So in practice you call them like so:

```rust
// https://jcgt.org/published/0006/01/01/paper.pdf
let TBN = orthonormalize(surface.world_normal);
let T = TBN[0];
let B = TBN[1];
let N = TBN[2];

let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N)); // Convert input from world space to tangent space
let wi_tangent = sample_ggx_vndf(wo_tangent, surface.material.roughness, &rng); // Swap wo and wi
let wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N; // Convert output from tangent space to world space

let pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, surface.material.roughness); // Swap wo and wi
```

One final thing to note is this line of code I added to `sample_ggx_vndf`, which doesn't appear in the paper:

```rust
if roughness <= 0.001 {
    return vec3(-wi_tangent.xy, wi_tangent.z);
}
```

Remember how earlier we clamped roughness to 0.001? Well that means we can no longer render perfect mirrors.

To get around this, when importance sampling the specular BRDF for a material with a roughness of 0.001, we just treat it like a perfect mirror and reflect the incident light direction around the Z axis.

This restores mirror-like behavior, while still preventing NaNs in BRDF evaluation.

### Specular DI

Now that we've covered Solari's updated material BRDF, let's talk about how lighting has changed.

To recap: For direct lighting, Solari is using ReSTIR DI.

We take a series of random initial samples from light sources, and use RIS to choose the best one. This is essentially fancy next event estimation (NEE).

We then do some temporal and spatial resampling to share good samples between frames/pixels.

Finally, we shade using the final selected sample (which in Bevy 0.17 used only the diffuse BRDF).

To add support for specular materials, there's a couple of different places that we should modify:

1. Account for the specular BRDF to the target function during initial resampling
2. Account for the specular BRDF during temporal and spatial resampling
3. Trace a BRDF ray during initial sampling and combine it with the NEE samples using multiple importance sampling (MIS)
    * This is the only way to sample DI for zero-roughness mirror surfaces
    * Improves quality for glossy (mid-roughness) surfaces
    * Improves quality for area lights that are very close to the surface
4. Account for the specular BRDF during shading of the final selected sample

For Bevy 0.18, I ended up spending most of my time on GI, so for DI I only did #4.

#1 and #2 are tricky because the whole point of ReSTIR is to share samples across pixels. But for specular, samples are not (easily) shareable, as unlike the diffuse lobe, a strong source of light for pixel A might be outside the specular lobe of pixel B and have zero contribution.

Maybe in practice it's not a big deal, or maybe using a second set of reservoirs for specular would help, but for now I've chosen to skip these, and treat all surfaces (including metallic ones) as purely diffuse during resampling.

#3 requires an extra raytrace, which costs a lot of performance, and so again I've skipped it.

When I get more time to experiment, I'll play around with these and see if any of them work well.

So to sum it up, for DI, all I did was swap `albedo / PI` with a call to `evaluate_brdf()` during the final shading step.

### Diffuse GI Changes

Indirect lighting is where specular gets much more interesting.

First off, as far as the world cache is concerned, all surfaces are diffuse only, with no specular lobe. This means that when you query the cache, you treat the query point as a diffuse surface. When updating cache entries, you also treat the cache point as a diffuse surface.

For per-pixel GI, Solari splits the lighting calculations into two seperate passes - one for the diffuse lobe, and one for the specular lobe.

The diffuse lobe is handled by the existing ReSTIR GI pass. ReSTIR GI resampling is exactly the same as in Bevy 0.17 - like DI, only the final shading changes.

For the ReSTIR GI final shading step, we're still shading using only the diffuse lobe, but now we need to skip shading metallic pixels that don't have a diffuse lobe.

### Specular GI

The specular lobe, on the other hand, is handled by an entirely new dedicated specular GI pass.

The basic structure of the pass looks like this (simplified):

```rust
let surface = load_from_gbuffer(pixel_id);
let wo = normalize(view.world_position - surface.world_position);

var radiance: vec3<f32>;
var wi: vec3<f32>;
if surface.material.roughness > 0.4 {
    // Surface is very rough, reuse the ReSTIR GI reservoir
    let gi_reservoir = gi_reservoirs_a[pixel_index];
    wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);

    radiance = gi_reservoir.radiance * gi_reservoir.unbiased_contribution_weight;
} else {
    // Surface is glossy or mirror-like, trace a new path
    let wi_tangent = sample_ggx_vndf(wo_tangent, surface.material.roughness, &rng);
    wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

    let pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, surface.material.roughness);
    radiance = trace_glossy_path(surface.world_position, wi, &rng) / pdf;
}

// Final shading
let brdf = evaluate_specular_brdf(surface.world_normal, wo, wi, surface.material...);
let cos_theta = saturate(dot(wi, surface.world_normal));
radiance *= brdf * cos_theta * view.exposure;
```

For rough surfaces, the specular lobe is wide enough to approximate the diffuse lobe. We can just skip tracing any new rays, and reuse the ReSTIR GI sample directly. This saves a lot of performance, with minimal quality loss.

For glossy or mirror surfaces, we need to trace a new path, following the best direction from importance sampling the GGX distribution.

The full code for `trace_glossy_path` is a bit long, so I'm just going to link to the [source on GitHub](https://github.com/bevyengine/bevy/blob/64c7bec4068aa063bfaa2cddcb90733f0e081cf8/crates/bevy_solari/src/realtime/specular_gi.wgsl#L71-L150).

The basic idea is:
* We trace up to three bounces (after three bounces, the quality loss from skipping further bounces is minimal)
* Lighting comes from either hitting an emissive surface, NEE, or querying the world cache
* Emissive contributions are skipped on the first bounce, as ReSTIR DI handles those paths
* We only query the world cache when hitting a rough surface (otherwise reflections would show the grid-like world cache)
* After hitting a rough surface and querying the world cache, we terminate the path
* We skip NEE for mirror surfaces
* We apply MIS between the emissive contribution and the NEE contribution
* Each bounce samples the GGX distribution to find the next bounce direction (if the surface was rough enough, we would have terminated in the world cache)

As you can see there's a lot of small details, which took me a while to figure out.

And there are still some large remaining issues:
* NEE is using entirely random samples, which leads to noisy reflections
* Glossy surfaces don't have any sort of path guiding to choose good directions, which also leads to noisy reflections
* No specular motion vectors to aid the denoiser leads to ghosting when objects in reflections move around
* Terminating in the world cache still leads to quality issues sometimes, especially on curved surfaces
  * TODO: Validate if this is what I was seeing in the cornell box scene

Specular motion vectors are something I plan to work on, following either ["Rendering Perfect Reflections and Refractions in Path-Traced Games"](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games) or ["Temporally Reliable Motion Vectors for Real-time Ray Tracing"](https://zheng95z.github.io/publications/trmv21). I just need to spend some more time understanding the theory.

For improving sampling during the path trace, this is technically what ReSTIR PT was invented to solve. However, ReSTIR PT is also very performance intensive, and I'm not convinced it's the path we should go down for Solari.

I have some other ideas in mind for improving sampling, which I'll talk about at the end of this post.

## Energy Loss Bug

One of the big problems Solari had in 0.17 was overall energy loss compared to a pathtraced reference.

At the time, I chalked it up to an inherent limitation of the world cache and moved on.

However, while experimenting with various things this cycle, I realized that not only was it not the world cache, but DI also was losing energy, and not just GI!

After many painful days of narrowing down the issue, I tracked it down to the [light tile code](@posts/2025-09-20-solari-bevy-0-17/#light-tile-presampling), which was shared between DI and the world cache.

The rgb9e5 packing of the light radiance I was doing did not have enough bits to encode the light, and so energy was being lost.

The fix (thanks to @SparkyPotato) was to apply a log2-based encoding to the radiance before packing. This allocates more bits towards the values that human perception cares about, and less bits towards the values that we have a harder time seeing.

```rust
fn pack_resolved_light_sample(sample: ResolvedLightSample) -> ResolvedLightSamplePacked {
    return ResolvedLightSamplePacked(
        // ...
        vec3_to_rgb9e5_(log2(sample.radiance * view.exposure + 1.0)),
        // ...
    );
}

fn unpack_resolved_light_sample(packed: ResolvedLightSamplePacked, exposure: f32) -> ResolvedLightSample {
    return ResolvedLightSample(
        // ...
        (exp2(rgb9e5_to_vec3_(packed.radiance)) - 1.0) / exposure,
        // ...
    );
}
```

With this fix, we're much closer to matching the reference.

## DI Resampling

One of the problems I wasn't able to solve in Bevy 0.17 was ReSTIR DI correlations introducing artifacts when denoising with DLSS-RR.

For ReSTIR GI, I was able to solve this with permutation sampling during temporal reuse. But for ReSTIR DI, trying to use permutation sampling lead to artifacts on shadow penumbras due to the way I was doing visibility reuse.

I played with resampling ordering a bit more this cycle, and was able to come up with a solution.

In Bevy 0.17, the whole ReSTIR DI algorithm looked like this:

1. Initial sampling
2. Test visibility of initial sample
3. Temporal resampling
4. Choose spatial sample
5. Test visibility of spatial sample
6. Spatial resampling
7. Store final reservoir for next frame temporal reuse
8. Final shading

In Bevy 0.18, it now looks like this:

1. Initial sampling
2. Test visibility of initial sample
3. Temporal resampling
4. Choose spatial sample
5. Spatial resampling
6. Store final reservoir for next frame temporal reuse
7. Test visibility of final reservoir
8. Final shading

The two big differences are:
* The second visibility test was moved from the spatial sample, to the final sample after all resampling steps
* The second visibility test is performed for the final shading, but is _not_ fed forward for next frame's temporal resampling

Moving the second visibility test from the spatial sample only to after all resampling was the key change.

Before permutation sampling, it was ok to not re-test visibility for the temporal sample. The the light was visible to the pixel last frame, it's probably still visible this frame. Same for if the light was not visible last frame. When this assumption is wrong, e.g. for moving objects, it just led to a 1-frame lag in shadows that's almost unnoticable - an acceptable tradeoff.

With permutation sampling, we can no longer trust that the visibility of the temporal sample is correct to reuse. The temporal sample now may come from a neighboring pixel, and at shadow pneumbras, the visibility is changing very frequently. It's no longer safe to reuse visibility, even on static scenes - we must retest visibility.

The best way to test visibility without using extra ray traces is to move it right before shading of the final sample, where incorrect visibility would show up on screen

The second change (not feeding forward the second visibility test to the next frame) is not strictly necessary, but keeps direct lighting unbiased.

If you were to feed forward the second visibility test, the following might happen:
1. A pixel checks visibility and finds that the light is occluded, setting the reservoir's contribution to 0
2. The reservoir is stored for reuse next frame
3. \<Next frame\>
4. The reservoir is reused temporally for the same pixel (say that the initial sample happened to also be 0 contribution)
5. The reservoir is reused spatially by a different pixel, which sees that it has zero contribution, and does not choose it via resampling
    * Except since this is a different pixel, the light is not occluded, and the sample should have had non-zero contribution!

Reusing visibility like this leads to bias in the form of shadows that "halo" objects, expanding further out than they should.

Interestingly, when I tried these modifications to ReSTIR GI, it made things _more_ biased. Indirect shadows became very feint and sometimes disappeared altogether. ReSTIR GI still uses the same algorithm it did in Bevy 0.17.

One final note on DI resampling: like we were doing with ReSTIR GI, we now use the balance heuristic for ReSTIR DI resampling, instead of constant MIS weights. This makes a small difference (hence why I never noticed it until now), but it _does_ slightly increase emissive light brightness, matching the pathtraced reference better.

## World Cache Improvements

The world cache is the oldest part of Solari - it was copied nearly wholesale from my original prototype 3 years ago, without any real changes except for the addition of the LOD system.

Because of this, it was also the jankiest part of Solari.

As I started testing on more complex scenes, it became clear that there were significant problems:
* On the cornell box scene, it worked fine.
* On the PICA PICA scene, it worked ok when conditions were static, but under dynamic conditions the GI was fairly laggy.
* On Bistro, performance wasn't good, especially as you started moving around the scene.

In Bevy 0.18, I spent a large amount of time fixing these issues.

### Cache Lag

In the PICA PICA scene, if you turn off all the lights, it would take a good while for the light to completely fade. The reason being that: A) the world cache samples itself, recursively propogating light around the scene for a while, and B) the exponential blend between new and current radiance samples keeps the old radiance around for a decent amount of time.

To combat this, we could increase the blend factor, to keep the lighting responsive. However that would lead to way more noise and instability under static lighting conditions.

What we really need is an adaptive blend factor, which [Guillaume Boissé](https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a) was kind enough to share with me.

We keep track of the change in luminance between frames, and use that to compute an adaptive blend factor.

```rust
let old_radiance = world_cache_radiance[cell_index];
let new_radiance = world_cache_active_cells_new_radiance[active_cell_id.x];
let luminance_delta = world_cache_luminance_deltas[cell_index];

// https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a
let sample_count = min(old_radiance.a + 1.0, WORLD_CACHE_MAX_TEMPORAL_SAMPLES);
let alpha = abs(luminance_delta) / max(luminance(old_radiance.rgb), 0.001);
let max_sample_count = mix(WORLD_CACHE_MAX_TEMPORAL_SAMPLES, 1.0, pow(saturate(alpha), 1.0 / 8.0));
let blend_amount = 1.0 / min(sample_count, max_sample_count);

let blended_radiance = mix(old_radiance.rgb, new_radiance, blend_amount);
let blended_luminance_delta = mix(luminance_delta, luminance(blended_radiance) - luminance(old_radiance.rgb), 1.0 / 8.0);

world_cache_radiance[cell_index] = vec4(blended_radiance, sample_count);
world_cache_luminance_deltas[cell_index] = blended_luminance_delta;
```

Now GI is stable under static conditions, but reacts pretty fast under dynamic conditions. It's not perfect - we're still heavily relying on temporal accumulation and denoising - but it's a heck of a lot better.

Once again, thanks a ton to Guillaume Boissé for this code! I was struggling to come up with something myself, and this perfectly solved my problem!

### Cache Lifetimes

While Solari was working great on smaller scenes, on larger scenes like Bistro, performance was much worse.

The world cache update pass was taking way too long, and worse, as I moved around the scene, it got worse and worse.

The reason is that since cache entries sample each other (in order to get multibounce lighting), they were keeping each other alive forever. So once you stepped into an area, it would forever be present in the world cache, even when you left the area.

The solution (thanks to @IsaacSM and @NthTensor) ended up being pretty simple!

```rust
fn query_world_cache(world_position: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, cell_lifetime: u32, rng: ptr<function, u32>) -> vec3<f32> {
    let cell_size = get_cell_size(world_position, view_position);

    let world_position_quantized = bitcast<vec3<u32>>(quantize_position(world_position, cell_size));
    let world_normal_quantized = bitcast<vec3<u32>>(quantize_normal(world_normal));
    var key = compute_key(world_position_quantized, world_normal_quantized);
    let checksum = compute_checksum(world_position_quantized, world_normal_quantized);

    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        let existing_checksum = atomicCompareExchangeWeak(&world_cache_checksums[key], WORLD_CACHE_EMPTY_CELL, checksum).old_value;

        // Cell already exists or is empty - reset lifetime
        if existing_checksum == checksum || existing_checksum == WORLD_CACHE_EMPTY_CELL {
#ifndef WORLD_CACHE_QUERY_ATOMIC_MAX_LIFETIME
            atomicStore(&world_cache_life[key], cell_lifetime);
#else
            atomicMax(&world_cache_life[key], cell_lifetime);
#endif
        }

        if existing_checksum == checksum {
            // Cache entry already exists - get radiance
            return world_cache_radiance[key].rgb;
        } else if existing_checksum == WORLD_CACHE_EMPTY_CELL {
            // Cell is empty - initialize it
            world_cache_geometry_data[key].world_position = world_position;
            world_cache_geometry_data[key].world_normal = world_normal;
            return vec3(0.0);
        } else {
            // Collision - linear probe to next entry
            key += 1u;
        }
    }

    return vec3(0.0);
}
```

When a ReSTIR GI or specular GI pixel is querying the world cache, nothing has changed. We still perform `atomicStore(&world_cache_life[key], WORLD_CACHE_CELL_LIFETIME)`, resetting the lifetime of the queried cache entry.

However when a world cache entry is querying another entry during the world cache update pass, the algorithm changes.

Now we're instead doing `atomicMax(&world_cache_life[key], cell_lifetime_of_querier)`.

When the camera is in a given area, ReSTIR GI and specular GI pixels will reset world cache entries to their max lifetime. Then during world cache update the next frame, those world cache entries will copy their max lifetime to other entries nearby.

However once the camera moves away from the area, there will be no more pixels querying the world cache. When world cache entries go to query each other, they'll copy over their current lifetimes (which is decreasing each frame). After a couple of frames, all the world cache entries will go dead.

No more performance wasted on areas away from the camera!

### Misc Cache Tweaks

Finally, I tweaked a bunch of other things based on my testing in Bistro:

* Limited indirect rays sent from cache entries during the world cache update step to a max of 50 meters - This prevents long raytraces from holding up the whole threadgroup, improving performance, and prevents far-away samples from influencing the cache, reducing variance.
* Switched the world cache update workgroup size from 1024 to 64 threads - Much more appropriate for raytracing workloads. This fixed some really weird GPU usage traces I was seeing in NSight.
* Make the world cache transition LODs faster - In a large scene like Bistro, we had way too many cache entries for far-away areas.

Combined, these changes brought the world cache update step from 1.42ms to a much more reasonable 0.09ms in Bistro.

## What's Next

Solari has improved a ton in Bevy 0.18, but there's (of course) still more work to be done!

First, some general issues (many of these carrying over from my last blog post):
* Feature parity for things like skinned and morphed meshes, alpha masks, transparent materials, support for more types of light sources, etc still need implementing.
* Specular motion vectors are not implemented, so mirror and glossy indirect reflections can have ghosting.
* Solari is still NVIDIA only in practice due to relying on DLSS-RR (FSR-RR _did_ release since my last blog post, but to my sadness is currently DirectX12 only - no Vulkan support. AMD employees - please reach out!)
* Shader execution reordering (blocked on wgpu support) and half-resolution GI (on top of DLSS upscaling) would bring major performance improvements.

DI quality: github.com/bevyengine/bevy/pull/21366, LTC https://ishaanshah.xyz/risltc, better-than-random light sampling https://blog.traverseresearch.nl/fast-cdf-generation-on-the-gpu-for-light-picking-5c50b97c552b

GI quality: Handling when ray_length less than cache cell_size, path guiding https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg

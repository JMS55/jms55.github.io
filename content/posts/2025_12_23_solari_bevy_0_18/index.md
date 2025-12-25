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

## Resampling Bias

## World Cache Improvements

## What's Next
* Specular motion vectors
* RT pipelines and SER
* Direct light guiding
* Indirect light guiding
* FSR4
* https://github.com/bevyengine/bevy/issues/20203

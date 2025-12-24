+++
title = "Realtime Raytracing in Bevy 0.18 (Solari)"
date = "2025-12-23"

[taxonomies]
tags = ["bevy", "raytracing"]
+++

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

Summed together and denoised we get full, pathtraced lighting, close to the quality from a offline movie-quality pathtracer - but running much, much faster due to heavy temporal and spatial amortization.

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

In Bevy 0.17, Solari only supported diffuse materials. Diffuse materials were easier to get started with, because they don't depend on the incident light direction - they scatter the same no matter where the light is coming from.

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

For specular, a lot of the code is reused from `bevy_pbr`, so the BRDF evaluation is only a couple of lines of function calls.

One thing to note is that special care must be taken to avoid NaNs.

In addition to clamping `NdotV` in the BRDF, we also limit roughness to 0.001 when loading materials, as zero roughness materials cause NaNs in the visibility function.

```rust
// Clamp roughness to prevent NaNs
m.perceptual_roughness = clamp(m.perceptual_roughness, 0.0316227766, 1.0); // Clamp roughness to 0.001
m.roughness = m.perceptual_roughness * m.perceptual_roughness;
```

### BRDF Sampling

Given this is a pathtracer, we don't just want to evaluate the BRDF; we also want to importance sample it, to choose directions that would contribute a lot of outgoing light.

There are a couple of different methods to sample the overall BRDF for non-metallic materials that have both a diffuse and specular lobe, but let's skip that for now and just discuss sampling each individually.

Sampling the diffuse (lambert) BRDF is pretty simple - it's just a cosine-weighted hemisphere (Code from Solari 0.17):

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
fn orthonormalize(z_basis: vec3<f32>) -> mat3x3<f32> {
    let sign = copysign(1.0, z_basis.z);
    let a = -1.0 / (sign + z_basis.z);
    let b = z_basis.x * z_basis.y * a;
    let x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    let y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3x3(x_basis, y_basis, z_basis);
}

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

### Specular GI

## Energy Loss

## Resampling Bias

## World Cache Improvements

## What's Next
* Specular motion vectors
* RT pipelines and SER
* Direct light guiding
* Indirect light guiding
* FSR4
* https://github.com/bevyengine/bevy/issues/20203

+++
title = "Realtime Raytracing in Bevy 0.20 (Solari)"
date = "2026-09-15"

[taxonomies]
tags = ["bevy", "raytracing"]

[extra]
math = true
+++

## Introduction

Bevy 0.20 is going to release soon, so it's time to talk about all the improvements I've been working on for Solari, Bevy's realtime pathtracer.

Whereas last cycle was mostly incremental improvements, this cycle was the complete opposite. Solari 0.20 features nothing less than a complete rewrite of the renderer!

Lots to cover, so let's get started!

## Reducing Shadow Lag

This section is going to start off with a good bit of math, but we'll get to the practical applications towards the end. Bear with me for a bit.

### MIS for the Combined Estimator

In pathtracing, when combining multiple sampling strategies (ways of generating samples, also called _domains_), you need to apply proper MIS weights to obtain both unbiased rendering, and optimal variance reduction.

Typically, you would use something like the balance heuristic to compute MIS weights.

Given a sample $a$ from strategy A, and sample $b$ from strategy B, in order to combine them you would start by computing the following 4 values:
1. $p_A(a)$ - The PDF of sample $a$ as if it were sampled from strategy A
2. $p_B(b)$ - The PDF of sample $b$ as if it were sampled from strategy B
3. $p_A(b)$ - The PDF of sample $b$ as if it were sampled from strategy A (a different domain!)
4. $p_B(a)$ - The PDF of sample $a$ as if it were sampled from strategy B (a different domain!)

The first two values you already have - they're how you obtained $a$ and $b$ in the first place.

If you were simply computing the (non-combined) estimator of e.g. just $a$, you would compute $f(a) / p_A(a)$, and that would be it.

But if you want to compute the _combined_ estimator of these two samples (which come from two different domains) using the balance heuristic, it gets more complicated. You need the last two values, which are new and need to be computed from scratch. These are _cross domain_ PDFs.

With the regular and cross domain PDFs in hand, we can now compute the MIS weights using the balance heuristic:

$$m(a) = \frac{p_A(a)}{p_A(a) + p_B(a)}$$

$$m(b) = \frac{p_B(b)}{p_A(b) + p_B(b)}$$

And then the final combined estimator:

$$f_1 = \frac{f(a)}{p_A(a)} m(a)$$

$$f_2 = \frac{f(b)}{p_B(b)} m(b)$$

$$f_{a,b} = f_1 + f_2$$

### MIS in RIS

In RIS, instead of _combining_ samples, you aim to _select_ a single "best" sample by randomly choosing from the samples with weights proportional to each sample's contribution.

As part of RIS, you always need to compute MIS weights for each sample. You can use a constant $1/N$ MIS weight for every sample if all samples are from the same domain, but with multiple domains you need e.g. the balance heuristic. See [A Gentle Introduction to ReSTIR](https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf) for details if the process of RIS with MIS is unfamiliar to you.

To give a concrete multi-domain example, let's say you're generating the "initial" reservoir for ReSTIR, and are doing RIS over the unshadowed light contributions of two different light samples.

The first sample is drawn by randomly selecting an emissive mesh, then randomly selecting a triangle in that mesh, and then randomly selecting a front-facing point on that triangle (this is NEE).

The second sample is drawn by shooting a ray according to the current pixel's BRDF, and hitting a point on an emissive mesh (this is BRDF sampling).

We have two different strategies: NEE and BRDF sampling. In order to do RIS over the two samples, we need to calculate MIS weights for each sample.

We already saw how to do this in the previous section: you calculate both $p_\text{NEE}$ and $p_\text{BRDF}$ for each sample, and simply apply the balance heuristic to get the two MIS weights $m(\text{sample}_1)$ and $m(\text{sample}_2)$. Then we can apply RIS to choose one of the two samples.

### MIS in ReSTIR

Now that we've covered how MIS works in RIS, it should be straightforward to apply the same concepts to ReSTIR, right? ReSTIR is just RIS, after all.

In ReSTIR, we aim to resample reservoirs (samples) from different pixels. E.g. you could have an "initial" reservoir, a "temporal" reservoir, and a "spatial" reservoir, and you aim to select the best reservoir to keep and use for shading from the three via RIS.

It's important to note that each of these reservoirs comes from a different domain: the initial reservoir comes from the frame+pixel, the temporal reservoir comes from the previous frame+pixel, and the spatial reservoir comes from the current frame but from a neighboring pixel.

Since multiple domains are involved, we can't use constant MIS weights, and will have to use the balance heuristic again.

To use the balance heuristic, we need PDFs. But with reservoirs, while the regular PDFs can be replaced by unbiased contribution weights (UCWs), we have no method of generating cross-domain PDFs!

Instead of the balance heuristic, we'll have to instead use the _generalized balance heuristic_, which uses a combination of the UCWs and target functions of each reservoir to compute MIS weights.

Unfortunately, whereas the regular balance heuristic involves cross-domain PDFs that are pretty cheap to compute, the generalized balance heuristic involves computing cross-domain _target functions_, which are _much_ more expensive to compute since you need to trace new visibility rays and even full paths.

### MIS in Solari

In realtime rendering, we often use hacks and approximations in the name of performance, and ReSTIR is no exception.

See, you don't strictly _need_ to use proper MIS weights. You can tradeoff various levels of performance/correctness (bias)/variance reduction by using purposefully incorrect MIS weights, depending on what you want your renderer to achieve.

For instance, when computing MIS weights for a reusing a temporal reservoir, you could do any of the following, in order of increasing performance cost:
1. Use constant MIS weights
2. Use the generalized balance heuristic, but with various assumptions in the cross-domain target functions:
  * The material properties and world position of the current and previous frame pixels are the same
  * The intensity and color of the light the sample is from is the same in the current and previous frame (but watch out for lights that were despawned between frames)
  * When computing the target function of the canonical sample in the temporal reservoir's domain, if the light was visible to the pixel last frame, it's probably visible to the pixel again this frame (e.g. there's no occluders that got in the way since last frame) (note that you'll still want to trace a visibility ray when computing the target function of the temporal sample in the canonical domain)
3. Same as #2, but:
  * Store and lookup the gbuffer and depth data from the previous frame
  * Keep a mapping of lights between frames, and lookup the new light proprties for the temporal sample
  * When computing the target function of the canonical sample in the temporal reservoir's domain, instead of skipping the visibility ray, trace a visibility ray using the current frame's TLAS as an approximation of the true visibility
4. Use the full, properly computed generalized balance heuristic / target functions, with no simplifications, by e.g. storing the previous frame's TLAS and BLAS data, and tracing the extra rays

In Bevy 0.19, Solari was (more or less) following #3. This led to bias, increased variance, and importantly, temporal lag. Shadows of moving objects often appeared to lag behind the object.

In Bevy 0.20, we switched to using the previous frame's TLAS when testing visibility in the domain of the previous frame. This costs a decent amount of extra VRAM and complexity due to having to keep a second TLAS around, but in return it greatly reduces temporal shadow lag.

## Unified ReSTIR

For any readers who felt that the previous section was a lot of math, and not much engineering, rejoice! This section is going to be the complete opposite :)

### Historical Background

A few months ago, NVIDIA researchers released a really interesting paper called [ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced).

The new paper is a followup to their 2022 paper on ReSTIR PT, this time detailing a lot of algorithmic improvements to improve the performance of (the quite expensive) ReSTIR PT (among some other changes to e.g. reduce correlations).

The main optimization from the paper that I want to talk about is the concept of a single, "unified" ReSTIR pass.

To give a quick recap of algorithms:
* 2020: ReSTIR DI resamples direct illumination
* 2021: ReSTIR GI resamples indirect, diffuse illumination
* 2022: ReSTIR PT resamples indirect, diffuse and specular illumination
* 2026: ReSTIR PT Enhanced resamples both direct and indirect illumination, and both diffuse and specular paths

Historically, ReSTIR DI would handle direct lighting, and ReSTIR GI or PT would handle indirect lighting.

For instance, in Bevy 0.19, the main lighting algorithm consisted of the following 5 compute dispatches:
1. Initial sampling of DI via NEE, and temporal resampling for DI
2. Spatial resampling and shading for DI
3. Initial sampling of diffuse GI via uniform hemisphere tracing, and temporal resampling for diffuse GI
4. Spatial resampling and shading for diffuse GI
5. Up to 3 bounce pathtracing of specular reflections (single-sample NEE at every vertex except the primary)

As another example, in [NVIDIA's NvRTX fork of Unreal Engine](https://www.youtube.com/watch?v=8AnVpcIczyk), ReSTIR DI handled direct lighting, and ReSTIR PT would handle diffuse and specular indirect lighting, computed at quarter-resolution because it's so expensive.

These are still pathtracing, just with different kernels to handle different parts of the overall path space.

> As an aside, up until recently, games would render each of these effects (DI, diffuse GI, specular GI) to different textures, and denoise them separately using specialized denoisers for each type.
>
> Oftentimes, games would only use raytracing for one or two of these effects, and e.g. handle DI via traditional shadow maps and screen-space light clustering, or use reflection probes and screen space reflections to handle specular GI.
>
> With the advent of really good denoisers like DLSS-RR, and with faster hardware, it's much more practical to fully raytrace all lighting effects (e.g. pathtracing), and combine all effects into a single texture and denoise them together.

### A Unified Pathtracer

What the ReSTIR PT Enhanced paper did, and now Solari in Bevy 0.20 does, is combine the two (DI and GI) into one.

The new lighting algorithm for Solari 0.20 has been simplified down to just two dispatches:
1. Initial pathtracing, up to 3 bounces, plus temporal resampling
2. Spatial resampling and shading

These two passes handle DI, GI, diffuse, and specular lighting effects, all with a single set of reservoirs and a single initial sampling routine that does pathtracing.

The basic idea is that you start with your bog standard pathtracer:
1. Perform NEE at the current path vertex (use RIS to choose a good sample for this)
2. Sample the BRDF, trace your next ray, continue if hit
3. Add emissive contribution, if any
4. Terminate in the world cache (if the path has been wide enough)
5. Perform russian roulette to terminate long paths early
6. Loop

Adding ReSTIR then transforms the algorithm to look like this:
1. Initialize an empty reservoir
2. Perform NEE at the current path vertex
    * Resample new path contribution into the reservoir
3. Sample the BRDF, trace your next ray, continue if hit
4. Add emissive contribution, if any
    * Resample new path contribution into the reservoir
5. Terminate in the world cache (if the path has been wide enough)
    * Resample new path contribution into the reservoir
6. Perform russian roulette to terminate long paths early
7. Loop

At the end, you're left with a single "initial" reservoir that we can then perform temporal and spatial resampling on, before finally using it to shade the pixel.

To give you an idea of what the reservoir structure looks like, here's what Solari's reservoir structs looked like in Bevy 0.19:
```rust
// 16 bytes
struct DIReservoir {
    sample: LightSample,
    confidence_weight: f32,
    unbiased_contribution_weight: f32,
}

// 48 bytes
struct GIReservoir {
    sample_point_world_position: vec3<f32>,
    weight_sum: f32, // could be deleted, don't actually need to store
    radiance: vec3<f32>,
    confidence_weight: f32,
    sample_point_world_normal: vec3<f32>,
    unbiased_contribution_weight: f32,
}
```

And here's what Solari's new unified reservoir structure looks like in Bevy 0.20:

```rust
// 48 bytes
struct UnifiedReservoir {
    sample_point_world_position: vec3<f32>,
    unbiased_contribution_weight: f32,
    radiance: vec3<f32>,
    confidence_weight: f32,
    sample_point_world_normal: vec2<f32>, // octahedral packed
    light_sample: LightSample,
}
```

### Implementation Choices

Unlike the paper, I chose not to implement the hybrid shift, and stick purely to the reconnection shift. Basically, when tracing paths, if it's from too sharp of a specular lobe, instead of putting it in the reservoir to use for resampling, we simply write the path's radiance directly to the screen.

Unlike a wide lobe, it's not possible to use a sharply directional lobe with the reconnection shift. The hybrid shift, which conditionally switches to "random replay", can handle these types of lobes, but it involves re-tracing the path, which in my opinion is too expensive for a realtime renderer.

Additionally, I have a fairly high roughness gate on the primary vertex to guard resampling. I realized that just because you _can_ resample specular paths, doesn't mean you _should_.

As denoisers have improved recently, correlations from temporal and spatial reuse have become a big problem. Because we're reusing discrete samples, the denoiser sees the same samples more often than it otherwise would with regular importance sampling. The denoiser then thinks that this is detail worth preserving (it's a repeating pattern in an otherwise noisy signal), which leads to ugly denoising artifacts.

See [Compatibility-Guided Neighbor Selection for ReSTIR](https://www.youtube.com/watch?v=MCqnETw3l8A) for a good explanation of correlations in ReSTIR.

Smooth specular surfaces, unlike rough diffuse surfaces, have a low amount of possible paths with strong contributions. So as soon as ReSTIR finds one good sample, it tends to propogate to all neighboring pixels, and stay there for a while without being overwritten by new samples. This over-reuse and correlated neighbor pixels confuse the denoiser.

> As a side note, in the previous post in this series, I recommended halving the spatial sampling radius after failing to find a valid sample. It turns out, this is a bad idea, and I have reverted it. Doing this _also_ introduces problematic correlations, as many pixels end up reusing the same few samples within a small radius.

I tried a few different ways of mitigating correlations, e.g. by copying ReSTIR PT Enhanced's duplication maps, or scaling temporal confidence weight by surface roughness, but in the end the only foolproof method I came up with was a hard cutoff based on roughness.

One final note on the algorithm - if you've been paying attention, you may be wondering: "what kind of MIS weights do you use during resampling?"

After all, BRDF-sampled emissive hits at the primary vertex, NEE hits at the primary vertex, BRDF-sampled emissive hits at further vertices, NEE hits at further vertices, and the world cache termination are all _very_ different types of sampling stategies.

The answer is kind of complicated, and you can [take a look at the code](https://github.com/bevyengine/bevy/blob/37256568acc84fd2c8e3c801b906b7dd253f84f5/crates/bevy_solari/src/realtime/initial_path.wesl) for exact details, but it's actually very similar to the way you would typically do MIS in pathtracing.

### Benefits

By merging the two sets of reservoirs into one structure, we're saving 16 bytes/pixel compared to non-unified ReSTIR's two separate sets of reservoirs.

More importantly than memory savings, that are a bunch of other improvements we either get for free, or can now afford to make due to combining the ReSTIR passes:
* In Solari 0.19's ReSTIR DI, we never traced BRDF rays to sample emissive lighting, instead relying only on NEE. Now that everything is unified into one pathtracer, we can trace and resample BRDF-based emissive lighting for free, as if we don't hit an emissive mesh, the ray is not wasted - it just becomes a GI path. This greatly improves direct lighting from nearby emissives.
* As discussed in the previous section, in Solari 0.19, we skipped tracing some visibility rays during MIS to increase performance, at the cost of some bias (shadow growing slightly larger and darker). Now, because we only have a single set of ReSTIR passes, we can afford to trace these rays, giving us fully unbiased (and lower variance) rendering. Overall we went from about 6-12 rays per pixel (3 DI, 3 GI, 0-6 specular BRDF + NEE), to 6-10 rays per pixel (2-6 pathtracing BRDF + NEE, 4 ReSTIR).
* Whereas before we never resampled specular GI, we can now resample sufficently rough/glossy specular reflections. More on the caveats here in a bit.
* Much better light leak prevention, as instead of always terminating into the world cache after the first bounce like we used to for ReSTIR GI, it's now easy to simply continue tracing the path.

Overall, we get some very nice quality wins, simplify the code, and reduce our memory usage. Performance tends be either about the same, slightly slower, or slightly faster depending on the scene and GPU in question.

### Downsides

Note that there _are_ some downsides I've found from the unified ReSTIR algorithm.

One simple downside is that a big unified pathtracing kernel uses more registers than separate kernels, which reduces occupancy and can hurt performance. It's not typically a huge loss, but it's something to be aware of.

The bigger issue is that we're no longer tracing paths for both the primary vertex's diffuse _and_ specular lobes every frame. Before, we had dedicated passes for diffuse and specular GI paths (really just a single-bounce for diffuse GI, we weren't doing a full path), but now we're only tracing a single path, with stochastic lobe selection for dielectric materials that have two lobes. Tracing two separate paths would be too expensive.

Remember from the last post that dielectric materials are implemented as a thin specular lobe layered _over_ a diffuse lobe. Depending on what angle you view it at, the top specular layer gets a certain percentage of the energy, with the rest being transmitted to the diffuse layer beneath it. When picking a BRDF lobe to follow to keeping tracing the path, we can importance sample the two layers according to these percentages, biasing towards picking the lobe that will recieve a higher amount of energy.

However, that only accounts for one part of the rendering equation. Incident radiance is _also_ an important quantity to account for. Let's say that for a given pixel we estimate that the specular layer will recieve 70% of the incoming energy, while the diffuse layer only gets 30%. We should choose to sample the specular layer, right?

But what if the specular lobe is pointing in a direction where no light is coming from? 70% of 0 energy is still 0 - it would've been better to sample the diffuse lobe after all...

Really, we'd like to sample according to some estimate (obviously if we had a perfect, cheap predictor of the full rendering equation, there would be no need to trace paths at all) of the combined `BRDF * incident_radiance` product. This is called "path guiding", and Bevy dosen't currently have a way of doing that. We have to stick to the inferior BRDF-only sampling.

In practical terms, I noticed that after switching to unified ReSTIR, that specular reflections in dielectric materials got much noisier, leading to the denoiser reconstructing a much more blurry reflection.

To mitigate this issue, I simply boosted the chance of sampling the specular lobe for smooth surfaces.

```rust
fn specular_lobe_sampling_probability(rho: LobeReflectances, perceptual_roughness: f32) -> f32 {
    let specular_luminance = luminance(rho.specular);
    let total_luminance = specular_luminance + luminance(rho.diffuse);
    let energy_probability = specular_luminance / max(total_luminance, 0.0001);

    // Give smooth dielectric reflections more specular samples to prevent undersampling
    let roughness_biased_probability = mix(0.5, energy_probability, perceptual_roughness);
    return max(energy_probability, roughness_biased_probability);
}
```

The logic is pretty simple: for smooth dielectric surfaces, ReSTIR will _already_ reduce noise from diffuse paths through reservoir reuse. Since we're not resampling specular paths, it makes more sense to allocate a higher percentage of our initial samples towards specular paths, since they don't have any form of ReSTIR to help them out unlike diffuse path. Additionally, sharp specular reflections are simply more detailed than blurry diffuse reflections, and therefore the denoiser needs more samples in order to properly reconstruct them.

## Saying Goodbye to ReSTIR

TODO

## Reflection Denoising

TODO

## CPU Performance Improvements

TODO

## Atmosphere and Skybox Lighting

TODO

## Future

TODO

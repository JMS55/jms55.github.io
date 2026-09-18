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

{{ figure(src="headline.png", caption="Zero Day rendered in realtime via Bevy Solari 0.20") }}

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

In ReSTIR, we aim to resample reservoirs (samples) from different pixels. E.g. you could have an "initial" reservoir, a "temporal" reservoir, and a "spatial" reservoir, and aim to select the best reservoir to keep and use for shading from the three via RIS.

It's important to note that each of these reservoirs comes from a different domain: the initial reservoir comes from the frame+pixel, the temporal reservoir comes from the previous frame+pixel, and the spatial reservoir comes from the current frame but from a neighboring pixel.

Since multiple domains are involved, we can't use constant MIS weights, and will have to use the balance heuristic again.

To use the balance heuristic, we need PDFs. But with reservoirs, although the regular PDFs can be replaced by unbiased contribution weights (UCWs), we have no method of generating cross-domain PDFs!

Instead of the balance heuristic, we'll have to instead use the _generalized balance heuristic_, which uses a combination of the UCWs and target functions of each reservoir to compute MIS weights.

Unfortunately, whereas the regular balance heuristic involves cross-domain PDFs that are pretty cheap to compute, the generalized balance heuristic involves computing cross-domain _target functions_, which are _much_ more expensive to compute since you need to trace new visibility rays, and even full paths.

### MIS in Solari

In realtime rendering, we often use hacks and approximations in the name of performance, and ReSTIR is no exception.

See, you don't strictly _need_ to use proper MIS weights. You can trade off various levels of performance/correctness (bias)/variance reduction by using purposefully incorrect MIS weights, depending on what you want to prioritize.

For instance, when computing MIS weights for reusing a temporal reservoir, you could do any of the following, in order of decreasing bias but increasing performance cost:
1. Use constant MIS weights
2. Use the generalized balance heuristic, but with various assumptions in the cross-domain target functions:
    * The material properties and world position of the current and previous frame pixels are the same
    * The intensity and color of the light the sample is from is the same in the current and previous frame (but watch out for lights that were despawned between frames)
    * When computing the target function of the canonical sample in the temporal reservoir's domain, if the light was visible to the pixel last frame, it's probably visible to the pixel again this frame (e.g. there are no occluders that got in the way since last frame) (note that you'll still want to trace a visibility ray when computing the target function of the temporal sample in the canonical domain)
3. Same as #2, but:
    * Store and look up the gbuffer and depth data from the previous frame
    * Keep a mapping of lights between frames, and look up the new light properties for the temporal sample
    * When computing the target function of the canonical sample in the temporal reservoir's domain, instead of skipping the visibility ray, trace a visibility ray using the current frame's TLAS as an approximation of the true visibility
4. Use the full, properly computed generalized balance heuristic / target functions, with no simplifications, by e.g. storing the previous frame's TLAS and BLAS data, and tracing the extra rays

In Bevy 0.19, Solari was (more or less) following #3. This led to bias, increased variance, and importantly, temporal lag. Shadows of moving objects often appeared to lag behind the object.

<video style="max-width: 100%; margin: var(--gap) var(--gap) 0 var(--gap); border-radius: 6px;" controls>
  <source src="shadow_lag.mp4" type="video/mp4">
</video>
<center>

*Lagging shadows behind the robot in Bevy 0.19*

</center>

In Bevy 0.20, we switched to using the previous frame's TLAS when testing visibility in the domain of the previous frame. This costs a decent amount of extra VRAM and complexity due to having to keep a second TLAS around, but in return it greatly reduces temporal shadow lag.

<video style="max-width: 100%; margin: var(--gap) var(--gap) 0 var(--gap); border-radius: 6px;" controls>
  <source src="fixed_shadow_lag.mp4" type="video/mp4">
</video>
<center>

*Much less shadow lag in Bevy 0.20, by using the previous frame's TLAS in the temporal reservoir MIS weights*

</center>

## Unified ReSTIR

For any readers who felt that the previous section was a lot of math, and not much engineering, rejoice! This section is going to be the complete opposite :)

### Historical Background

A few months ago, NVIDIA researchers released a really interesting paper called [ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced).

The paper is a followup to their 2022 paper on ReSTIR PT, this time detailing a lot of algorithmic changes to improve the performance of (the quite expensive) ReSTIR PT (among some other changes to e.g. reduce correlations).

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

The basic idea is that you start with your bog-standard pathtracer:
1. Perform NEE at the current path vertex (use RIS to choose a good sample for this)
2. Sample the BRDF, trace your next ray, continue if hit
3. Add emissive contribution, if any
4. Terminate in the world cache (if the path has been wide enough)
5. Perform Russian roulette to terminate long paths early
6. Loop

And then add ReSTIR, transforming the algorithm to look like this:
1. Initialize an empty reservoir
2. Perform NEE at the current path vertex
    * Resample new path contribution into the reservoir
3. Sample the BRDF, trace your next ray, continue if hit
4. Add emissive contribution, if any
    * Resample new path contribution into the reservoir
5. Terminate in the world cache (if the path has been wide enough)
    * Resample new path contribution into the reservoir
6. Perform Russian roulette to terminate long paths early
7. Loop

At the end, you're left with a single "initial" reservoir that you can then perform temporal and spatial resampling on, before finally using it to shade the pixel.

You are essentially resampling between different paths, regardless of whether those paths are primarily made of DI or GI contributions. It's all just one set of reservoirs.

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

Unlike the paper, I chose not to implement the hybrid shift, and stuck purely to the reconnection shift. Basically, when tracing paths, if it's from too sharp of a specular lobe, instead of putting the path into the reservoir to use for resampling, we simply write the path's radiance directly to the screen.

Unlike a wide lobe, it's not possible to use a sharply directional lobe with the reconnection shift. The hybrid shift, which conditionally switches to "random replay", can handle these types of lobes, but it involves re-tracing the path, which in my opinion is too expensive for a realtime renderer.

Additionally, I have a fairly high roughness gate on the primary vertex to guard resampling. I realized that just because you _can_ resample specular paths, doesn't mean you _should_.

As denoisers have improved recently, correlations from temporal and spatial reuse have become a big problem. Because we're reusing discrete samples, the denoiser sees the same samples more often than it otherwise would with plain importance sampling. The denoiser then thinks that this is detail worth preserving (it's a repeating pattern in an otherwise noisy signal), which leads to ugly denoising artifacts.

Smooth specular surfaces, unlike rough diffuse surfaces, have a small number of possible paths with strong contributions. So as soon as ReSTIR finds one good sample, it tends to propagate to all neighboring pixels, and stay there for a while without being overwritten by new samples. This over-reuse and correlated neighbor pixels confuses the denoiser.

> See [Compatibility-Guided Neighbor Selection for ReSTIR](https://www.youtube.com/watch?v=MCqnETw3l8A) for a good explanation of correlations in ReSTIR.

> As a side note, in the previous post in this series, I recommended halving the spatial sampling radius after failing to find a valid sample. It turns out, this is a bad idea, and I have reverted it. Doing this _also_ introduces problematic correlations, as many pixels end up reusing the same few samples within a small radius.

I tried a few different ways of mitigating correlations, e.g. by copying ReSTIR PT Enhanced's duplication maps, or scaling temporal confidence weight by surface roughness, but in the end the only foolproof method I came up with was a hard cutoff based on roughness.

One final note on the algorithm - if you've been paying attention, you may be wondering: "what kind of MIS weights do you use during resampling?"

After all, BRDF-sampled emissive hits at the primary vertex, NEE hits at the primary vertex, BRDF-sampled emissive hits at further vertices, NEE hits at further vertices, and the world cache termination are all _very_ different types of sampling strategies.

The answer is kind of complicated, and you can [take a look at the code](https://github.com/bevyengine/bevy/blob/37256568acc84fd2c8e3c801b906b7dd253f84f5/crates/bevy_solari/src/realtime/initial_path.wesl) for exact details, but it's actually very similar to the way you would typically do MIS in pathtracing.

### Benefits

By merging the two sets of reservoirs into one structure, we're saving 16 bytes/pixel compared to non-unified ReSTIR's two separate sets of reservoirs.

More important than the memory savings, there are a bunch of other improvements we either get for free, or can now afford to make due to combining the ReSTIR passes:
* In Solari 0.19's ReSTIR DI, we never traced BRDF rays to sample emissive lighting, instead relying only on NEE. Now that everything is unified into one pathtracer, we can trace and resample BRDF-based emissive lighting for free, since if we don't hit an emissive mesh, the ray is not wasted - it just becomes a GI path. This greatly improves direct lighting from nearby emissives.
* As discussed in the previous section, in Solari 0.19, we skipped tracing some visibility rays during MIS to increase performance, at the cost of some bias (shadows growing slightly larger and darker). Now, because we only have a single set of ReSTIR passes, we can reallocate the rays we saved from no longer having two ReSTIR passes to be used in MIS for the single pass, giving us fully unbiased (and lower variance) rendering. Overall we went from about 6-12 rays per pixel (3 DI, 3 GI, 0-6 specular BRDF + NEE), to 6-10 rays per pixel (2-6 pathtracing BRDF + NEE, 4 ReSTIR).
* Whereas before we never resampled specular GI, we can now resample sufficiently rough/glossy specular reflections. More on the caveats here in a bit.
* Much better light leak prevention, as instead of always terminating into the world cache after the first bounce like we used to for ReSTIR GI, it's now easy to simply continue tracing the path if terminating would risk leaking light.

Overall, we get some very nice quality wins, simplify the code, and reduce our memory usage. Performance tends to be either about the same, slightly slower, or slightly faster depending on the GPU and scene you're benchmarking.

{{ figure(src="pica_pica_reference.png", caption="Pathtraced PICA PICA scene") }}
{{ figure(src="before_unifed_pica_pica.png", caption="Separate ReSTIR DI and GI passes, no denoising") }}
{{ figure(src="unified_pica_pica.png", caption="Unified ReSTIR, no denoising - less biased and less visibility noise") }}

### Downsides

Note that there _are_ some downsides I've found from the unified ReSTIR algorithm.

Shadow boundaries where some pixels prefer GI, and some prefer DI, get a bit worse due to the single set of reservoirs.

Another downside is that a big unified pathtracing kernel uses more registers than separate kernels, which reduces occupancy and can hurt performance. It's not typically a huge loss, but it's something to be aware of.

The real issue is that we're no longer tracing paths for both the primary vertex's diffuse _and_ specular lobes every frame. Before, we had dedicated passes for diffuse and specular GI paths (really just a single-bounce for diffuse GI, we weren't doing a full path), but now we're only tracing a single path, with stochastic lobe selection for dielectric materials that have two lobes. Tracing two separate paths would be too expensive.

{{ figure(src="no_specular_boost.png", caption="Standard BRDF sampling - specular reflections are badly undersampled") }}

Remember from the last post that dielectric materials are implemented as a thin specular lobe layered _over_ a diffuse lobe. Depending on what angle you view it at, the top specular layer gets a certain percentage of the energy, with the rest being transmitted to the diffuse layer beneath it. When picking a BRDF lobe to follow to keep tracing the path, we can importance sample the two layers according to these percentages, biasing towards picking the lobe that will receive a higher amount of energy.

However, that only accounts for one part of the rendering equation. Incident radiance is _also_ an important quantity to account for. Let's say that for a given pixel we estimate that the specular layer will receive 70% of the incoming energy, while the diffuse layer only gets 30%. We should choose to sample the specular layer, right?

But what if the specular lobe is pointing in a direction where no light is coming from? 70% of 0 energy is still 0 - it would've been better to sample the diffuse lobe after all...

Really, we'd like to sample according to some estimate (obviously if we had a perfect, cheap predictor of the full rendering equation, there would be no need to trace paths at all) of the combined `BRDF * incident_radiance` product. This is called "path guiding", and Bevy doesn't currently have a way of doing that. We have to stick to the inferior BRDF-only sampling.

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

The logic is pretty simple: for smooth dielectric surfaces, ReSTIR will _already_ reduce noise from diffuse paths through reservoir reuse. Since we're not resampling specular paths, it makes more sense to allocate a higher percentage of our initial samples towards specular paths, since they don't have any form of ReSTIR to help them out unlike diffuse paths. Additionally, sharp specular reflections are simply more detailed than blurry diffuse reflections, and therefore the denoiser needs more samples in order to properly reconstruct them.

{{ figure(src="specular_boost.png", caption="With specular boost - noticeably cleaner reflections") }}

## Saying Goodbye to ReSTIR

So, we just spent all that time, and really I've spent the last year or two talking about and working to improve ReSTIR. That's a lot of time invested.

The thing is, a few weeks after landing the unified ReSTIR rewrite, I started to think... Do we even really need ReSTIR?

As great as ReSTIR is, it's actually quite expensive, costing 4 rays for resampling just two extra reservoirs (1 temporal, 1 spatial) in the current version of Solari.

And then you run into issues with correlations. As I talked about above, ReSTIR just... doesn't play well with current denoiser designs.

### What in the Denoiser?

I'm going to go on a bit of a tangent here, but bear with me.

When you think about it, what actually _is_ a denoiser doing anyways?

Denoising is essentially signal smoothing. The radiance of a pixel can be thought of as a function, which I'll simplify down to a 2d function for the purposes of this metaphor.

{{ figure(src="denoiser_function.svg", caption="Ground truth pathtracing - perfectly noise free, but impossible to construct") }}

However in pathtracing, it would be prohibitively expensive to calculate the full function. Instead, we sample the function, calculating its value at discrete points. This is what produces the noisy input we feed a denoiser.

{{ figure(src="denoiser_samples.svg", caption="What the pathtracer hands the denoiser: the function evaluated at a handful of discrete points") }}

The denoiser can then look at those points, and try to "guess", and fill in the rest of the function, usually by interpolating (averaging) neighboring points together.

{{ figure(src="denoiser_good.svg", caption="With enough samples, interpolating neighbors together lands close to the real function") }}

If the input samples are a good representation of the overall function, then the denoiser can do a pretty good job at getting close to what the function should be.

If the input samples _aren't_ a good representation, well, then the denoiser is going to do a poor job of guessing. This is where denoiser artifacts and blurriness come from.

{{ figure(src="denoiser_poor.svg", caption="With poorly selected samples, the denoiser guesses the function wrong") }}

Now, denoisers have an advantage - they can look at multiple pixels at once. The good thing is, lighting tends to be a pretty "smooth" signal. It fades out smoothly over a distance, and doesn't tend to change that drastically between pixels. So long as _some_ of the nearby pixels have a good estimate of the incoming radiance, modern denoisers can do a pretty solid job at reconstructing a smooth lighting signal.

Now, after reading that, does my description of denoisers sound... familiar?

It's very similar to what ReSTIR is doing. In ReSTIR, we're taking discrete paths, and shifting them to terminate at different pixels, re-evaluating how much radiance the path should give under the new conditions. We do this spatially and temporally, reusing paths between pixels.

{{ figure(src="bistro_noisy.png", caption="Bistro without ReSTIR, before denoising") }}
{{ figure(src="bistro_noisy_restir.png", caption="Bistro with ReSTIR, before denoising") }}

Denoisers do the same, just with path _radiance_, instead of the paths themselves.

The advantage of denoisers is that interpolating color data is pretty cheap, relative to shifting a whole path and having to trace new rays.

In ReSTIR, more than a single temporal and spatial sample is too expensive for realtime (that's 4 rays/pixel, which even then is a decent chunk of performance). Denoisers, on the other hand, through the use of CNNs and transformers, can afford to look at tons of other pixels.

Moreover, while denoisers can more easily cause bias, they don't cause any issues with correlations. And even with ReSTIR, we need a denoiser anyways, so - why not rely more heavily on the denoiser?

Testing I've done shows that current (v4.5) versions of DLSS-RR just... don't really need ReSTIR a lot of the time. Our GI signal is already dense enough for DLSS-RR to do a perfectly good job. Our DI signal is pretty undersampled if you have more than a few lights, as RIS is not sufficient, but even then it tends to manifest as shadows that fade out at a distance, which - ever used a shadow map?

{{ figure(src="bistro_denoised.png", caption="Bistro without ReSTIR, after DLSS-RR - missing some shadows") }}
{{ figure(src="bistro_restir_denoised.png", caption="Bistro with ReSTIR, after DLSS-RR") }}

If this still sounds like a crazy idea to you, consider - lots of games are currently doing quarter resolution and even variable resolution GI. Denoisers are already designed to deal with a fairly noisy signal - we don't need it to be _that_ dense. ReSTIR is often overkill.

With this in mind, I've made the decision to disable ReSTIR by default in Solari 0.20. There's no point in deleting it outright as it can help in tricky scenes, and it's still useful for DI for now, but users should consider whether they actually need it or not.

In the future, I'll be focusing more on ways to cheaply guide the initial sampling process, rather than focusing on making ReSTIR smarter at reusing those samples.

For instance, we can reallocate paths from pixels in neighborhoods with simple, easy to sample lighting conditions, and spend them on pixels with more complicated lighting conditions (variable rate pathtracing).

And I'm going to look into cheaper methods like MegaLights or light clustering to guide DI sampling, in addition to RIS.

## Reflection Denoising

In Solari 0.20, I spent a good few weeks working on improving the guide buffers we provide to DLSS-RR so that reflections are more temporally stable. A lot of this was motivated by testing in the newly-setup Zero Day scene.

A lot of this was trial and error via AI to make test scenes and generate comparison videos. I had a "feeling" that reflections were a bit blurry and shimmery, but it was hard to really tell for sure. I did a _lot_ of pixel peeping during this time.

The changes are pretty complicated, and not very scientific, but it boiled down to doing primary surface replacement (PSR) on more types of materials and with more guide buffers (depth, regular non-specular motion vectors, etc).

In previous versions of Solari, PSR was done only for perfectly smooth metals. Now we've expanded it to dielectrics and slightly less smooth metals, with pretty good results!

<video style="max-width: 100%; margin: var(--gap) var(--gap) 0 var(--gap); border-radius: 6px;" controls>
  <source src="reflection_denoising.mp4" type="video/mp4">
</video>
<center>

*More temporally stable reflections in Zero Day after improved PSR (left before, right after)*

*(recommended to view in fullscreen)*

</center>

Review [the code](https://github.com/bevyengine/bevy/pull/25423) if you're interested in exactly what changed.

I would love if NVIDIA could publish some official docs on what the right way to do PSR is with DLSS-RR.

## CPU Performance Improvements

Another big goal for me this development cycle was to improve Solari's CPU performance.

Bevy is very modular, and Solari is no exception. The entire raytracing scene code is one plugin, with the realtime lighting plugin being another, and the reference pathtracer a third plugin.

Previously, the raytracing scene code was functional, but very naive. It rebuilt the scene from scratch every frame, iterating over every mesh, light source, and material in the scene.

Now, with a _lot_ of ugly and careful code, and after a long time instrumenting things and comparing Tracy traces, Solari has much better CPU performance!

TODO: Before/after tracy screenshots

Similar to past efforts for the standard renderer over the last several Bevy releases, Solari now caches the entire scene, and does incremental updates to both the render world ECS and GPU buffers using tools like Bevy's change detection and [AtomicSparseBufferVec](https://docs.rs/bevy/latest/bevy/render/render_resource/struct.AtomicSparseBufferVec.html).

Additionally, TLAS builds are now much more GPU driven. Partially to avoid wgpu overhead, and mainly to avoid uploading a large amount of data to the GPU every frame, TLAS instances are now written to a buffer on the GPU via a compute shader.

Every frame, the compute shader iterates over the existing list of entity transforms on the GPU, and simply copies the transform and BLAS address of each entity into a new buffer. GPU -> GPU copies are much faster than doing a second set of CPU->GPU copies for the whole list of transforms!

```rust
@group(0) @binding(0) var<storage, read> transforms: array<array<vec4<f32>, 3>>;
@group(0) @binding(1) var<storage, read> blas_refs: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> instances: array<TlasInstance>;

struct TlasInstance {
    transform: array<vec4<f32>, 3>,
    custom_data_and_mask: u32,
    sbt_offset_and_flags: u32,
    blas_ref: vec2<u32>,
}

@compute @workgroup_size(64, 1, 1)
fn setup_tlas_instances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot = global_id.x;
    if slot >= arrayLength(&instances) || slot >= arrayLength(&blas_refs) || slot >= arrayLength(&transforms) {
        return;
    }

    instances[slot] = TlasInstance(
        transforms[slot],
        (slot & 0xFFFFFFu) | 0xFF000000u,
        0u,
        blas_refs[slot],
    );
}
```

## Atmosphere and Skybox Lighting

For the final major change in Solari 0.20, I added support for lighting from Atmosphere (Bevy's procedural skybox plugin), and EnvironmentMapLight (Bevy's traditional split-sum cubemaps, where mip 0 of the specular cubemap is basically the skybox).

It's pretty simple for now - just sample the cubemap when on ray miss, and fold that into the typical path contribution and resampling routines.

In the future, we'll want to do NEE against the cubemap and build a hierarchical CDF to accelerate sampling for NEE, but for now the simple method works pretty well.

{{ figure(src="atmosphere.png", caption="Procedural atmosphere rendering used for lighting via Solari") }}

This feature is part of a shift in my focus towards getting Solari to support more rendering features to bring it up to parity with Bevy's standard renderer. In particular, I'd like to add support to Solari for the remaining rect/point/spotlight types, animated meshes, and alpha masked and transparent/transmissive/refractive materials.

Light transport is good and all, and of course I will continue to experiment in that department and always work to improve performance, but I'd like Solari to reach a more usable state in the near future.

And with that, we're at the end of the blog post. Thanks for reading, and look forward to the release of Bevy 0.20 soon!

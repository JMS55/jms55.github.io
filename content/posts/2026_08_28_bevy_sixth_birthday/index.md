+++
title = "Bevy's Sixth Birthday - Lighting for All"
date = "2026-08-28"

[taxonomies]
tags = ["bevy", "bevy birthday"]
+++

> Written in response to [Bevy's Sixth Birthday](https://bevy.org/news/bevys-sixth-birthday).

### Introduction

It's the sixth year of Bevy, and now 4 years in total that I've been contributing to it!

It's been a pretty packed year, so unlike past years, I'm only going to cover what I've personally been doing this year.

So, what have I done?

### A Year in Review

#### Solari

4 years into Bevy, and I'm still heavily focused on 3d rendering.

If you've been following my work, it shouldn't come as a surprise that 90% of what I've done this year has been working on Solari.

For those of you who haven't been following, Bevy Solari is a realtime pathtracer aiming to be a high-end, forward-looking alternative to Bevy's main renderer.

I've written plenty of articles on Solari before (TODO: link to blog tag), but the TLDR is that I want to make lighting more accessible to beginners, and push the boundries of what Bevy's capable of. Rather than having to learn a mix of shadow maps, baking lightmaps, baking light probes, environment maps, screen space methods, what their pitfalls and limitations are, and spending hours tweaking things, pathtracing replaces all that complexity with a single, much easier to reason about lighting algorithm.

Solari has been a years long project, but it's finally becoming production ready, and I'm really happy with what we've ended up with!

It's been a long, _long_ journey through several different rewrites:
* Lumen/AMD GI like screen-space probes (diffuse GI only), a spatial hashing based world irradiance cache, and a spherical-harmonic based spatial denoiser
* The above, but now with radiance cascades added
* Complete rewrite, now ReSTIR DI + GI (still diffuse only) + the same world cache + DLSS-RR
* Added a seperate pathtracing pass dedicated just to indirect specular (ReSTIR DI now shades direct specular, but still resamples based on diffuse)
* Another complete rewrite, now with a single unified pathtracer tracing direct and indirect diffuse and specular lighting, and a single ReSTIR pass resampling everything using the reconnection shift
* DLSS-RR 4.5 has gotten so good, ReSTIR is now optional!

There's still a decent amount to do, notably supporting transparent materials and more types of light sources, but the core lighting algorithm is pretty solid!

I'm really proud of how everything's turned out. Expect another blog post on Solari soon for Bevy 0.20, where I'll cover things in more detail, and stay tuned for a formal publication coming at the end of the year that I can't talk about too much yet :)

#### dlss_wgpu

Of course, a key part of my pivot to per-pixel pathtracing in Solari was predicated on having a good denoiser (ideally one I didn't have to write myself). For that, NVIDIA's DLSS-RR and NRD were the best options available.

One of my first contributions to Bevy was a TAA implementation I wrote, so adding DLSS support was a nice blast from the past :)

Getting DLSS/FSR integration in Bevy is been something I have been eyeing for a long while, but getting them working with wgpu's resource tracking and automatic barrier system took some effort!

I ended up submitting a PR to wgpu to add a new transition_resources() function https://github.com/gfx-rs/wgpu/pull/6678 (which should probably have been named something like record_resource_usage). This lets you insert new [usage scopes](https://gpuweb.github.io/gpuweb/#usage-scope) into wgpu's resource tracker, enabling users to integrate vulkan/directx/metal libraries directly with wgpu.

Otherwise, there would be no (simple) way for users to e.g. insert a barrier and transition a texture from color attachment to shader read layout. This, along with some new as_hal() APIs, allows Bevy to now record depth/normal/color/etc data, pass it to DLSS, and then read DLSS's result back in wgpu-land.

I've packaged up the barrier code, C FFI, wgpu<->Vulkan interopt, and DLL-linking build scripts into an open source, Bevy-independent crated called [dlss_wgpu](https://github.com/bevyengine/dlss_wgpu). If you're looking to add DLSS-SR or DLSS-RR to your wgpu project, give it a try!

Back in Bevy land, DLSS-RR works as a great denoiser + upscaler for Solari, and DLSS-SR works as a better, mostly drop-in replacement for Bevy's TAA.

I'd love to add FSR4 and FSR4-RR support to Bevy as well, but unfortunately AMD has not released any Vulkan bindings for them.

#### CompressedImageSaver revamp

TODO

### Last year goal review
* Solari - yes obv
* Virtual geo - no
* More docs - no
* UI - somewhat
* Editor - no
* Material - no

### Next Year Goals

> I'd like to hear more about future rendering plans that might be more unknown to the broader community, if there is any? (not sure if you're a part of that but it seems like you've worked on every part of the renderer 😄 )

> I think thoughts on stability and reliability and well tested well used parts of bevy rendering could be good. Lots of software out there today, and I think people are starting to look for quality

> Testing and quality refinement strategies for rendering in general would be super interesting to read about

### AI (maybe)

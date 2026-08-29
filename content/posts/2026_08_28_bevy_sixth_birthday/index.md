+++
title = "Bevy's Sixth Birthday - Lighting and Libraries"
date = "2026-08-28"

[taxonomies]
tags = ["bevy", "bevy birthday"]
+++

> Written in response to [Bevy's Sixth Birthday](https://bevy.org/news/bevys-sixth-birthday).

### Introduction

It's the sixth year of Bevy, and now 4 years in total that I've been contributing to it!

It's been a pretty packed year, so unlike past years, I'm only going to cover what I've personally been working on this year.

### A Year in Review

#### Solari

4 years into Bevy, and I'm still heavily focused on 3d rendering.

If you've been following my work, it shouldn't come as a surprise that 90% of what I've done this year has been working on Solari.

For those of you who haven't been following, Bevy Solari is a realtime pathtracer aiming to be a high-end, forward-looking alternative to Bevy's main renderer.

I've written [plenty of articles on Solari](/tags/raytracing) before, but the TLDR is that I want to make lighting more accessible to beginners, and push the boundries of what Bevy's capable of. Rather than having to learn a mix of shadow maps, baking lightmaps, baking light probes, environment maps, screen space methods, what their pitfalls and limitations are, and spending hours tweaking things, pathtracing replaces all that complexity with a single, much easier to reason about lighting algorithm.

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

Of course, a key part of my pivot to per-pixel pathtracing in Solari was predicated on having a good denoiser (ideally one that I didn't have to write myself). For that, NVIDIA's DLSS-RR and NRD were the best options available.

One of my first contributions to Bevy was a TAA implementation I wrote, so adding DLSS support was a nice blast from the past :)

Getting DLSS/FSR integration in Bevy is been something I have been eyeing for a long while, but getting them working with wgpu's resource tracking and automatic barrier system took some effort!

I ended up submitting a PR to wgpu to add a new [`transition_resources()`](https://github.com/gfx-rs/wgpu/pull/6678) function (which in hindsight should probably have been named something like `record_resource_usage`). This lets you insert new [usage scopes](https://gpuweb.github.io/gpuweb/#usage-scope) into wgpu's resource tracker, enabling users to integrate vulkan/directx/metal libraries directly with wgpu.

Otherwise, there would be no (simple) way for users to e.g. insert a barrier and transition a texture from color attachment to shader read layout. This, along with some new as_hal() APIs, allows Bevy to now record depth/normal/color/etc data, pass it to DLSS, and then read DLSS's result back into wgpu-land.

I've packaged up the barrier code, C FFI, wgpu<->Vulkan interopt, and DLL-linking build scripts into an open source, Bevy-independent crated called [dlss_wgpu](https://github.com/bevyengine/dlss_wgpu). If you're looking to add DLSS-SR or DLSS-RR to your wgpu project, give it a try!

In Bevy, DLSS-RR works as a great denoiser + upscaler for Solari, and DLSS-SR works as a better, mostly drop-in replacement for Bevy's TAA.

I'd love to add FSR4 and FSR4-RR support to Bevy as well, but unfortunately AMD has not released any Vulkan bindings for them.

#### CompressedImageSaver Revamp

The other major project I worked on this year was revamping Bevy's texture compression pipeline.

Until recently, Bevy's only option for texture compression and mipmap generation was an old version of [Basis Universal](https://github.com/BinomialLLC/basis_universal). Basis Universal is sort of an intermediate format, that gets transcoded to BCn/ASTC at runtime. It's useful for web, as it lets you avoid shipping two sets of textures, but it's not so great for desktop and mobile. It's much better to just encode directly as BCn/ASTC for these platforms.

The problem is that until recently, there's been no good tool or library for compressing textures for different platforms. There are several disparate tools, each with their own weird input formats, quirks, and supported or unsupported target formats. Trying to wrap different tools in a unified interface was... not very easy, and I gave up trying.

However, @cwfitzgerald (a maintainer of wgpu) released [ctt](https://github.com/cwfitzgerald/ctt) this year - a Rust texture compression library that can take texture data of various layouts, generate mipmaps, encode into the various BCn/ASTC/etc formats (via FFI to all the different encoder libraries), super-compress with ZSTD for smaller disk sizes, and finally spit out a ktx2 file - all through one unified interface. It's a huge achievement, and I highly reccomend using it if you need texture compression. There are even C bindings available!

I took ctt, and used it as the new backend for CompressedImageSaver in Bevy. Look forward to much improved texture compression in Bevy 0.20!

Huge thanks to @cwfitzgerald for helping me integrate ctt, fixing bugs I reported, and for creating ctt in the first place.

### AI

TODO

### Questions & Answers

For this blog post, I solicited some questions from the Bevy community to answer (questions have been lightly edited for the purposes of a smoother reading experience).

> I'd like to hear more about future rendering plans that might be more unknown to the broader community, if there is any?

My personal rendering plans for next year are to continue wrapping up Solari, and potentially go back to virtual geometry. I have been thinking about building a material layering system, although there are some technical blockers holding that up at the moment.

In terms of broader plans, here's what I've been seeing discussed (keep in mind that none of these are commitments!):
* HDR display support (currently being worked on by @stuartparmenter!)
* Bevy's material API is too low level, and tied to the technical details around how the renderer works (e.g. APIs for supplying a fragment shader for deferred/forward/virtual geometry/etc). We need some higher level APIs that under the hood, we can then adapt to different rendering and data storage methods. GpuComponentArrayBuffer is a recent step in the right direction.
* Unifying the 2D and 3D renderers, and e.g. replacing Sprite with SpriteMesh
* Adding some sort of built-in API for texture/mesh streaming (mesh streaming could potentially be delegated to virtual geometry)
* More work on volumetric rendering
* As always, hoping that WebGPU finally ships everywhere, so we can deprecate WebGL2 and assume storage buffers and compute shaders everywhere (fingers crossed that this is the year!)

> I think thoughts on stability and reliability and well tested well used parts of Bevy rendering could be good. Lots of software out there today, and I think people are starting to look for quality

Yeah, this is a hard one. It's been discussed a _lot_ in the Bevy community. We're very aware of the issues users are facing.

Starting with reliability:

On one hand, shipping bugs, especially regressions, sucks and contributes towards a sense of unreliability that makes it hard for people to trust that Bevy is "production ready".

On the other hand, squashing every or even most bugs is not realistically something we can, or should really try to do.

Shipping with ~zero bugs on the level of e.g. SQLite would mean _drastically_ slowing down on developing new features. This would be a net negative overall, as an engine with no features is way less useful than a much more featureful engine that's a bit buggy.

Every new feature, especially in rendering, tends to involve a large rewrite of and intersection with existing systems, with more and more potential for breakage. We could introduce some new feature X, but actually under TAA there's an issue, or when using multiple cameras, or it used to work with shadow maps and then we change the shadow pass internals which breaks that interaction or regressed performance on certain GPU, etc.

So you could say, add more tests! And [we do](https://pixel-eagle.com/project/b04f67c0-c054-4a6f-92ec-f599fec2fd1d). It's helped a lot compared to when we had nothing, but there's only so much we can cover. We will never be able to cover every interaction between techniques, regressions on specific mobile devices with poor GPU drivers, etc.

The release candidate process we adopted last year _does_ help a lot with both reliability, and giving library authors time to update their plugins before the official release. But it's also brought it's own share of issues, e.g. Bevy 0.19 being very delayed from our typical 3 month schedule, in part due to a large amount of bugs we spent time fixing rather than sticking to the release schedule.

In terms of stability, I'd much rather make breaking changes then try for backwards compability. Game engines are huge, have a lot of surface area to cover (rendering, physics, ui, assets, artist tooling, etc), and are constantly evolving. I'd much rather improve things but cause some breakage, rather than try to introduce more mess trying to make things backwards compatible.

My advice for game developers: stick to a single version of Bevy, and backport individual parts if needed. The same advice applies to pretty much every game engine on the market, but one of Bevy's major benefits is modularity. Don't be afraid to pull in a new feature from a newer version via copy and pasting the plugin into your project, rather than trying to upgrade your whole game to a newer version of Bevy all at once.

> Testing and quality refinement strategies for rendering in general would be super interesting to read about

I talked about this a bit in the AI section above, but to me this comes down to good tooling and test scenes. Automatic screenshot and video comparisons, profiling runs, hot reloading, etc are essential to improving rendering. I need to be able to make a change, see what changed (both visually, and numerically compared to path-traced ground truth), how that affected perf, etc. And then do the same in 10 other scenes with varying levels of geometric complexity, size, material types, animations, lighting conditions, etc to ensure it's an improvement across the board.

For Solari, I've slowly been building up a set of test scenes and one-off AI generated tools for helping me with this. E.g. a debug overlay to run through a specific part of a scene while visualizing correlations on both smooth and rough dielectrics and metals, for tweaking ReSTIR heuristics.

> For me and I think a lot of other gamedevs, rendering is one of those things that is a voodoo sorcery black box where materials and meshes go in and pretty pictures come out.
> So, this may be a big ask, but my question is "as an engine dev do you have any resources to link or recommend that help crack open that black box and make it easier to understand?"

Honestly, not really! I did try and write some higher-level docs for the Bevy book this year, but never finished or merged them. For specific rendering topics, there's lots of individual papers and articles I could point at for, but I don't have anything for people just starting out.

Rendering is a really broad area! There's a lot of things that boil down to "it's done in this extremely complex, specialized way, because performance".

I would say, unless you're really interested in being a rendering developer, you probably don't need to learn the details! The goal of engines like Bevy is to do all that for you, so that you don't need to learn and implement it all from scratch yourself. Learning how to use the tools, as an artist, should suffice for most users.

If you _are_ interested in rendering development, my typical recommendation is:
* Read through a couple of recent [frame breakdowns](https://www.gamedevpensieve.com/graphics/3d/3d_frame-breakdown) to get a broad sense of how real games are rendered. If you see something you don't recognize, do a quick google to get an idea of what it is, and why the game does it that way.
* For Bevy specifically, come ask questions in #rendering or #rendering-dev in the Bevy discord! We're always happy to talk and help people understand things!

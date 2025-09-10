+++
title = "Hardware Raytracing in Bevy 0.17 (Solari)"
date = "2025-09-06"

[taxonomies]
tags = ["bevy", "raytracing"]
+++

# Hardware Raytracing in Bevy 0.17 (Solari)

## Introduction

Lighting is hard. Anyone who's tried to make a 3D scene look good knows the frustration of placing light probes, tweaking shadow cascades, and trying to figure out why their materials don't look quite right.

Over the past few years, real-time raytracing has gone from a research curiosity to a shipping feature in major game engines, promising to solve many of these problems by simulating how light actually behaves. With the release of v0.17, [Bevy](https://bevy.org) now joins that club with experimental support for hardware raytracing!

<video style="max-width: 100%; padding: var(--gap) var(--gap) 0 var(--gap);" controls>
  <source src="solari_recording.mp4" type="video/mp4">
</video>
<center>

*[PICA PICA scene by SEED](https://github.com/SEED-EA/pica-pica-assets)*

</center>

Back in early 2023, I [started](@/posts/2023_09_12_bevy_third_birthday/index.md#bevy-solari) an ambitious project called Solari to integrate hardware raytracing into Bevy's rendering pipeline. I was experimenting with [Lumen-style](https://youtu.be/2GYXuM10riw) screen space probes for global illumination, later extending it to use [radiance cascades](https://radiance-cascades.com).

These techniques, while theoretically sound, proved challenging to use in practice. Screen space probes were tricky to get good quality out of (reusing and reprojecting the same probe across multiple pixels is hard!), and radiance cascades brought its own set of artifacts and performance costs.

On top of the algorithmic challenges, the ecosystem simply wasn't ready. wgpu's raytracing support existed only as a work-in-progress PR that never got merged upstream. Maintaining a fork of wgpu (and by extension, Bevy) was time-consuming and unsustainable. After months of dealing with these challenges, I shelved the project.

In the 2 years since, I've learned a bunch more, raytracing has been upstreamed into wgpu, and raytracing algorithms have gotten much more developed. I've restarted the project with some new algorithms (ReSTIR, DLSS-RR), and soon it will be released as an official Bevy plugin!

In this post, I'll be doing a frame breakdown of how Solari works in Bevy 0.17, why I made certain choices, some of the challenges I faced, and some of the issues I've yet to solved.

## Why Raytracing for Bevy?

Before we start, I think it's fair to ask why an "indie" game engine needs high-end raytracing features that requires an expensive graphics card. The answer comes from my own experience learning 3D graphics.

Back when I was a teenager experimenting with small 3D games in Godot, I had a really hard time figuring out why my lighting looked so bad. Metallic objects didn't look reflective, scenes felt flat, and everything just looked wrong compared to the games I was playing.

I didn't understand that I was missing indirect light, proper reflections, and accurate shadows - and I had no idea I was supposed to bake lighting (which wouldn't have worked for my dynamic game anyway).

This is the core problem that raytracing solves for indie developers. Even if most players won't have hardware capable of running ray-traced effects, having a reference implementation of what lighting is *supposed* to look like is incredibly valuable.

With fully dynamic global illumination, reflections, shadows, and direct lighting, developers can see how their scenes should be lit. Then they can work backwards to replicate those results with baked lighting, screen-space techniques, or other cheaper approximations.

Without that reference, it's really hard to know what you're missing or how to improve your lighting setup. Raytracing provides the ground truth that other techniques are trying to approximate.

There's also the practical consideration that if Bevy ever wants to attract AAA game developers, we need these kinds of systems. Recent AAA games like DOOM: The Dark Ages and Cyberpunk 2077 rely heavily on raytracing, and artists working on these types of projects expect their tools to support similar techniques.

And honestly? It's just cool, and something I personally love working on :)

## Frame Breakdown

In its initial release, Solari supports raytraced diffuse direct (DI) and indirect lighting (GI). Light can come from either emissive triangle meshes, or analytic directional lights.

Direct lighting is handled via ReSTIR DI, while indirect lighting is handled by a combination of ReSTIR GI, and a world-space irradiance cache. Denoising is handled by DLSS Ray Reconstruction.

As opposed to coarse screen-space probes, per-pixel ReSTIR brings much better detail, along with being _considerably_ easier to get started with. I had my first prototype working in a weekend.

While I won't be covering ReSTIR from first principles, [A Gentle Introduction to ReSTIR:
Path Reuse in Real-time](https://intro-to-restir.cwyman.org) and [A gentler introduction to ReSTIR](https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir) are both really great resources. If you haven't played with ReSTIR before, I suggest giving them a skim before continuing with this post. Or continue anyways, and just admire the pretty pixels!

Let's get started.

### GBuffer Raster

TODO: Images

The first step of Solari is also the most boring: rasterize a standard G-buffer.

The G-buffer pass remains completely unchanged from standard Bevy (it's the same plugin). This might seem like a missed opportunity - after all, I could have used raytracing for primary visibility instead of rasterization - but there's an important reason I kept rasterization here.

By using raster for primary visibility, I maintain the option for people to use low-resolution proxy meshes in the raytracing scene, while still getting high quality meshes and textures in the primary view. The raster meshes can be full resolution with all their geometric detail, while the raytracing acceleration structure contains simplified versions that are cheaper to trace against. This hybrid approach gives the best of both worlds: detailed primary surfaces, with efficient secondary rays.

Bevy's G-buffer uses quite a bit of packing. The main attachment is a `Rgba32Uint` texture with each channel storing multiple values:

- **First channel**: sRGB base color and perceptual roughness packed as 4x8unorm
- **Second channel**: Emissive color stored as Rgb9e5
- **Third channel**: Reflectance, metallic, baked diffuse occlusion (unused by Solari), and an unused slot, again packed as 4x8unorm
- **Fourth channel**: World-space normal encoded into 24 bits via [octahedral encoding](https://www.jcgt.org/published/0003/02/01), plus 8 bits of flags meant for Bevy's default deferred shading (unused by Solari)

There's also a second `Rg16Float` attachment for motion vectors, and of course the depth attachment.

The G-buffer rendering itself uses with `multi_draw_indirect` to draw several meshes at once, using sub-allocated buffers. Culling is done on the GPU using [two-pass occlusion culling](@/posts/2024_06_09_virtual_geometry_bevy_0_14/index.md#culling-first-pass) against a hierarchal depth buffer. Textures are handled bindlessly, and we try to minimize overall pipeline permutations.

These combined techniques keep draw call overhead and per-pixel overdraw fairly low, even for complex scenes.

### Light Tile Presampling

Before we can start calculating direct light, we need





### Direct Lighting

> https://intro-to-restir.cwyman.org

After the G-buffer pass, Solari replaces Bevy's traditional shadow mapping and light accumulation with a ReSTIR-based direct lighting system. ReSTIR (Reservoir-based Spatiotemporal Importance Resampling) is a technique that allows efficient sampling of large numbers of lights by reusing samples across pixels and frames. The implementation follows the light-tile presampling approach from [Chris Wyman's HPG 2021 paper](https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf), which provides a solid foundation for handling large numbers of lights efficiently.

#### Light Tile Presampling[#](#light-tile-presampling)

The first step is a prepass that does light-tile presampling. The system picks random lights from the scene and random samples on each light (for emissive objects), packing this information as a `vec2<u32>` "light sample". The resolved light sample data gets stored in a compact format:

```wgsl
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
```

#### Two-Pass Resampling[#](#two-pass-resampling)

The actual ReSTIR process happens in two passes:

**First pass**: 32-sample RIS (Resampled Importance Sampling) using the presampled light blocks, followed by temporal resampling. I ended up using constant MIS weights for the temporal resampling since I didn't see much difference with the balance heuristic in practice.

**Second pass**: 1-sample spatial resampling in a 30-pixel radius. This is crucial for spreading good samples across the image. Without this spatial pass, you get really ugly artifacts under motion due to temporal reuse, and the overall quality suffers because there's no way to distribute good samples effectively.

#### Current Light Support[#](#current-light-support)

Right now the system handles directional lights and emissive meshes, but doesn't support point/spot lights or IBL yet. This is enough to demonstrate the core benefits of the approach, though there's obviously room for expansion.

#### Performance and Quality[#](#performance-and-quality)

Compared to traditional rasterized lighting, Solari can handle way more lights performance-wise, with nice effects like soft shadows and - most importantly - proper emissive area lights. That said, the quality is still a work in progress. There are lots of good ideas we can borrow from Unreal's Megalights system to improve things further.

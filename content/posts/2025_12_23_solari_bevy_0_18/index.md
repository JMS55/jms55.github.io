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

Summed together and denoised, we get full, pathtraced lighting, able to match an offline movie-quality pathtracer - but running much, much faster due to heavy temporal and spatial amortization.

Or at least, that's the theory.

In practice, all the amortization and shortcuts gives up some accuracy (making the result biased) in order to improve performance.

My goal with Solari is to get _as close as possible_ to the offline reference (zero bias), while getting "good enough" performance for realtime. Quality is the priority.

To that end, Bevy 0.18 brings many quality (and some performance) improvements to Solari:

* Specular material support
  * Solari 0.17 was diffuse only, which was a major limitation!
* Fixed the loss of brightness in the scene compared to the reference
* Eliminated bias from ReSTIR DI resampling
* Greatly reduced GI lag
* Greatly improved performance on larger scenes

## Specular Materials

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

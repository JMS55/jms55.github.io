+++
title = "Direct Lighting - Importance Sampling, RIS, and ReSTIR"
date = "2024-08-01"

[taxonomies]
tags = ["restir"]
+++

<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Introduction
### The Big Question
You have a scene. The scene is composed of a bunch of meshes with different materials, as well as a bunch of light sources, which are meshes whose triangles emit light. You also have a camera from which to view the scene from.

Given a specific point in the scene visible through the camera, we want to answer the question "what light is being emitted from light sources and directly hitting this point, without first bouncing off of another surface (direct light)".

Oh, and those lights sources? 3 million emissive triangles. We want to know how much direct light is hitting the point from _all_ 3 million light sources. The light sources are also moving around constantly, so you can't precompute anything.

How are we going to answer this question fast enough for real time rendering?

### Background
In 2020, Dartmouth college and NVIDIA researchers released a paper called ["Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting"](https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal). In it, they detailed a new technique for computing dynamic, raytraced direct lighting that they called "Reservoir-based Spatio-Temporal Importance Resampling" (ReSTIR). This technique was much faster than existing methods, and proved that realtime, fully dynamic direct lighting on current-gen consumer GPUs was feasible.

ReSTIR was pretty foundational for the field of realtime lighting, and after the first paper's publishing there were a large number of new papers exploring different ways of using and improving on ReSTIR.

At SIGGRAPH 2023, a group of these researchers published an [introductory course](https://intro-to-restir.cwyman.org) that covered the original ReSTIR paper as well as several of the more notable advancements and clarifications found in the three years since.

If you search online, you can also find a bunch of blog posts about ReSTIR. All these resources are really great (read them after reading this!), but both the course and many online articles are fairly math-heavy. The math _is_ really important to understand; you're not going to be able to implement ReSTIR without a solid grasp of it.

But I think that there's a lot of value in higher-level learning material that _avoids_ treating the math as individual numbers and symbols, and sticks to discussing what the math _represents_. Math provides the algorithm to model behavior, but the behavior itself, in my opinion, _should_ be discussed on it's own before you're ready to dive into how the math models it.

So to that goal, I'm writing this article to explain what problem ReSTIR is trying to solve, and what it's doing on a conceptual level. Less a lecture on math, and more a conversation. If you're like me and find that conversations are the best way for you to learn, then read on.

# The Problem
[Direct lighting equation, iterating over every light/point, monte carlo]

# Smarter Sampling
[Importance sampling, RIS]

# Sharing the Load
[ReSTIR]

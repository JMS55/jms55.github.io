+++
title = "Bevy's Fifth Birthday - Progress and Production Readiness"
date = "2025-09-02"

[taxonomies]
tags = ["bevy"]
+++

> Written in response to [Bevy's Fifth Birthday](https://bevy.org/news/bevys-fifth-birthday).

### Introduction

Welcome to the review of my third year of Bevy development!

After three years, I'm still enjoying Bevy as much as ever. Not only that, but development is the smoothest it's ever been!

As is my usual writing style, this post is going to be a bit dry and disjointed, and maybe not the most hype-oriented. It's not exactly what I aim for, but it seems to be how I end up writing things :). I _did_ try using an LLM to help with writing, but it was way too fawning, so in the end I've written this by hand (with the limited time and energy I have to perfect the word choice and structure).

While I _am_ really excited about Bevy, this year and every year, I'll leave hyping Bevy up to others, so go read their blog posts once they're posted to https://bevy.org! Consider this post more a reflection on my own experiences, rather than on Bevy as a project.

Anyways, let's talk about how this year went.

### My Stuff

The Bevy community has collectively landed a metric truckload of features and improvements this year. Like, just a mind-blowingly large amount.

The Bevy 0.15, 0.16, and 0.17 release notes do a good job of highlighting what's new in each release, so I'm going to give a brief overview of just the (rendering) work I did this year that I'm particularly proud of.

The biggest project for me this year has been the massive amounts of improvements I (@JMS55), @atlv24, and @SparkyPotato have landed for virtual geometry. When I wrote about Bevy's fourth birthday, we had landed the initial virtual geometry feature in Bevy 0.14. Since then, we've made numerous improvements to asset deserialization, compression, rasterization performance, LOD selection, LOD building, and culling in Bevy 0.15-0.17.

I'm optimistic that this year will be the year that we add streaming, improve CPU performance, and fix the remaining culling bugs. With any luck with asset processing (more on this later), we'll finally take virtual geometry out of experimental status later this year!

![Virtual geometry scene with thousands of dragon meshes](meshlet.png)

> Side note: Unlike the last few releases, I won't be writing a blog post about virtual geometry for Bevy 0.17. I didn't work on the BVH-culling PR for Bevy 0.17 (that was all @atlv24 and @SparkyPotato) due to a combination of burnout and life getting in the way. While I've taken a break from virtual geometry, I've started _another_ huge project: Solari.

Bevy Solari is a brand new crate for raytraced lighting coming in Bevy 0.17. While most of the work was technically done in Bevy 0.17, it's actually a ~2 year old project. I've mentioned in past blog posts how I started it, and then later abandoned it (which lead to me starting virtual geometry) after poor results, and issues with keeping forks of bevy/wgpu/naga_oil up to date.

Now that I'm taking a break from virtual geometry, and due to the introduction of some new algorithms and research papers, along with 2 additional years of learning under my belt and upstreamed raytracing in wgpu, I've restarted the project with a completely new approach. I'm really _super_ excited to share what we have so far, so expect a more detailed blog post about this soon!

![Solari demo scene](solari.png)

Like with Solari, DLSS integration is another abandoned project that I've revived thanks to work done in wgpu to enable interopt with underlying graphics APIs like Vulkan. Bevy 0.17 will be shipping support for DLSS (and DLSS-RR), alongside it's existing anti-aliasing options in MSAA, FXAA, SMAA, and TAA. NVIDIA users now have a great option for anti-aliasing, and much cheaper rendering via upscaling.

I also wanted to add FSR4 support, but sadly FSR4 was released as a DirectX-only SDK, without any Vulkan support. This would have meant redoing a lot of work, and wasn't going to be done in time for Bevy 0.17 (and I don't own an RX 9070 XT). Still, eventually we could add support for FSR and XeSS (and potentially MetalFX), now that the infrastructure for temporal upscaling is in place.

![DLSS demo](dlss.jpg)

The last major feature I landed this year was hooking up our existing GPU timestamps to Tracy, the profiling tool we use. Now Bevy users can see combined CPU and GPU bottlenecks in one place, which is super useful!

![Tracy screenshot showing GPU timings](tracy.png)

This year I would also like to shout-out several contributors (besides @altv24 and @SparkyPotato) I've been working with: @cart and @alice-i-cecile of course, as well as (in no particular order) @mockersf, @tychedelia, @Elabajaba, @DGriffin91, @IceSentry, @mate-h, @ecoskey, @NthTensor, @viridia, @pcwalton, and @ickshonpe, as well as @cwfitzgerald and @Vecvec for their work on wgpu. Without their help, none of this would have been possible!

### Unexpected Improvements

Along with the usual headline features, there's also been a lot of work done by the Bevy community this year that has ended up suprising me.

The major one would be required components. I was fairly skeptical of them when they were first introduced, and thought it wasn't really an "ECS" way of doing this. In retrospect, I was totally wrong. As a user, using required components is _way_ more pleasant than the older bundle-based API, and is easier to get started with. It ends up being _easier_ to explain to users to spawn and query an entity with a `Camera` component, rather than spawning with a `CameraBundle` and then querying for a `Camera`. As a plugin author, required components give me some peace of mind knowing that users can't easily add a component without adding its dependencies (and make the API a little nicer compared to bundles). There _is_ still some rough edges to sort out, mainly making some sort of priority system for required components to override each other (e.g. `Camera` requiring `Msaa::Sample4`, and then `TemporalAntiAliasing` being able to override that with `Msaa:Off`), but overall it was an unexpectedly nice improvement. Cart absolutely cooked with this change.

Retained rendering, and the other GPU-driven rendering parts have also worked out really well. Again some sharp edges with the APIs (mainly cleaning up render world entities being hard to write and easy to mess up), but these were hugely foundational changes, that overall landed really smoothly! I don't think the Bevy of 1-2 years ago would have landed these so easily, and they've _drastically_ improved performance.

The introduction of working groups was, to some extent, a big contributor of this. In the past I've complained about review speed for PRs, but this year it's been _way_ better. Working groups give a set of built-in reviewers for larger projects, and Alice's Monday merge train often ends up being the final maintainer-review-once-over before merging, which keeps PRs merging smoothly. It's been working quite well!

Similarly, having release notes in the main Bevy repo, and required as part of submitting PRs (something I've been advocating for!) has made the release process for Bevy 0.17 _so much easier_. Rather than having to crunch out release notes, changelogs, and showcases out at the end of the cycle (when we're all burnt out from writing PRs), writing them incrementally as a part of the PR process has been a huge time stress relief. As an unexpected benefit, it also makes reviewing PRs much easier, as it forces the author to write a good user-facing description of the changes for reviewers.

### Next Year?

The end of the birthday post is usually where I take some time to talk about what I'm planning to work on for the next year, but honestly I don't have a ton of plans at the moment.

Continuing virtual geometry and Solari is a given, but otherwise I don't have any other concrete goals in terms of features.

Neural-compressed textures would be cool, but is maybe a little too-researchy, and requires better asset processing APIs.

Writing more blog posts would be great, but I probably don't have it in me to write these more frequently. I _have_ been using a [Bluesky page](https://bsky.app/profile/jms5517.bsky.social) to document short progress snippets as I work on PRs, so maybe follow that if you're interested in my content.

I _would_ like to write more documentation this year though - both API docs, and module docs / Bevy book content. As Bevy is getting increasingly mature, docs have become one of the bigger sticking points. Rendering in particular needs a lot more docs, both because it's under-documented, and because it's a fairly arcane subject.

When I first started making 3d games, and later when I started working on rendering, I had absolutely no clue what to do. How to light a scene, how to write a custom material, what's important for rendering performance, how do I wrote my own rendering feature \<FOO\>, and more are all questions that would greatly benefit from some longer-form written documents.

As we start to run out of major rendering features, putting my energy towards writing more docs seems like a good way to move Bevy closer to being production-ready. I've already [started writing some stuff](github.com/bevyengine/bevy-website/pull/2195).

### Production Ready - What's Missing?

So instead of writing my plans for next year, let's talk about what I think Bevy is missing (besides docs). I don't necessarily plan or not plan on working on any of this myself, but here are the things that I feel make it hard to say "Just use Bevy, duh!"

**UI** continues to be a weak point. While `bevy_ui` is a great foundation for rendering UI (in large part to @ickshonpe's and the taffy team's heroic efforts), no third-party crate (including my own bevy_dioxus) has proven out a good high-level API for declaring and updating UI trees. BSN is coming soon, but it only solves the declarative part of UI, and not the reactivity part. Until we resolve this, it's hard to reccomend Bevy for UI-heavy games and apps, and more importantly, we can't build the-

**Editor** absence continues to be a big, big hole for Bevy. Not just in terms of being production ready, but I think the first release with an official editor is going to get an exponential influx of new users and eventually new contributors. Working on the Solari demo scene has made me feel the lack of an editor badly. It was quite frustrating trying to get the materials correct for everything without an editor. I _did_ work on a [prototype](https://github.com/bevyengine/bevy_editor_prototypes/pull/167) scene tree + inspector using a third-party BSN crate, but it was exceedingly difficult to write and understand, and I gave up on it. I'm really excited to come back to working on the editor once reactive UI lands.

**Asset processing** is another big bottleneck. Hard to say Bevy is production ready when the only texture compressor it has is BasisU. While cart added some asset processing APIs with Assets V2, it's clunky and dosen't support enough features. Trying to write a glTF -> virtual geometry processor proved to be impossible. Once cart is done with BSN and reactivity, I would like to see him go back to this area. I would also like to move away from recommending that users ship their games with glTF/glb scenes, and instead provide some kind of glTF -> BSN + seperate image/mesh assets importer, that we can then run further asset processing on. This is another area that would greatly benefit from having an Editor.

**Animation** isn't something I know a ton about, but after recently trying it out in Bevy, I can definitely say it's lacking. The API is quite clunky, with too many confusingly-named components and amount of entities needed.

**Custom Materials** in Bevy are currently servicable, but not enjoyable. Users have a lot of power, but that's because we don't really provide much in the way of customizable abstractions. Mostly on the shader side, but also partly on the Rust side, with users having to resort to `MeshTag` and `ShaderStorageBuffer` to get good performance. The Material API should be completely redesigned, unified across 3d/2d/UI, and made much easier to use for common use-cases. We've been throwing around ideas in the #rendering-dev channel on Discord, but nothing concrete yet. This is a good area to get involved in!

Overall, I do see paths to improving all of these areas over the next year (or likely two for animations and editor). I am a little disappointed with how long it has taken to land BSN, mostly with the opaqueness of the process (which cart has talked about), but I'm hopeful about the next steps!

See you next year!

+++
title = "Bevy's Fourth Birthday - A Year of Meshlets"
date = "2024-08-30"

[taxonomies]
tags = ["bevy"]
+++

> Written in response to [Bevy's Fourth Birthday](https://bevyengine.org/news/bevys-fourth-birthday).

### Introduction
The subtitle of this post is "Bevy's Fourth Birthday; Already???". I feel like I _just_ wrote Bevy's _third_ birthday reflections only a couple of months ago... time flies!

It's been an awesome year, with a lot accomplished. Lets talk about that.

### ~~A Year~~ 11 Months of Meshlets

What have I been doing in Bevy in the last year? The answer is learning and reimplementing the techniques behind Nanite (virtual geometry). That's mostly it.

No really - the first commit I can find related to Bevy's meshlet feature is dated September 30th 2023. That's a little less than 2 months since the time I wrote Bevy's third birthday post, and around 11 months before the time of this writing.

I _did_ work on some other stuff - PCF being the most notable feature, along with some optimizations like async pipeline compilation to prevent shader stutter, and some experimental work that didn't pan out like solari and an improved render graph. But the large majority of my time spent has been on meshlets. In fact, this is going to be the third post on my blog in total - the first being Bevy's third birthday post, and the second being a huge writeup on my initial learnings from implementing meshlets.

And I'm going to say it - I'm really proud of my work on this. It's an absolutely _massive_ project spanning so many different concepts. It's been immensely rewarding, but also immensely draining. I've felt like quitting at times, and questioned the value it provides given that it's an AAA focused feature for a non-AAA-ready engine. But I've stuck with the project, and right now I can say that it's been worth it. Maybe it's not production ready yet (it's definitely not). Maybe there's still a ton of major things left to do, let alone optimize and tweak. Maybe occlusion culling is broken and I'm really avoiding looking at it because it's going to be painful to debug and fix; who can say?

But I've learned a lot (really, a _lot_). It got referenced during a SIGGRAPH 2024 Advances in Real-Time Rendering in Games presentation. Brian Karis (the author of Nanite) mentioned that they enjoyed my blog post explaining it. Getting recognition and seeing people enjoy it has been awesome. And most of all, I'm immensely proud of myself and the work I put into it. Meshlets has been a journey, but a worthwhile one.

Needless to say, in the next year, expect even more meshlet work. A lot has already been done since my last blog post - you'll see some of that when Bevy 0.15 releases. Hopefully I'll continue to be able to avoid burnout.

### Bevy in General
Bevy 0.12, 0.13, and 0.14 were all released in the last year, and have brought an absolutely massive amount of improvements. Unlike last year, I don't have much I wanted to discuss here, but there's a few things I want to talk about, in no particular order.

Alice was hired (thank you sponsors) as Bevy's project manager. She's done an amazing job helping push forward PRs, coordinate developers, and generally get things done. I think I speak for all the Bevy devs when I say getting things done is nice. I'm looking forwards to more of that next year - thanks Alice!

One of Bevy's most requested features (including from me) is a GUI program (editor) for modifying, inspecting, and profiling scenes. A couple of months ago I volunteered to help coordinate and push forward editor-related work, and then uhh, pretty much stopped working on it a few weeks after. Turns out, I didn't have the motivation to do both editor work, and meshlet work. Meshlet work ended up winning out. Sorry to everyone I let down on that. I _am_ still excited to work on the editor, but unfortunately I've realized I'm not so motivated to work on more foundational work such as scene editing, asset processing, and especially UI frameworks.

Side note, I also released my own UI framework this year (competing with the tens of other Bevy UI projects). It's called [bevy_dioxus](https://github.com/JMS55/bevy_dioxus/blob/main/examples/demo.rs), and it builds on top of the excellent Dioxus library to provide reactivity. Rendering is handled by spawning bevy_ui entities. No documentation, but it's a fairly small amount of fairly clean code, and it's usable and integrates well with Bevy's ECS. No reinventing the wheel here. For a few weeks work, I'm pretty happy with the process of making it and how it turned out.

Rendering is in pretty good shape now. Still lots more to implement or improve, but it's pretty usable! Going forwards, it would be nice to put more focus on documentation, ease of use, and ergonomics. The Material/AsBindGroup API is pretty footgun-y and not always performant, and there's a general lack of documentation for lower-level APIs besides "ask existing developers how to use things". A new render graph that automatically handled resources could help a lot with this, and there's been some interest and design work in this space.

Assets and asset processing needs a _lot_ of work. Ignoring the editor (which will need to build on these APIs), Bevy still needs a lot of work on extending the asset processing API, and implementing asset workflows for baking lighting, compressing textures, etc. A real battle-tested end-to-end asset workflow, from artists to developers to built game, really needs developing. I'm hoping that this will be a bigger focus next year, in parallel with the editor.

### Solarn't
Last year I demo'd a realtime, fully dynamic raytraced GI solution I called Bevy Solari... and now a year later I've written nothing else on it, and a lot about virtual geometry. What gives? Well, I did work on it for a few months more, but the project kind of died for a variety of reasons.

I was using a custom, somewhat buggy fork of wgpu/naga/naga_oil, and it became very difficult to constantly rebase on top of Bevy's and those project's upstream branches. The approach I was using (screen space probes based on Lumen and GI-1.0, and later screen space radiance cascades) started souring on me for complexity and quality reasons. My world space radiance cache was completely broken and I lost motivation to work on it. And finally I ended up starting meshlets, and later transitioned all of my time to it. So, Solari is dead, at least for now.

I would definitely love to pick this back up, particularly as I start to feel like I need a break from meshlets. Nowadays I feel like ReSTIR-based techniques (ReSTIR DI and GI plus screen space denoisers) hold much more promise. DDGI is also a great solution that I initially discarded for quality reasons, but is pretty simple to implement, very easy to scale up or down in cost, and gives fairly decent results all things considered.

If raytracing gets upstreamed into wgpu, I would happily pick this projects back up.

### Writing
Last year I finally started a blog... but didn't end up writing much. Or I did, but 80% of it was concentrated into one really long, really time-consuming post. It took something like a month to write.

I also wrote a rather long post on reddit's /r/rust about things I disliked in Rust (after using it for so long, and recently using a lot of Java developing an enterprise application) (I do still love Rust, that hasn't changed). Surprisingly to me a lot of people liked it and it sparked some interesting discussions. Seperate from the post's contents, people also asked me if I had a blog where they could read more of my writing. I of course, had to tell them that yes I do, but it only has two posts.

Needless to say, this year, I'd like to try to blog more. I'm going to try to get more writing out, and focus less on quality and spending so much time editing. Starting of course, with this post.

With my new focus on spending less time writing, I'm ending this post now without trying for a conclusion that flows better. See everyone next year!

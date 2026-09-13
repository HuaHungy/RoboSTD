# RoboSTD Project Video — Storyboard, On-Screen Copy & Layout

> Goal: follow the PCD project's visual-first philosophy—show the problem and the Baseline/RoboSTD contrast immediately, then explain the method with a short diagram, and end with the strongest real-world and bias-mitigation results.
>
> Recommended main video length: **95–110 s**
>
> Format: **16:9, 1920×1080, 30 fps**
>
> The video should work **without narration**. Use large on-screen text, short subtitles, synchronized comparison clips, and light background music. A narrated version can use the same storyboard.

# Global visual language

Use the same color mapping as the paper:

```text
Blue    = Single-Arm / source skill
Red     = mirrored / contralateral skill
Orange  = LLM / coordination constraints
Green   = RoboSTD / final supervision
Gray    = Oracle / reference
```

Typography:

- Main title: 54–64 px
- Section title: 38–44 px
- Card title: 28–32 px
- Captions: 22–26 px
- Result numbers: 34–42 px bold

Transitions:
- mostly simple fade / slide
- avoid flashy effects
- use animated arrows to explain transfer and reconstruction

# 0:00–0:06 — Opening

## Layout

Centered title on white background.

Large title:

> **RoboSTD**

Subtitle:

> **Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring**

Bottom line:

> **From single-arm demonstrations to coordinated bimanual supervision**

Small badges at bottom:

```text
Sagittal-Plane Mirroring
+
LLM-Guided Spatio-Temporal Reconstruction
```

## Visual

Fade in a blue single arm on the left and two coordinated arms on the right.

Arrow:

> **Single → Dual**

# 0:06–0:18 — The Problem

## Layout

Split screen.

### Left 55%

Show the OXE trajectory-density visualization / single-arm workspace.

Header:

> **Abundant Single-Arm Data**

Small label:

> Spatially biased interaction coverage

### Right 45%

Show the π0 / ACT arm-side comparison.

Header:

> **Arm-Side Spatial Preference**

Bottom center:

> **Single-arm data do not directly provide balanced bimanual supervision.**

## Animated keywords

Appear one by one:

```text
Action-Space Gap
Geometric Validity
Inter-Arm Coordination
```

# 0:18–0:27 — What Single-to-Dual Requires

## Layout

Three horizontal cards.

### Card 1

Icon: mirrored robot arms

> **Physically Valid Transfer**

Small text:

> Observation · State · Action

### Card 2

Icon: two arms over shared workspace

> **Spatial Coordination**

Small text:

> Who acts where?

### Card 3

Icon: timeline / task stages

> **Temporal Coordination**

Small text:

> What happens first?

Bottom reveal:

> **RoboSTD addresses both transfer and coordination at the data level.**

# 0:27–0:43 — Stage 1: Sagittal-Plane Mirroring

## Layout

Use panel (b) of the method figure, enlarged.

Left:
> **Original Single-Arm Skill**

Animated blue trajectory.

Center:
> **Sagittal-Plane Mirroring**

Right:
> **Contralateral Pseudo-Demonstration**

Animated red trajectory.

## Three small boxes below

```text
Observation Mirroring
Proprioceptive Mapping
Action Mapping
```

## On-screen key line

> **Platform-aware transformation preserves observation–state–action correspondence.**

Optional formula in corner:

> \(\tilde{\tau}^{\bar{k}}=\mathcal M(\tau^k)\)

# 0:43–1:00 — Stage 2: LLM-Guided Reconstruction

## Layout

Left 35%:
task scene + instruction

Text:

> **Task Context**

Under it:

```text
Instruction
Objects
Workspace
```

Center 30%:
GPT / LLM icon

Header:

> **GPT-4.1**

Right 35%:
three stacked output cards.

### Card 1

> **Arm Assignment**

Small:

> Who executes each skill?

### Card 2

> **Precedence**

Small:

> Which skill comes first?

### Card 3

> **Conflict Constraints**

Small:

> What should not overlap?

Then animate arrows into a timeline.

Timeline caption:

> **Constraint-Guided Rearrangement**

Inactive arm icon:

> **Hold Current Pose**

Bottom:

> **Stage-Level Language Annotation**

Small:

> Current subtask · arm role · temporal stage

Final output label:

> **Pseudo-Bimanual Supervision**

# 1:00–1:11 — Q1: Does the Constructed Supervision Work?

## Layout

Use Fig. 4 bar chart or a simplified three-bar animation.

Large center numbers:

```text
Single-Arm       39.1%
RoboSTD          66.5%
Oracle           67.6%
```

Header:

> **10 RoboTwin 2.0 Tasks**

Bottom line:

> **50 single-arm demos/task → performance close to 100 native bimanual demos/task**

Do not animate all 10 task bars individually; reveal the average first, then briefly show the full plot.

# 1:11–1:27 — Real-World Low-Coordination Tasks

## Layout

Three equal-width columns.

### Column 1

Header:
> **Bowl Placement**

Two stacked clips:

```text
Single-Arm
RoboSTD
```

Small badge:
> Rigid object

### Column 2

> **Towel Storage**

Badge:
> Deformable object

### Column 3

> **Flower Arrangement**

Badge:
> Precision placement

Bottom-wide result:

> **Average Success: 40.0% → 77.3%**

Transition text:

> **But skill transfer alone is not enough when coordination becomes critical.**

# 1:27–1:43 — High Coordination: Cup Collection

## Layout

2×2 comparison grid:

```text
Single-Arm          RoboSTD w/o LLM
RoboSTD             Oracle
```

Header:

> **Cup Collection — Spatial Coordination**

Overlay small workspace divider.

Animate average count underneath each clip:

```text
2.00        2.76
4.22        3.92
```

Highlight RoboSTD in green.

Bottom line:

> **Arm assignment + conflict constraints coordinate both sides of the workspace.**

# 1:43–1:59 — High Coordination: Sandwich Making

## Layout

2×2 grid again:

```text
Single-Arm          RoboSTD w/o LLM
RoboSTD             Oracle
```

Header:

> **Sandwich Making — Long-Horizon Coordination**

Show stage numbers ① ② ③ ④ over the RoboSTD clip.

Small timeline:

```text
Stage 1 → Stage 2 → Stage 3 → Stage 4
```

Result badges:

```text
Completion:
55% → 63% → 74%
Oracle: 72.5%

Success:
26% → 30% → 50%
Oracle: 50%
```

Bottom line:

> **Stage-level language supervision helps organize multi-stage bimanual execution.**

# 1:59–2:12 — Q3: Arm-Side Preference Mitigation

## Layout

Match the structure of Fig. 6.

Top:
Single-Arm-finetuned trajectory density + arm-side bars

Bottom:
RoboSTD-based trajectory density + arm-side bars

Center arrow:

> **Bias Mitigation**

Two large result callouts:

### Left

> **Density Gap**  
> **45.8% → 25.6%**

### Right

> **Bias Gap**  
> **28.3 pp → 13.0 pp**

Bottom:

> **Broader left–right coverage without sacrificing the stronger right-arm performance**

# 2:12–2:20 — Final Takeaway

## Layout

Return to the overview teaser.

Center:

> **RoboSTD**

Three lines appear sequentially:

> **Reuse abundant single-arm demonstrations**

> **Construct executable pseudo-bimanual supervision**

> **Coordinate both arms across space and time**

Final tagline:

> **A data-centric bridge from Single to Dual**

Bottom buttons/icons:

```text
Paper   Code   Project Page
```

Fade out.

# Website Video Gallery Layout

In addition to the main project video, use short looping clips like the PCD page.

## Section title

# Robot Demos

Subtitle:

> **Single-Arm vs. RoboSTD across real-world bimanual tasks**

## Grid 1 — Direct comparison

Three columns:

```text
Single-Arm: Bowl Placement
+RoboSTD: Bowl Placement

Single-Arm: Towel Storage
+RoboSTD: Towel Storage

Single-Arm: Flower Arrangement
+RoboSTD: Flower Arrangement
```

Recommended layout per card:

```text
[Task name]
[Single-Arm clip | RoboSTD clip]
[one-line task type]
```

## Grid 2 — Spatial coordination

One wide 2×2 card:

```text
Cup Collection

Single-Arm          RoboSTD w/o LLM
RoboSTD             Oracle
```

Bottom badges:

```text
2.00   2.76   4.22   3.92
```

## Grid 3 — Temporal coordination

One wide 2×2 card:

```text
Sandwich Making

Single-Arm          RoboSTD w/o LLM
RoboSTD             Oracle
```

Bottom badges:

```text
Completion: 55% | 63% | 74% | 72.5%
Success:    26% | 30% | 50% | 50%
```

# Video editing rules

1. Keep comparison videos synchronized.
2. Use identical crop and scale for all conditions.
3. Do not hide failures; use representative complete episodes.
4. Keep labels on screen throughout the clip.
5. Autoplay, muted, looping for web cards.
6. Use MP4/H.264 for broad browser support; optionally provide WebM.
7. Keep each web demo clip around 6–12 s where possible.
8. Avoid narration-dependent information; every claim should be readable on screen.
9. Use no more than one headline and one metric badge per visual region.
10. Put detailed explanations below the videos rather than inside them.

# Design — GamiSenpai (Washi Craft Studio / 和紙クラフト)

A locked, authentic design system for GamiSenpai, tailored specifically for the ergonomics and soul of physical origami paper folding.

## Rationale: Why Washi Paper Craft Over Dark Theme?
1. **Visual Ergonomics**: The user is folding physical paper on a well-lit desk. A dark UI forces constant pupil dilation shock when alternating gaze between bright paper and a dark screen. Warm washi paper matches the physical environment.
2. **Cultural & Aesthetic Authenticity**: Origami (折り紙) is rooted in tactile Japanese paper craft. The interface should feel like an artisan's folding desk or a modern Japanese craft book (折り図 - Orizu) with crisp crease lines, sumi-ink typography, and traditional vermilion accents.

## Genre
editorial-craft (tactile, mindful, authentic)

## Macrostructure Family
- Interactive App / Learning Stage: 05 · Workbench (Craft Studio)
- Step Progression: Orizu Milestone Grid (折り図 6 Checkpoints)

## Theme — Washi & Shu-iro (和紙と朱色)
- `--color-paper`:       oklch(96.5% 0.012 85)   /* Natural Kinari washi paper cream (#faf7f2) */
- `--color-paper-2`:     oklch(99.0% 0.005 85)   /* Crisp white paper sheet surface (#ffffff) */
- `--color-paper-3`:     oklch(93.5% 0.018 85)   /* Folded crease depth & sub-panels (#ede8de) */
- `--color-stage-bg`:    #18191c                 /* Deep neutral backdrop for high-contrast camera feed */

- `--color-ink`:         oklch(22% 0.012 60)     /* Kurogane sumi charcoal ink (#21201e) */
- `--color-ink-muted`:   oklch(46% 0.015 60)     /* Secondary instructional ink (#5c5955) */
- `--color-ink-faint`:   oklch(62% 0.012 60)     /* Subtle metadata & diagram notes (#8a8680) */

- `--color-rule`:        oklch(87% 0.018 85)     /* Crisp paper fold crease hairline (#ded7ca) */
- `--color-rule-subtle`: oklch(92% 0.012 85)     /* Inner card dividers */

- `--color-accent`:      oklch(56% 0.22 28)      /* Traditional Vermilion Shu-iro 朱色 (#c8382d) */
- `--color-accent-deep`: oklch(44% 0.22 28)      /* 3D solid edge for tactile button press (#9b261b) */
- `--color-accent-cast`: oklch(56% 0.22 28 / 0.28);

- `--color-bamboo`:      oklch(58% 0.16 142)     /* Bamboo green 竹色 for verified steps (#337a4a) */
- `--color-bamboo-deep`: oklch(46% 0.16 142);
- `--color-bamboo-cast`: oklch(58% 0.16 142 / 0.24);

- `--color-indigo`:      oklch(45% 0.14 240)     /* Aizome indigo 藍染 for secondary links (#2b4b73) */
- `--color-amber`:       oklch(76% 0.16 75)      /* Warm amber for escape hatch guidance (#c28a1c) */
- `--color-focus`:       var(--color-accent)

## Typography
- Display: 'Zen Kaku Gothic New', 'Plus Jakarta Sans', 'Noto Sans Thai', sans-serif (weights 500, 700)
- Body: 'Plus Jakarta Sans', 'Noto Sans Thai', sans-serif (weights 400, 500)
- Mono: 'JetBrains Mono', monospace (weights 500, 600, tabular-nums for step telemetry)

## Motion & Tactile Feedback
- Easings: `cubic-bezier(0.2, 0.7, 0.3, 1)` (snappy paper press)
- Primary Button: Lift on hover (`translateY(-2px)` + shadow expansion), physical press DOWN on `:active` (`translateY(3px)` + edge compression)
- Checkpoint Pills: Clean transition between locked (folded cream), active (vermilion crease), and completed (bamboo green seal).

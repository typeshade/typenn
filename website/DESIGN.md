# TypeNN site design

This site follows the design language and product-writing rules from the official TypeShade
site. The source documents are preserved in `references/typeshade-site/`; this document records
the TypeNN-specific choices.

## Source

- TypeShade site design: `references/typeshade-site/DESIGN.md`
- TypeShade site product brief: `references/typeshade-site/PRODUCT.md`
- Source repository: `https://github.com/typeshade/typeshade.github.io`
- Source commit: `55bde15993e7df5015be20c01646399210be60ba`
- TypeNN implementation: `https://github.com/typeshade/typenn`

## Stack

- Astro renders the static English `/` and Korean `/ko/` pages.
- React supplies hydrated interactive islands for code tabs and the theme control.
- TailwindCSS v4 styles the page layout and responsive utilities.
- Ant Design v5 supplies the interactive tabs, buttons, and theme switch. Its palette and
  component tokens follow the TypeShade site's design language.
- IBM Plex Sans, IBM Plex Sans KR, and IBM Plex Mono are self-hosted under `public/fonts/`.

## Voice

- Write in plain sentences for TypeScript developers.
- Explain what TypeNN does, then link to source code for implementation detail.
- Keep each sentence to one idea. Use sentence-case headings.
- English is the default at `/`; Korean is a separately authored locale at `/ko/`.
- Use the exact TypeNN APIs and behavior documented in the source repository.
- Publish no performance numbers, hardware claims, or unsupported roadmap promises.

## Visual system

- Use the Ant Design v5 palette from TypeShade: primary `#1677ff`, page ground `#f5f5f5`,
  white containers, neutral text, and hairline borders.
- Use IBM Plex Sans for interface copy, IBM Plex Sans KR for Korean, and IBM Plex Mono for
  code and identifiers. Fonts are served locally.
- Use 6px control corners and 8px card and code-frame corners.
- Keep surfaces flat at rest. Use the primary blue for links, focus, and the main action.
- Support light and dark themes. Default to the system preference and provide a remembered
  theme control.
- Respect reduced-motion preferences. Keep focus rings visible and controls keyboard usable.

## Page structure

- The header identifies TypeNN and links to the execution model, compiler path, project status,
  GitHub, language route, and theme control.
- The first screen shows the TypeNN README operation and its actual TypeShade kernel together.
- The code control also exposes the documented MNIST training loop.
- Follow with the deferred Tensor graph, GPU-resident optimizer behavior, the
  TypeNN-to-TypeShade responsibility boundary, and current implementation limits.
- Use links to source files and TODO entries as evidence. Do not simulate compiler output,
  benchmark results, or GPU execution.
- Use no stock images or decorative graphics. Code and the documented execution path carry the
  product story.
- Keep the content in a centered 1320px shell with 16px mobile and 24px wider gutters. Split
  the hero at 64rem and stack it below that width.

## Accessibility

- Keep one `h1`, ordered section headings, landmarks, and a keyboard-visible skip link.
- Give language and theme controls accessible names and visible focus states.
- Keep code examples readable on narrow screens with horizontal scrolling.
- Keep status understandable without color alone.

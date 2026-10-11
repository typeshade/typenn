# TypeNN website

This directory is the source for [typenn.typeshade.dev](https://typenn.typeshade.dev), the
TypeNN project site. It is a static Astro site with React islands, Tailwind CSS v4, and
Ant Design v5.

## Develop

```sh
npm ci
npm run dev
```

English is served at `/` and Korean at `/ko/`. Run `npm run build` to generate both routes in
`dist/`.

`PRODUCT.md` and `DESIGN.md` describe the project's verified product scope and visual system.
The page's implementation claims link to the TypeNN source repository and its current README and
TODO. GPU software-renderer smoke results are identified as software-rendered; they are not
physical GPU benchmarks.

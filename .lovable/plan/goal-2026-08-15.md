Replace the placeholder home page with a minimal blank page

## Goal
Make the home page at `/` a clean, blank page by removing the placeholder image and centered content.

## What we will do
- Rewrite `src/routes/index.tsx` to render a minimal component that fills the viewport with a neutral background and no visible content or placeholder graphics.
- Keep the route structure and root layout intact (`src/routes/__root.tsx`, `src/router.tsx`, `src/routeTree.gen.ts`).
- Do not add navigation, text, hero, or imagery unless explicitly requested later.
- Ensure the page remains SSR-friendly and does not introduce client-side state or hydration issues.

## Files to change
- `src/routes/index.tsx` — replace the placeholder image with a blank page.

## Out of scope
- No new routes.
- No database, auth, or backend changes.
- No styling system changes beyond the index page itself.

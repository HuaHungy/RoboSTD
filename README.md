# RoboSTD Project Page

Vue + Vite academic project page for the anonymous RoboSTD ICRA submission.

The current layout follows a concise academic project-page structure with:

- Fixed navigation for Demos, Abstract, Method, Experiments, Conclusion and BibTeX
- No poster section until a poster asset is available
- Four headline result cards in the hero section
- Click-to-enlarge SVG figures and an auto-scrolling eight-task demo gallery

## Local preview

```powershell
npm install
npm run dev
```

Production build:

```powershell
npm run build
```

## Where to edit

- Paper text, metrics, links, captions and BibTeX: `src/content/project.js`
- Page section order and layout: `src/App.vue`
- Colors, spacing, responsive layout and animation: `src/styles.css`
- Click-to-enlarge paper figures: `src/components/ZoomableFigure.vue`
- Simulation and real-world auto-playing demo groups: `src/components/DemoMarquee.vue`
- Demo grouping and paper content: `src/content/project.js`
- GIF preparation script: `scripts/prepare_demos.py`
- Paper figures and PDF: `public/media/paper/`
- Optimized task GIFs: `public/media/demos/`

## Before public release

1. Replace `Anonymous Authors` and the anonymous-review note.
2. Enable the arXiv, code and dataset links in `src/content/project.js`.
3. Update the BibTeX author field and publication year.
4. Confirm the bundled PDF is the final public manuscript.
5. Run `npm run build` before deploying to GitHub Pages.

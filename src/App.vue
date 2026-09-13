<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import BibtexBlock from './components/BibtexBlock.vue'
import DemoMarquee from './components/DemoMarquee.vue'
import ProjectLinks from './components/ProjectLinks.vue'
import ZoomableFigure from './components/ZoomableFigure.vue'
import { project } from './content/project'

const lightboxFigure = ref(null)
const activeSection = ref('demos')
const isDarkMode = ref(false)
let revealObserver
let sectionObserver

function openFigure(figure) {
  lightboxFigure.value = figure
}

function closeFigure() {
  lightboxFigure.value = null
}

function handleKeydown(event) {
  if (event.key === 'Escape') closeFigure()
}

function readThemePreference() {
  try {
    return window.localStorage.getItem('robostd-theme')
  } catch {
    // Anonymous GitHub serves project pages in a sandboxed, opaque origin.
    return null
  }
}

function saveThemePreference(value) {
  try {
    window.localStorage.setItem('robostd-theme', value)
  } catch {
    // Theme still works for the current page even when storage is unavailable.
  }
}

function setTheme(isDark) {
  document.documentElement.dataset.theme = isDark ? 'dark' : 'light'
  saveThemePreference(isDark ? 'dark' : 'light')
}

function toggleTheme() {
  isDarkMode.value = !isDarkMode.value
  setTheme(isDarkMode.value)
}

watch(lightboxFigure, (figure) => {
  document.body.classList.toggle('lightbox-open', Boolean(figure))
})

onMounted(() => {
  const savedTheme = readThemePreference()
  isDarkMode.value = savedTheme === 'dark'
  setTheme(isDarkMode.value)
  window.addEventListener('keydown', handleKeydown)
  revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return
        entry.target.classList.add('is-visible')
        revealObserver.unobserve(entry.target)
      })
    },
    { threshold: 0.08, rootMargin: '0px 0px -8% 0px' },
  )
  document.querySelectorAll('[data-reveal]').forEach((element) => revealObserver.observe(element))

  const sections = ['demos', 'abstract', 'motivation', 'method', 'experiments', 'conclusion', 'bibtex']
    .map((id) => document.getElementById(id))
    .filter(Boolean)
  sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
      if (visible[0]) activeSection.value = visible[0].target.id
    },
    { rootMargin: '-18% 0px -68% 0px', threshold: [0.05, 0.2, 0.5] },
  )
  sections.forEach((section) => sectionObserver.observe(section))
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
  revealObserver?.disconnect()
  sectionObserver?.disconnect()
  document.body.classList.remove('lightbox-open')
})
</script>

<template>
  <main>
    <header class="top-nav">
      <a class="top-nav__brand" href="#top">
        <span class="brand-word">Robo</span><span class="brand-letter brand-letter--s">S</span><span class="brand-letter brand-letter--t">T</span><span class="brand-letter brand-letter--d">D</span>
      </a>
      <nav aria-label="Project page sections">
        <a href="#demos" :class="{ 'is-active': activeSection === 'demos' }">Demos</a>
        <a href="#abstract" :class="{ 'is-active': activeSection === 'abstract' }">Abstract</a>
        <a href="#motivation" :class="{ 'is-active': activeSection === 'motivation' }">Motivation</a>
        <a href="#method" :class="{ 'is-active': activeSection === 'method' }">Method</a>
        <a href="#experiments" :class="{ 'is-active': activeSection === 'experiments' }">Experiments</a>
        <a href="#conclusion" :class="{ 'is-active': activeSection === 'conclusion' }">Conclusion</a>
        <a href="#bibtex" :class="{ 'is-active': activeSection === 'bibtex' }">BibTeX</a>
      </nav>
      <div class="top-nav__actions">
        <a class="top-nav__paper" href="./media/paper/RoboSTD-paper.pdf" download>Paper ↗</a>
        <button
          type="button"
          class="theme-toggle"
          :aria-label="isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'"
          :aria-pressed="isDarkMode"
          @click="toggleTheme"
        >
          <svg v-if="isDarkMode" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
          </svg>
          <svg v-else viewBox="0 0 24 24" aria-hidden="true">
            <path d="M20.7 15.3A8.5 8.5 0 0 1 8.7 3.3 8.5 8.5 0 1 0 20.7 15.3Z" />
          </svg>
        </button>
      </div>
    </header>

    <section id="top" class="hero">
      <div class="hero__dots" aria-hidden="true"></div>
      <div class="page-container hero__content" data-reveal>
        <p class="venue-label">{{ project.meta.venue }} · {{ project.meta.year }}</p>
        <h1>
          <span class="hero__title-prefix">{{ project.hero.titlePrefix }}</span>
          <span
            v-for="letter in project.hero.titleAccent.split('')"
            :key="letter"
            class="hero__title-letter"
            :class="`hero__title-letter--${letter.toLowerCase()}`"
          >{{ letter }}</span>
        </h1>
        <h2>
          {{ project.hero.subtitle.lead }}
          <span class="hero__subtitle-word hero__subtitle-word--single">{{ project.hero.subtitle.single }}</span><span class="hero__subtitle-word hero__subtitle-word--to">{{ project.hero.subtitle.connector }}</span><span class="hero__subtitle-word hero__subtitle-word--dual">{{ project.hero.subtitle.dual }}</span>
          {{ project.hero.subtitle.tail }}
        </h2>
        <p class="hero__tagline">{{ project.hero.tagline }}</p>

        <div class="authors">
          <strong v-for="author in project.authors" :key="author">{{ author }}</strong>
          <span>{{ project.affiliation }}</span>
          <small>{{ project.authorNote }}</small>
        </div>

        <ProjectLinks :links="project.links" />

        <div class="review-badge">
          <span></span>{{ project.meta.status }}
        </div>

        <div class="hero-metrics">
          <article v-for="metric in project.metrics" :key="metric.label">
            <strong>{{ metric.value }}</strong>
            <span>{{ metric.label }}</span>
            <small>{{ metric.note }}</small>
          </article>
        </div>
      </div>
    </section>

    <section id="demos" class="section videos-section">
      <div class="page-container">
        <header class="module-heading" data-reveal>
          <h2>Demos</h2>
          <p>Representative rollouts from simulation and real-world bimanual manipulation</p>
        </header>

        <div v-for="group in project.videoGroups" :key="group.id" class="demo-group" data-reveal>
          <div class="demo-group__heading">
            <div>
              <span>{{ group.id === 'simulation' ? 'Simulation' : 'Real World' }}</span>
              <h3>{{ group.title }}</h3>
            </div>
            <p>{{ group.subtitle }}</p>
          </div>
          <DemoMarquee :tasks="group.tasks" :rows="group.rows" />
        </div>
        <p class="marquee-note">Auto-playing · Hover or focus to pause</p>
      </div>
    </section>

    <section id="abstract" class="section">
      <div class="page-container page-container--narrow">
        <header class="module-heading" data-reveal>
          <h2>Abstract</h2>
          <p>What problem does RoboSTD solve?</p>
        </header>
        <div class="abstract-layout" data-reveal>
          <div class="abstract-copy">
            <p v-for="paragraph in project.abstract" :key="paragraph">{{ paragraph }}</p>
          </div>
          <ZoomableFigure :figure="project.teaser" size="abstract" @open="openFigure" />
        </div>
      </div>
    </section>

    <section id="motivation" class="section section--blue">
      <div class="page-container">
        <header class="module-heading" data-reveal>
          <h2>Motivation</h2>
          <p>{{ project.motivation.title }}</p>
        </header>

        <div class="explain-layout" data-reveal>
          <div class="explain-copy">
            <p v-for="paragraph in project.motivation.paragraphs" :key="paragraph">
              {{ paragraph }}
            </p>
            <div class="motivation-challenges">
              <article v-for="challenge in project.motivation.challenges" :key="challenge.title">
                <strong>{{ challenge.title }}</strong>
                <p>{{ challenge.question }}</p>
              </article>
            </div>
            <p class="motivation-closing">{{ project.motivation.closing }}</p>
          </div>
          <div class="motivation-visual">
            <ZoomableFigure :figure="project.motivation.figure" size="compact" @open="openFigure" />
            <div class="key-numbers">
              <span><strong>28.6 pp</strong> data-side spatial gap</span>
              <span><strong>9.6 pp</strong> π0 arm-side success gap</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section id="method" class="section">
      <div class="page-container">
        <header class="module-heading" data-reveal>
          <h2>{{ project.method.title }}</h2>
          <p>{{ project.method.lead }}</p>
        </header>

        <div class="method-copy" data-reveal>
          <div class="method-stages">
            <article v-for="stage in project.method.stages" :key="stage.title" class="method-stage">
              <h3>{{ stage.title }}</h3>
              <p>{{ stage.body }}</p>
              <span class="method-stage__output"><b>Output:</b> {{ stage.output }}</span>
            </article>
          </div>
          <ul>
            <li v-for="point in project.method.points" :key="point.label">
              <span>{{ point.label }}</span>
              <strong>{{ point.title }}</strong>
            </li>
          </ul>
        </div>

        <div data-reveal>
          <ZoomableFigure :figure="project.method.figure" size="method" @open="openFigure" />
        </div>
      </div>
    </section>

    <section id="experiments" class="section section--soft">
      <div class="page-container">
        <header class="module-heading" data-reveal>
          <h2>Experiments</h2>
          <p>{{ project.experimentsLead }}</p>
        </header>

        <div class="experiment-list">
          <article
            v-for="experiment in project.experiments"
            :key="experiment.title"
            class="experiment-row"
            data-reveal
          >
            <div class="experiment-copy">
              <span>{{ experiment.label }}</span>
              <h3>{{ experiment.title }}</h3>
              <p v-for="paragraph in experiment.paragraphs" :key="paragraph">{{ paragraph }}</p>
              <p v-if="experiment.note" class="experiment-note">{{ experiment.note }}</p>
              <div class="highlight-list">
                <strong v-for="highlight in experiment.highlights" :key="highlight.label + highlight.value">
                  <span>{{ highlight.label }}</span>
                  <em>{{ highlight.value }}</em>
                </strong>
              </div>
            </div>
            <ZoomableFigure :figure="experiment.figure" size="experiment" @open="openFigure" />
          </article>
        </div>
      </div>
    </section>

    <section id="conclusion" class="section conclusion-section">
      <div class="page-container page-container--narrow">
        <header class="module-heading" data-reveal>
          <h2>Conclusion</h2>
          <p>{{ project.conclusion.lead }}</p>
        </header>

        <div class="conclusion-card" data-reveal>
          <div class="conclusion-copy">
            <p v-for="paragraph in project.conclusion.paragraphs" :key="paragraph">
              {{ paragraph }}
            </p>
          </div>
          <div class="conclusion-takeaways">
            <span v-for="(takeaway, index) in project.conclusion.takeaways" :key="takeaway.label">
              <b>0{{ index + 1 }}</b>
              <strong>{{ takeaway.label }}</strong>
              <em>{{ takeaway.text }}</em>
            </span>
          </div>
        </div>
      </div>
    </section>

    <section id="bibtex" class="section section--blue">
      <div class="page-container page-container--narrow">
        <header class="module-heading" data-reveal>
          <h2>BibTeX</h2>
          <p>Citation information will be updated after anonymous review</p>
        </header>
        <div data-reveal>
          <BibtexBlock :value="project.bibtex" />
        </div>
      </div>
    </section>

    <footer>
      <div class="page-container">
        <strong><span class="brand-word">Robo</span><span class="brand-letter brand-letter--s">S</span><span class="brand-letter brand-letter--t">T</span><span class="brand-letter brand-letter--d">D</span></strong>
        <p>Zero-Shot Single-to-Dual Transfer for Bimanual Learning</p>
      </div>
    </footer>

    <Transition name="lightbox">
      <div
        v-if="lightboxFigure"
        class="figure-lightbox"
        role="dialog"
        aria-modal="true"
        :aria-label="lightboxFigure.alt"
        @click.self="closeFigure"
      >
        <button type="button" class="figure-lightbox__close" aria-label="Close image" @click="closeFigure">
          ×
        </button>
        <div class="figure-lightbox__panel">
          <img :src="lightboxFigure.src" :alt="lightboxFigure.alt" />
          <p>{{ lightboxFigure.caption }}</p>
        </div>
      </div>
    </Transition>
  </main>
</template>

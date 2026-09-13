<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  tasks: { type: Array, required: true },
  rows: { type: Number, default: 1 },
})

const taskRows = computed(() => {
  if (props.rows < 2) return [props.tasks]

  return props.tasks.reduce(
    (groups, task, index) => {
      groups[index % 2].push(task)
      return groups
    },
    [[], []],
  )
})

const isPaused = ref(false)
let resumeTimer

function pause() {
  isPaused.value = true
  window.clearTimeout(resumeTimer)
}

function resume() {
  window.clearTimeout(resumeTimer)
  resumeTimer = window.setTimeout(() => {
    isPaused.value = false
  }, 180)
}

onMounted(resume)
onBeforeUnmount(() => window.clearTimeout(resumeTimer))
</script>

<template>
  <div
    class="video-marquee"
    :class="{
      'video-marquee--dual': props.rows >= 2,
      'video-marquee--paused': isPaused,
    }"
    @mouseenter="pause"
    @mouseleave="resume"
    @focusin="pause"
    @focusout="resume"
  >
    <div v-for="(row, rowIndex) in taskRows" :key="rowIndex" class="video-marquee__row">
      <div class="video-marquee__track">
        <template v-for="loop in 2" :key="loop">
          <article
            v-for="task in row"
            :key="rowIndex + '-' + loop + '-' + task.title"
            class="video-card"
            :aria-hidden="loop === 2"
          >
            <img :src="task.src" :alt="task.title" loading="lazy" />
            <div>
              <strong>{{ task.title }}</strong>
              <span>{{ task.subtitle }}</span>
            </div>
          </article>
        </template>
      </div>
    </div>
  </div>
</template>

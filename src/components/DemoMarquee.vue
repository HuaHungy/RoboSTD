<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  tasks: { type: Array, required: true },
  rows: { type: Number, default: 1 },
})

function shuffle(items) {
  const result = [...items]

  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1))
    ;[result[index], result[swapIndex]] = [result[swapIndex], result[index]]
  }

  return result
}

function hasAdjacentDuplicate(row) {
  if (row.length < 2) return false

  return row.some((task, index) => {
    const nextTask = row[(index + 1) % row.length]
    return task.title === nextTask.title
  })
}

function buildTaskRows(tasks, rowCount) {
  if (rowCount < 2) return [shuffle(tasks)]

  let lastRows = [[], []]

  for (let attempt = 0; attempt < 300; attempt += 1) {
    const rows = [[], []]

    shuffle(tasks).forEach((task, index) => {
      rows[index % 2].push(task)
    })
    lastRows = rows

    if (rows.every((row) => !hasAdjacentDuplicate(row))) return rows
  }

  return lastRows
}

const taskRows = computed(() => buildTaskRows(props.tasks, props.rows))

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
            :key="rowIndex + '-' + loop + '-' + task.src"
            class="video-card"
            :aria-hidden="loop === 2"
          >
            <img :src="task.src" :alt="task.title" loading="lazy" />
            <div>
              <strong>{{ task.title }}</strong>
            </div>
          </article>
        </template>
      </div>
    </div>
  </div>
</template>

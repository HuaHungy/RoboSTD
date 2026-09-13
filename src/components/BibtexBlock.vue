<script setup>
import { ref } from 'vue'

defineProps({
  value: { type: String, required: true },
})

const copied = ref(false)
let timeoutId

async function copyBibtex(value) {
  try {
    await navigator.clipboard.writeText(value)
  } catch {
    const textarea = document.createElement('textarea')
    textarea.value = value
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    textarea.remove()
  }
  copied.value = true
  window.clearTimeout(timeoutId)
  timeoutId = window.setTimeout(() => {
    copied.value = false
  }, 1800)
}
</script>

<template>
  <div class="bibtex-card">
    <div class="bibtex-card__bar">
      <span>BibTeX</span>
      <button type="button" @click="copyBibtex(value)">
        {{ copied ? 'Copied' : 'Copy citation' }}
      </button>
    </div>
    <pre><code>{{ value }}</code></pre>
  </div>
</template>

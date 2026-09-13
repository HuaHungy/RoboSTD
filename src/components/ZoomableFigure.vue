<script setup>
defineProps({
  figure: { type: Object, required: true },
  size: { type: String, default: 'default' },
})

defineEmits(['open'])
</script>

<template>
  <figure class="zoomable-figure" :class="`zoomable-figure--${size}`">
    <button type="button" :aria-label="`Enlarge ${figure.alt}`" @click="$emit('open', figure)">
      <span class="zoomable-figure__image">
        <img :src="figure.src" :alt="figure.alt" loading="lazy" />
      </span>
      <span class="zoomable-figure__hint">
        <span>＋</span> Click to enlarge
      </span>
    </button>
    <figcaption>{{ figure.caption }}</figcaption>
  </figure>
</template>

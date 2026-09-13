<script setup>
defineProps({
  name: { type: String, required: true },
})
</script>

<template>
  <svg v-if="name === 'paper'" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M6 3h9l3 3v15H6z" />
    <path d="M15 3v4h4M9 11h6M9 15h6" />
  </svg>
  <svg v-else-if="name === 'archive'" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 6h16v14H4zM3 3h18v4H3zM9 11h6" />
  </svg>
  <svg v-else-if="name === 'code'" viewBox="0 0 24 24" aria-hidden="true">
    <path d="m9 7-5 5 5 5M15 7l5 5-5 5M13 4l-2 16" />
  </svg>
  <svg v-else viewBox="0 0 24 24" aria-hidden="true">
    <ellipse cx="12" cy="5" rx="7" ry="3" />
    <path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" />
  </svg>
</template>

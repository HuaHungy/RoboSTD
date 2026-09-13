<script setup>
import ProjectIcon from './ProjectIcon.vue'

defineProps({
  links: { type: Array, required: true },
})
</script>

<template>
  <div class="project-links" aria-label="Project resources">
    <a
      v-for="link in links"
      :key="link.label"
      class="resource-link"
      :class="{ 'resource-link--placeholder': link.disabled || link.href === '#' }"
      :href="link.href"
      :download="link.download || undefined"
      :aria-disabled="link.disabled || link.href === '#'"
      @click="(link.disabled || link.href === '#') && $event.preventDefault()"
    >
      <ProjectIcon :name="link.icon" />
      <span>
        <strong>{{ link.label }}</strong>
        <small v-if="link.note">{{ link.note }}</small>
      </span>
    </a>
  </div>
</template>

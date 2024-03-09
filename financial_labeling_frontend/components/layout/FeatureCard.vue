<template>
  <v-card>
    <v-img :src="imageSrc" height="200px" />
    <v-card-title primary-title class="layout justify-center">
      <div class="headline text-xs-center font-weight-bold mb-2 justify-center">
        {{ title }}
      </div>
    </v-card-title>
    <v-card-text class="subtitle-1 text-xs-center layout justify-center">
      {{ text }}
    </v-card-text>
  </v-card>
</template>

<script lang="ts">
import Vue from 'vue'

export default Vue.extend({
  props: {
    title: {
      type: String,
      default: '',
      required: true
    },
    text: {
      type: String,
      default: '',
      required: true
    },
    imageSrc: {
      type: String,
      default: '',
      required: true
    }
  }
})
</script>

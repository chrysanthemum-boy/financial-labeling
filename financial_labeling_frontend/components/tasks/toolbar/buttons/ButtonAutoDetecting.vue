<template>
    <v-tooltip bottom>
      <template #activator="{ on }">
        <v-btn icon v-on="on" @click="$emit('click:label-clear')">
          <v-icon>
            {{ mdiDeleteOutline }}
          </v-icon>
        </v-btn>
      </template>
      <span>Auto Detecting</span>
    </v-tooltip>
  </template>
  
  <script>
//   import { mdiDeleteOutline } from '@mdi/js'
  
  export default {
    data() {
      return {
        mdiDeleteOutline
      }
    },
    methods: {

    }
  }
  </script>
  
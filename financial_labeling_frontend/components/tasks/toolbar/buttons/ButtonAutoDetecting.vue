<template>
  <v-tooltip bottom>
    <template #activator="{ on }">
      <v-btn icon v-on="on" @click="handleClick">
        <v-icon>
          {{ mdiLabelMultipleOutline }}
        </v-icon>
      </v-btn>
    </template>
    <span>Auto Detecting</span>
  </v-tooltip>
</template>

<script>
import { mdiLabelMultipleOutline } from '@mdi/js';

export default {
  data() {
    return {
      mdiLabelMultipleOutline
    }
  },
  methods: {
    handleClick() {
      // 触发两个不同的事件
      this.$emit('click:detecting');
      // this.$emit('click:segment');
    }
  }
}
</script>
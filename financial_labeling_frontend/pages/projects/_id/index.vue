<template>
  <v-card>
    <v-card-title>
      {{ $t('projectHome.welcome') }}
    </v-card-title>
    <main-card :is-project-admin="isProjectAdmin" :project="currentProject" />
    
  </v-card>
</template>

<script>
import { mapGetters } from 'vuex'
import MainCard from '@/components/layout/MainCard'

export default {
  components: {
    MainCard
  },
  layout: 'project',


  middleware: ['check-auth', 'auth', 'setCurrentProject'],

  validate({ params }) {
    return /^\d+$/.test(params.id)
  },



  data() {
    return {
      drawerLeft: null,
      isProjectAdmin: false
    }
  },

  computed: {
    ...mapGetters('projects', ['currentProject'])
  },

  async created() {
    const member = await this.$repositories.member.fetchMyRole(this.$route.params.id)
    this.isProjectAdmin = member.isProjectAdmin
  }

}
</script>

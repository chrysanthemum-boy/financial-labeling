<template>
    <v-container>
        <v-layout column wrap class="my-5" align-center>
            <v-flex xs12>
                <v-container grid-list-xl>
                    <v-layout wrap align-center>
                        <v-flex xs12 md3 @click="toLabeling">
                            <feature-card 
                                :image-src="require(`~/assets/start-annotation.png`)" 
                                :title= "$t('home.startAnnotation')"
                                />
                        </v-flex>
                        <v-flex v-for="(item, i) in filteredItems" :key="i" xs12 md3
                    @click="$router.push(localePath(`/projects/${$route.params.id}/${item.link}`))">
                            <feature-card 
                                :image-src="require(`~/assets/${item.imageSrc}`)" 
                                :title="item.text"
                                />
                        </v-flex>
                    </v-layout>
                </v-container>
            </v-flex>
        </v-layout>
    </v-container>
</template>

<script>
import {
    mdiAccount,
    mdiBookOpenOutline,
    mdiChartBar,
    mdiCog,
    mdiCommentAccountOutline,
    mdiDatabase,
    mdiLabel,
    mdiPlayCircleOutline
} from '@mdi/js'

import FeatureCard from './FeatureCard.vue'
import { getLinkToAnnotationPage } from '~/presenter/linkToAnnotationPage'

export default {
    components: {
        FeatureCard
    },
    props: {
        isProjectAdmin: {
            type: Boolean,
            default: false,
            required: true
        },
        project: {
            type: Object,
            default: () => { },
            required: true
        }
    },

    data() {
        return {
            selected: 0,
            mdiPlayCircleOutline
        }
    },

    computed: {
        filteredItems() {
            const items = [
                {
                    icon: mdiDatabase,
                    text: this.$t('dataset.dataset'),
                    link: 'dataset',
                    isVisible: true,
                    imageSrc: 'database.png',
                },
                {
                    icon: mdiLabel,
                    text: this.$t('labels.labels'),
                    link: 'labels',
                    isVisible:
                        (this.isProjectAdmin || this.project.allowMemberToCreateLabelType) &&
                        this.project.canDefineLabel,
                    imageSrc: 'labels.png',
                },
                {
                    icon: mdiAccount,
                    text: this.$t('members.members'),
                    link: 'members',
                    isVisible: this.isProjectAdmin,
                    imageSrc: 'members.png',
                },
                {
                    icon: mdiCommentAccountOutline,
                    text: this.$t('comments.comments'),
                    link: 'comments',
                    isVisible: this.isProjectAdmin,
                    imageSrc: 'comments.png',
                },
                {
                    icon: mdiBookOpenOutline,
                    text: this.$t('guideline.guideline'),
                    link: 'guideline',
                    isVisible: this.isProjectAdmin,
                    imageSrc: 'guide.png',
                },
                {
                    icon: mdiChartBar,
                    text: this.$t('statistics.statistics'),
                    link: 'metrics',
                    isVisible: this.isProjectAdmin,
                    imageSrc: 'statistics.png',
                },
                {
                    icon: mdiCog,
                    text: this.$t('settings.title'),
                    link: 'settings',
                    isVisible: this.isProjectAdmin,
                    imageSrc: 'setting.png',
                }
            ]
            return items.filter((item) => item.isVisible)
        }
    },

    methods: {
        toLabeling() {
            const query = this.$services.option.findOption(this.$route.params.id)
            const link = getLinkToAnnotationPage(this.$route.params.id, this.project.projectType)
            this.$router.push({
                path: this.localePath(link),
                query
            })
        }
    }
}
</script>
export default {
  locales: [
    {
      name: '中文',
      code: 'zh',
      iso: 'zh-CN',
      file: 'zh'
    },
    {
      name: 'English',
      code: 'en',
      iso: 'en-CA',
      file: 'en'
    },

    // {
    //   name: 'Français',
    //   code: 'fr',
    //   iso: 'fr-CA',
    //   file: 'fr'
    // },
    // {
    //   name: 'Deutsch',
    //   code: 'de',
    //   iso: 'de-DE',
    //   file: 'de'
    // }
  ],
  lazy: true,
  langDir: 'i18n/',
  defaultLocale: 'zh',
  vueI18n: {
    fallbackLocale: 'zh'
  },
  detectBrowserLanguage: {
    useCookie: true,
    cookieKey: 'i18n_redirected',
    onlyOnRoot: true // for SEO purposes
  }
}

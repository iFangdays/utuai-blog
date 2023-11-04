import * as $ from "jquery"
// 使用异步函数也是可以的
export default ({
                  Vue, // VuePress 正在使用的 Vue 构造函数
                  options, // 附加到根实例的一些选项
                  router, // 当前应用的路由实例
                  siteData, // 站点元数据
                  isServer, // 当前应用配置是处于 服务端渲染 或 客户端
                }) => {
  // ...做一些其他的应用级别的优化
  // 全局注入组件
  // Vue.use(ElementUI);
  Vue.mixin({
    // 混合注入,加载全局文件
    mounted() {
      if(!this.$frontmatter.article){
        const container = $('.theme-vdoing-wrapper>  .articleInfo-wrap + .content-wrapper .theme-vdoing-content');
        if (!container) return;
        container.attr('id', 'container');
        window.btw = new BTWPlugin();
        window.btw.init({
          id: 'container',
          blogId: '32233-1699069242965-284',
          name: '乌图AI',
          qrcode: 'https://www.utuai.com/assets/img/weixin.jpg',
          keyword: 'uTuAIBlog',
        });
      }
    }
  });
};

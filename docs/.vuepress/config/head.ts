import { HeadTags } from "vuepress/config";

export default <HeadTags>[
  // ["link", { rel: "icon", href: "/favicon.ico" }], //站点图标, 默认为 public/favicon.ico
  [
    "meta",
    {
      name: "viewport",
      content: "width=device-width,initial-scale=1,user-scalable=no",
    },
  ],
  [
    "meta",
    {
      name: "baidu-site-verification",
      content: "codeva-HJDBxwx3Fd",
    },
  ],
  // [
  //   'script',
  //   { charset: 'utf-8', src: 'https://my.openwrite.cn/js/readmore.js' },
  // ],
  [
    "meta",
    { name: 'keywords', content: '网站搭建,小程序开发,抖音小程序开发,域名评估,chatGPT,AI'}
  ]
];

// nav 配置, 即上方导航栏

import { NavItem } from "vuepress/config";

export default <Array<NavItem>>[
  { text: "Home", link: "/" },
  {
    text: "配置",
    items: [
      {
        text: "项目配置",
        link: "/start/",
      },
      {
        text: "部署",
        link: "/deploy/",
      },
      {
        text: "更多参考",
        link: "/more/",
      },
    ],
  },
  {
    text: "友情链接",
    items: [
      {
        text: "题海后台",
        link: "http://admin.utuai.com",
      },
      {
        text: "ChatGPT",
        link: "http://chat.utuai.com",
      }
    ],
  },
];

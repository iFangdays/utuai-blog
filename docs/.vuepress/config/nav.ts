// nav 配置, 即上方导航栏

import { NavItem } from "vuepress/config";

export default <Array<NavItem>>[
  { text: "首页", link: "/" },
  {
    text: "前端笔记",
    items:[
      {
        text:'《vue》系列',
        link:'/vue/'
      },
      {
        text:'《react》系列',
        link:'/react/'
      },
      {
        text:'《uniApp》系列',
        link:'/uniApp/'
      },
      {
        text:'《微信小程序》系列',
        link:'/wxApp/'
      }
    ]
  },
  {
    text:'面试题',
    link:'/interview/'
  },
  {
    text:'技术总结',
    link:'/summary/',
    items: [
      {
        text:'常用代码块',
        link:'/pages/b54661/'
      },
      {
        text:'命令合集',
        link:'/pages/ad247c4332211551/'
      },
      {
        text:'技巧分享',
        link:'/pages/3ac198/'
      }
    ]
  },
  {
    text:'速查索引',
    items:[
      {
        text:'分类',
        link:'/categories/'
      },
      {
        text:'标签',
        link:'/tags/'
      },
      {
        text:'归档',
        link:'/archives/'
      }
    ]
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

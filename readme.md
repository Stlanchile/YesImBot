<div align="center">
    <img src="https://raw.githubusercontent.com/HydroGest/YesImBot/main/img/logo.png" width="90%" />
	<h1>Athena | YesImBot</h1>

<h6>感谢 <a href="https://github.com/MizuAsaka">@MizuAsaka</a> 提供 <a href="https://github.com/HydroGest/YesImBot/issues/6">Logo</a></h6>

[![npm](https://img.shields.io/npm/v/koishi-plugin-yesimbot?style=flat-square)](https://www.npmjs.com/package/koishi-plugin-yesimbot) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/) ![Language](https://img.shields.io/badge/language-TypeScript-brightgreen) ![NPM Downloads](https://img.shields.io/npm/dw/koishi-plugin-yesimbot) ![Static Badge](https://img.shields.io/badge/QQ交流群-857518324-green)


*✨机器壳，人类心。✨*

</div>

## 🎐 简介

YesImBot / Athena 是一个 [Koishi](https://koishi.chat/zh-CN/) 插件，旨在让人工智能大模型也能参与到群聊的讨论中。

*新的文档站已上线：[https://athena.mkc.icu/](https://athena.mkc.icu/)*

## 🎹 特性

- 轻松自定义：Bot 的名字、性格、情感，以及其他额外的消息都可以在插件配置中轻易修改。

- 负载均衡：你可以配置多个大模型的 API 接口， Athena 会均衡地调用每一个 API。

- 沉浸感知：大模型感知当前的背景信息，如日期时间、群聊名字，At 消息等。

- 防提示注入：Athena 将会屏蔽可能对大模型进行注入的消息，防止机器人被他人破坏。

- Prompt 自动获取：无需自行配制，多种优质 Prompt 开箱即用。

- *AND MORE...*

## 🌈 开始使用

> [!IMPORTANT]
> 继续前, 请确保正在使用 Athena 的最新版本。

> [!CAUTION]
> 请仔细阅读此部分, 这很重要。

下面来讲解配置文件的用法:

```yaml
# 会话设置
MemorySlot:
    # 记忆槽位，每一个记忆槽位都可以填入一个或多个会话id（群号或private:私聊账号），在一个槽位中的会话id会共享上下文
    SlotContains:
        - 114514 # 收到来自114514的消息时，优先使用这个槽位，意味着bot在此群中无其他会话的记忆
        - 114514, private:1919810 # 收到来自1919810的私聊消息时，优先使用这个槽位，意味着bot此时拥有两个会话的记忆
        - private:1919810, 12085141, 2551991321520
    # 规定机器人能阅读的上下文数量
    SlotSize: 100
    # 机器人在每个会话开始发言所需的消息数量，即首次触发条数
    FirstTriggerCount: 2
    # 以下是每次机器人发送消息后的冷却条数由LLM确定或取随机数的区间
    # 最大冷却条数
    MaxTriggerCount: 4
    # 最小冷却条数
    MinTriggerCount: 2
    # 距离会话最后一条消息达到此时间时，将主动触发一次Bot回复，设为 0 表示关闭此功能
    MaxTriggerTime: 0
    # 单次触发冷却（毫秒），冷却期间如又触发回复，将处理新触发回复，跳过本次触发
    MinTriggerTime: 1000
    # 每次收到 @ 消息，机器人马上开始做出回复的概率。 取值范围：[0, 1]
    AtReactPossibility: 0.50
    # 过滤的消息。这些包含这些关键词的消息将不会加入到上下文。
    # 这主要是为了防止 Bot 遭受提示词注入攻击。
    Filter:
        - You are
        - 呢
        - 大家

# LLM API 相关设置
API:
    # 这是个列表，可以配置多个 API，实现负载均衡。
    APIList:
        # API 返回格式类型，可选 OpenAI / Cloudflare / Ollama / Custom
        - APIType: OpenAI
          # API 基础 URL，此处以 OpenAI 为例
          # 若你是 Cloudflare，请填入 https://api.cloudflare.com/client/v4
          BaseURL: https://api.openai.com/
          # 你的 API 令牌
          APIKey: sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX
          # 模型
          AIModel: gpt-4o-mini
          # 若你正在使用 Cloudflare，不要忘记下面这个配置
          # Cloudflare Account ID，若不清楚可以看看你 Cloudflare 控制台的 URL
          UID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 机器人设定
Bot:
    # 名字
    BotName: 胡梨
    # 原神模式（什
    CuteMode: true
    # Prompt 文件的下载链接或文件名。如果下载失败，请手动下载文件并放入 koishi.yml 所在目录
    # 非常重要! 如果你不理解这是什么，请不要修改
    PromptFileUrl:
        - "https://raw.githubusercontent.com/HydroGest/promptHosting/main/src/prompt.mdt" # 一代 Prompt，所有 AI 模型适用
        - "https://raw.githubusercontent.com/HydroGest/promptHosting/main/src/prompt-next.mdt" # 下一代 Prompt，效果最佳，如果你是富哥，用的起 Claude 3.5 / GPT-4 等，则推荐使用
        - "https://raw.githubusercontent.com/HydroGest/promptHosting/main/src/prompt-next-short.mdt" # 下一代 Prompt 的删减版，适合 GPT-4o-mini 等低配模型使用
    # 当前选择的 Prompt 索引，从 0 开始
    PromptFileSelected: 2
    # Bot 的自我认知
    WhoAmI: 一个普通群友
    # Bot 的性格
    BotPersonality: 冷漠/高傲/网络女神
    # 屏蔽其他指令（实验性）
    SendDirectly: true
    # 机器人的习惯，当然你也可以放点别的小叮咛
    BotHabbits: 辩论
    # 机器人的背景
    BotBackground: 校辩论队选手
    ... # 其他应用于prompt的角色设定。如果这些配置项没有被写入prompt文件，那么这些配置项将不会体现作用
    # 机器人消息后处理，用于在机器人发送消息前的最后一个关头替换消息中的内容，支持正则表达式
    BotSentencePostProcess:
        - replacethis: 。$
          tothis: ''
        - replacethis: 哈哈哈哈
          tothis: 嘎哈哈哈
    # 机器人的打字速度
    WordsPerSecond: 30 # 30 字每秒
... # 其他配置项参见文档站
```

然后，将机器人拉到对应的群组中。机器人首先会潜水一段时间，这取决于 `SlotContains.FirstTriggerCount` 的配置。当新消息条数达到这个值之后，Bot 就要开始参与讨论了（这也非常还原真实人类的情况，不是吗）。

> [!TIP]
> 如果你认为 Bot 太活跃了，你也可以将 `SlotContains.MinTriggerCount` 数值调高。

<!-- 现在好像不遵循这个关系也没事
> [!WARNING]
> 频次配置应保持如下关系: `Group.MinPopNum` < `Group.MaxPopNum` < `Group.SendQueueSize`，否则会导致问题。
-->

接下来你可以根据实际情况调整机器人设定中的选项。在这方面你大可以自由发挥。但是如果你用的是 Cloudflare Workers AI，你可以会发现你的机器人在胡言乱语。这是 Cloudflare Workers AI 的免费模型效果不够好，中文语料较差导致的。如果你想要在保证 AI 发言质量的情况下尽量选择价格较为经济的 AI 模型，那么 ChatGPT-4o-mini 或许是明智之选。当然，你也不一定必须使用 OpenAI 的官方 API，Athena 支持任何使用 OpenAI 官方格式的 API 接口。

> [!NOTE]
> 经过测试, Claude 3.5 模型在此场景下表现最佳。

## 📃 自定义系统提示词
将prompt.mdt文件下载到本地后，如果你觉得我们写得不好，或者是有自己新奇的想法，你可能会想要自定义这部分内容。接下来我们就来教你如何这么做。

首先，你需要在插件的配置中关闭 `每次启动时尝试更新 Prompt 文件` 这个选项，它在配置页面最下面的调试工具配置项中。之后，你可以在koishi的资源管理器中找到prompt.mdt这个文件。你可以在koishi自带的编辑器中自由地修改这个文件，不过下面有几点你需要注意：

- 某些字段会被替换，下面是所有会被替换的字段：
```
${BotName} -> 机器人的名字
${BotSelfId} -> 机器人的账号
${config.Bot.WhoAmI} -> 机器人的自我认知
${config.Bot.BotHometown} -> 机器人的家乡
${config.Bot.BotYearold} -> 机器人的年龄
${config.Bot.BotPersonality} -> 机器人的性格
${config.Bot.BotGender} -> 机器人的性别
${config.Bot.BotHabbits} -> 机器人的习惯
${config.Bot.BotBackground} -> 机器人的背景
${config.Bot.CuteMode} -> 开启|关闭

${curDate} -> 当前时间 # 2024年12月3日星期二17:34:00
${curGroupId} -> 当前所在群的群号
${outputSchema} -> LLM期望的输出格式模板
${coreMemory} -> 要附加给LLM的记忆
```
- 当前，消息队列呈现给 LLM 的格式是这样的：
```
[messageId][{date} from_guild:{channelId}] {senderName}<{senderId}> 说: {userContent}
```
- 当前，Athena 希望 LLM 返回的格式是这样的：
```json
{
      "status": "success",    // "success" 或 "skip" (跳过回复) 或 "function" (运行工具)
      "replyTo": "123456789", // 要把finReply发送到的会话id
      "nextReplyIn": 2,       // 下次回复的冷却条数，让LLM参与控制发言频率
      "logic": "", // LLM思考过程
      "reply": "", // 初版回复
      "check": "", // 检查初版回复是否符合 "消息生成条例" 过程中的检查逻辑。
      "finReply": "", // 最终版回复
      "functions": [] // 要运行的指令列表
}
```

> [!NOTE]
> 自己修改prompt时，请确保 LLM 的回复符合要求的JSON格式。~~但缺少某些条目好像也没关系？Σ(っ °Д °;)っ~~

## 🚧 从 v1 版本迁移

> [!NOTE]
> 由于图片查看器配置项发生了变动，会导致控制台中此项配置无法正常展开。如果你是从v1.7.x版本升级到v2，请自行将 `koishi.yml` 中本插件的 `ImageViewer` 部分删除

## 🌼 推荐的 API 提供商

我们强烈推荐大家使用非 Token 计费的 API，这是因为 Athena 每次对话的前置 Prompt 本身消耗了非常多的 Token。你可以使用一些以调用次数计费的 API，比如：

- [GPTGOD](https://gptgod.online/#/register?invite_code=envrd6lsla9nydtipzrbvid2r)

## ✨ 效果
<details>
  <summary>截图</summary>

  ![](https://raw.githubusercontent.com/HydroGest/YesImBot/main/img/screenshot-1.png)
  ![](https://raw.githubusercontent.com/HydroGest/YesImBot/main/img/screenshot-2.png)
</details>

## 🍧 TODO

我们的终极目标是——即使哪一天你的账号接入了 Athena，群友也不能发现任何端倪——我们一切的改进都是朝这方面努力的。

- [x] At 消息识别
- [x] 表情发送
- [x] 图片多模态与基于图像识别的伪多模态
- [ ] 转发消息拾取
- [ ] TTS/STT 文字转语音
- [ ] RAG 记忆库
- [ ] 读取文件
- [x] 工具调用

## 🚩 Build

请务必按照此顺序依次构建每个模块，确保可以正确处理依赖关系。

多次构建同一个模块会产生报错 `error TS5055: Cannot write file 'xxx' because it would overwrite input file.`，请先运行 `yarn clean` 后再次构建。

```bash
# Install dependencies
yarn install

# Build
yarn build core # or yarn build:core
yarn build memory
yarn build webui
```

## 💫 贡献者

感谢贡献者们, 是你们让 Athena 成为可能。

![contributors](https://contrib.rocks/image?repo=HydroGest/YesImBot)

## 💡 反馈

欢迎发布 issue，或是直接加入 Athena 官方交流 & 测试群：[857518324](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=k3O5_1kNFJMERGxBOj1ci43jHvLvfru9&authKey=TkOxmhIa6kEQxULtJ0oMVU9FxoY2XNiA%2B7bQ4K%2FNx5%2F8C8ToakYZeDnQjL%2B31Rx%2B&noverify=0&group_code=857518324)，我们随时欢迎你的来访！

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hydrogest/Yesimbot&type=Date)](https://star-history.com/#Hydrogest/Yesimbot&Date)

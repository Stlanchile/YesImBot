import { XMLParser } from "fast-xml-parser";
import { jsonrepair } from 'jsonrepair';
import { Context, Random, Session } from "koishi";

import { AdapterSwitcher } from "./adapters";
import { AssistantMessage, ImageComponent, Message, SystemMessage, TextComponent, ToolCall, ToolMessage, UserMessage } from "./adapters/creators/component";
import { getFunctionSchema, ToolSchema } from "./adapters/creators/schema";
import { Config } from "./config";
import { Extension, getExtensions, getFunctionPrompt, getToolSchema } from "./extensions/base";
import { EmojiManager } from "./managers/emojiManager";
import { LLMResponse, Tool } from "./models/LLMResponse";
import { ImageViewer } from "./services/imageViewer";
import { toolsToString } from "./utils";
import { isEmpty, isNotEmpty, Template } from "./utils/string";
import { ResponseVerifier } from "./utils/verifier";

interface Dependencies {
    readonly ctx: Context;
    readonly config: Config;
    readonly imageViewer: ImageViewer;
    readonly emojiManager?: EmojiManager;
    readonly verifier?: ResponseVerifier;
}

export class Bot {
    private contextSize: number;    // 以对话形式给出的上下文长度

    private minTriggerCount: number;
    private maxTriggerCount: number;
    private allowErrorFormat: boolean;
    readonly finalFormat: "JSON" | "XML";

    private context: Message[] = []; // 对话上下文
    private recall: Message[] = [];  //
    private prompt: string;          // 系统提示词
    private template: Template;

    private sendResolveOK: boolean;
    private sendAssistantMessageAs: "USER" | "ASSISTANT";
    private addRoleTagBeforeContent: boolean;

    private intelligentRouterConfig: Config['IntelligentRouter'];

    private extensions: { [key: string]: Extension & Function } = {};
    private toolsSchema: ToolSchema[] = [];

    private emojiManager: EmojiManager;
    readonly verifier: ResponseVerifier;
    readonly imageViewer: ImageViewer;

    private adapterSwitcher: AdapterSwitcher;
    public session: Session;
    private ctx: Context;

    constructor(private deps: Dependencies) {
        const { ctx, config } = this.deps;
        this.ctx = ctx;
        this.sendResolveOK = config.Settings.SendResolveOK;
        this.sendAssistantMessageAs = config.Settings.SendAssistantMessageAs;
        this.addRoleTagBeforeContent = config.Settings.AddRoleTagBeforeContent
        this.contextSize = config.MemorySlot.SlotSize;
        this.minTriggerCount = Math.min(config.MemorySlot.MinTriggerCount, config.MemorySlot.MaxTriggerCount);
        this.maxTriggerCount = Math.max(config.MemorySlot.MinTriggerCount, config.MemorySlot.MaxTriggerCount);
        this.allowErrorFormat = config.Settings.AllowErrorFormat;
        this.intelligentRouterConfig = config.IntelligentRouter;
        this.adapterSwitcher = new AdapterSwitcher(
            config.API.APIList,
            config.Parameters
        );
        this.template = new Template(config.Settings.SingleMessageStrctureTemplate, /\{\{(\w+(?:\.\w+)*)\}\}/g, /\{\{(\w+(?:\.\w+)*),([^,]*),([^}]*)\}\}/g);
        this.emojiManager = this.deps.emojiManager;
        this.verifier = this.deps.verifier;
        this.imageViewer = this.deps.imageViewer;

        for (const extension of getExtensions(ctx, this)) {
            this.extensions[extension.name] = extension as any;
            this.toolsSchema.push(getToolSchema(extension));
        }
        this.finalFormat = this.adapterSwitcher.getAdapter().adapter.ability.includes("结构化输出") ? "JSON" : config.Settings.LLMResponseFormat;
    }

    setSystemPrompt(content: string) {
        this.prompt = content;
    }

    setSession(session: Session) {
        this.session = session;
    }

    addContext(message: Message) {
        while (this.context.length >= this.contextSize) {
            this.recall.push(this.context.shift());
        }
        this.context.push(message);
    }

    setChatHistory(chatHistory: Message[]) {
      this.context = [];
      if (this.deps.config.Settings.MultiTurn) {
          for (const message of chatHistory) {
              this.addContext(message);
          }
      } else {
          let components: (TextComponent | ImageComponent)[] = [];
          chatHistory.forEach(message => {
              if (typeof message.content === 'string') {
                  components.push(TextComponent(message.content));
              } else if (Array.isArray(message.content)) {
                  const validComponents = message.content.filter((comp): comp is TextComponent | ImageComponent =>
                      comp.type === 'text' || (comp.type === 'image_url' && 'image_url' in comp));
                  components.push(...validComponents);
              }
          });
          // 合并components中相邻的 TextComponent
          components = components.reduce((acc, curr, i) => {
              if (i === 0) return [curr];
              const prev = acc[acc.length - 1];
              if (prev.type === 'text' && curr.type === 'text') {
                  prev.text += '\n' + (curr as TextComponent).text;
                  return acc;
              }
              return [...acc, curr];
          }, []);
          this.addContext(UserMessage(...components));
      }
    }

    getAdapter() {
        return this.adapterSwitcher.getAdapter();
    }

    async generateResponse(messages: Message[], debug = false): Promise<LLMResponse> {
        const lastUserMessageContent = messages
            .filter(m => m.role === 'user')
            .map(m => typeof m.content === 'string' ? m.content : (m.content as any[]).filter(c => c.type === 'text').map(c => c.text).join(''))
            .pop() || '';

        const { Enabled, LatencyOptimization, StandardModel, EnhancedModel } = this.intelligentRouterConfig;

        if (!Enabled) {
            return this._executeStandardRequest(messages, debug);
        }

        if (LatencyOptimization.Enabled) {
            const heuristicDecision = this._applyHeuristics(lastUserMessageContent);
            if (heuristicDecision === 'ENHANCED') {
                this.ctx.logger.info(`路由决策：启发式规则命中 -> ENHANCED`);
                return this._executeEnhancedRequest(messages, debug);
            }
            if (heuristicDecision === 'STANDARD') {
                this.ctx.logger.info(`路由决策：启发式规则命中 -> STANDARD`);
                return this._executeStandardRequest(messages, debug);
            }
        }

        const routerPromise = this._determineModel(lastUserMessageContent);
        const standardPayloadPromise = this._prepareRequestPayload(StandardModel.AdapterId, false, messages);

        const routerResult: any = await Promise.race([
            routerPromise,
            new Promise(resolve => setTimeout(() => resolve({ timedOut: true }), LatencyOptimization.RouterTimeout))
        ]);

        if (!routerResult.timedOut && routerResult.isEnhanced) {
            this.ctx.logger.info(`路由决策：并行处理后选择 -> ENHANCED`);
            return this._executeEnhancedRequest(messages, debug);
        }
        
        this.ctx.logger.info(`路由决策：并行处理后选择 -> STANDARD (或超时回退)`);
        const standardPayload = await standardPayloadPromise;
        return this._executeFinalRequest(StandardModel.AdapterId, standardPayload, debug);
    }

    private _applyHeuristics(message: string): 'STANDARD' | 'ENHANCED' | null {
        const { ShortMessageThreshold, CodeBlockTriggersEnhanced } = this.intelligentRouterConfig.LatencyOptimization.Heuristics;
        
        if (CodeBlockTriggersEnhanced && message.includes('```')) {
            return 'ENHANCED';
        }
        if (ShortMessageThreshold > 0 && message.split(' ').length <= ShortMessageThreshold) {
            return 'STANDARD';
        }
        return null;
    }

    private async _determineModel(lastUserMessage: string): Promise<{ adapterId: string; isEnhanced: boolean }> {
        const { RouterModel, StandardModel, EnhancedModel } = this.intelligentRouterConfig;
    
        try {
            const { adapter: routerAdapter } = this.adapterSwitcher.getAdapterById(RouterModel.AdapterId);
            if (!routerAdapter) {
                throw new Error(`Router adapter with ID '${RouterModel.AdapterId}' not found.`);
            }
    
            const routerPrompt = new Template(RouterModel.RouterPrompt).render({ lastUserMessage });
    
            const response = await routerAdapter.chat([SystemMessage(routerPrompt)], [], true);
            const decision = response.message.content.trim().toUpperCase();
    
            this.ctx.logger.info(`路由模型决策: ${decision}`);
    
            if (decision === 'ENHANCED') {
                return { adapterId: EnhancedModel.AdapterId, isEnhanced: true };
            }
        } catch (error) {
            this.ctx.logger.error(`路由模型调用失败: ${error.message}。将回退到标准模型。`);
        }
    
        return { adapterId: StandardModel.AdapterId, isEnhanced: false };
    }

    private async _prepareRequestPayload(adapterId: string, isEnhanced: boolean, messages: Message[]): Promise<{ systemPrompt: string, context: Message[], adapterIndex: number }> {
        const { adapter, current } = this.adapterSwitcher.getAdapterById(adapterId);
        if (!adapter) throw new Error(`Adapter with ID ${adapterId} not found`);

        const lastUserMessageContent = messages
            .filter(m => m.role === 'user')
            .map(m => typeof m.content === 'string' ? m.content : '')
            .pop() || '';
        
        const retrievedMemories = await this.ctx.memory.retrieve(this.session.userId, lastUserMessageContent);

        let longTermMemoryPrompt = '';
        if (retrievedMemories.length > 0) {
            longTermMemoryPrompt = `[相关长期记忆]\n${retrievedMemories.join('\n')}\n\n`;
        }

        let baseSystemPrompt = longTermMemoryPrompt + this.prompt;
        let finalSystemPrompt = baseSystemPrompt;

        if (isEnhanced) {
            finalSystemPrompt += `\n\n${this.intelligentRouterConfig.EnhancedModel.EnhancedPrompt}`;
        }

        const tempContext = [...this.context];
        for (const message of messages) {
            while (tempContext.length >= this.contextSize) {
                tempContext.shift();
            }
            tempContext.push(message);
        }

        return { systemPrompt: finalSystemPrompt, context: tempContext, adapterIndex: current };
    }

    private async _executeFinalRequest(adapterId: string, payload: { systemPrompt: string, context: Message[], adapterIndex: number }, debug: boolean): Promise<LLMResponse> {
        const { adapter } = this.adapterSwitcher.getAdapterById(adapterId);
        if (!adapter) throw new Error(`Adapter ${adapterId} not found`);

        for (const message of payload.context.slice(this.context.length)) {
            this.addContext(message);
        }

        if (!adapter.ability.includes("原生工具调用")) {
            let str = Object.values(this.extensions)
                .map((extension) => getFunctionPrompt(extension))
                .join("\n");
            payload.systemPrompt = payload.systemPrompt.replace("{{functionPrompt}}", getFunctionSchema(this.finalFormat) + `${isEmpty(str) ? "No functions available." : str}`);
        }

        const response = await adapter.chat([SystemMessage(payload.systemPrompt), ...(this.sendResolveOK ? [AssistantMessage("Resolve OK")] : []), ...payload.context], adapter.ability.includes("原生工具调用") ? this.toolsSchema : undefined, debug);
        let content = response.message.content;

        if (adapter.ability.includes("深度思考")) {
            const contentWithoutReasoning = content.replace(
                new RegExp(`${adapter.reasoningStart}[\\s\\S]*?${adapter.reasoningEnd}`, 'g'),
                ''
            );
            content = contentWithoutReasoning.trim();
        }
        if (debug) this.ctx.logger.info(`Adapter: ${adapterId}, Response: \n${content}`);

        if (adapter.ability.includes("原生工具调用")) {
            const toolResponse = await this.handleToolCalls(response.message.tool_calls || [], debug);
            if (toolResponse) return toolResponse;
        }

        let LLMResponse: any = {};
        const regex = new RegExp(`\\\`\\\`\\\`(json|xml)\\s*\\n([\\s\\S]*?)\\n\\\`\\\`\\\`|({[\\s\\S]*?}|<[\\s\\S]*?>[\\s\\S]*<\\/[\\s\\S]*?>)`,'gis');
        let contentToParse = null;
        let match;

        while ((match = regex.exec(content)) !== null) {
            const lang = match[1];
            const codeContent = match[2];
            const directContent = match[3];

            if (lang && lang.toUpperCase() === this.finalFormat) {
                contentToParse = codeContent;
                break;
            }

            if (directContent) {
                if (
                    (this.finalFormat === 'JSON' && directContent.trim().startsWith('{')) ||
                    (this.finalFormat === 'XML' && directContent.trim().startsWith('<'))
                ) {
                    contentToParse = directContent;
                    break;
                }
            }
        }

        if (contentToParse) {
            try {
                if (this.finalFormat === "JSON") {
                    LLMResponse = JSON.parse(jsonrepair(contentToParse));
                } else if (this.finalFormat === "XML") {
                    const parser = new XMLParser({ ignoreAttributes: false, processEntities: false, stopNodes: ['*.logic', '*.reply', '*.check', '*.finalReply'] });
                    LLMResponse = parser.parse(contentToParse);
                }
                this.addContext(AssistantMessage(JSON.stringify(LLMResponse)));
            } catch (e) {
                return { status: "fail", raw: content, usage: response.usage, reason: `${this.finalFormat} 解析失败。请上报此消息给开发者: ${e.message}`, adapterIndex: payload.adapterIndex };
            }
        } else {
            try {
                if (this.finalFormat === "JSON") {
                    LLMResponse = JSON.parse(jsonrepair(content));
                } else {
                    const parser = new XMLParser({ ignoreAttributes: false, processEntities: false, stopNodes: ['*.logic', '*.reply', '*.check', '*.finalReply'] });
                    LLMResponse = parser.parse(content);
                }
                this.addContext(AssistantMessage(JSON.stringify(LLMResponse)));
            } catch (err) {
                return { status: "fail", raw: content, usage: response.usage, reason: `没有找到有效的 ${this.finalFormat} 结构: ${content}`, adapterIndex: payload.adapterIndex };
            }
        }

        let nextTriggerCount: number = Random.int(this.minTriggerCount, this.maxTriggerCount + 1);
        const nextTriggerCountbyLLM = Math.max(this.minTriggerCount, Math.min(Number(LLMResponse.nextReplyIn) ?? this.minTriggerCount, this.maxTriggerCount));
        nextTriggerCount = Number(nextTriggerCountbyLLM) || nextTriggerCount;
        const finalLogic = LLMResponse.logic || "";

        let functions: Tool[] = [];
        if (Array.isArray(LLMResponse.functions)) {
            functions = LLMResponse.functions;
        } else if (isNotEmpty(LLMResponse.functions?.name)) {
            functions = [LLMResponse.functions];
        } else if (Array.isArray(LLMResponse.functions?.function)) {
            functions = LLMResponse.functions.function;
        } else if (isNotEmpty(LLMResponse.functions?.function?.name)) {
            functions = [LLMResponse.functions.function];
        }

        if (LLMResponse.status === "success") {
            let finalResponse: string = (LLMResponse.finalReply || LLMResponse.reply || "").toString();
            if (this.allowErrorFormat) {
                finalResponse += LLMResponse.msg || LLMResponse.text || LLMResponse.message || LLMResponse.answer || "";
            }

            if (isEmpty(finalResponse)) {
                return { status: "skip", raw: content, usage: response.usage, nextTriggerCount, functions, logic: finalLogic, adapterIndex: payload.adapterIndex };
            }

            const replyTo = this.extractReplyTo(LLMResponse.replyTo);
            finalResponse = await this.unparseFaceMessage(finalResponse);

            return { status: "success", raw: content, finalReply: finalResponse, replyTo, nextTriggerCount, logic: finalLogic, functions, usage: response.usage, adapterIndex: payload.adapterIndex };
        } else if (LLMResponse.status === "skip") {
            return { status: "skip", raw: content, nextTriggerCount, logic: finalLogic, usage: response.usage, functions, adapterIndex: payload.adapterIndex };
        } else if (LLMResponse.status === "interaction") {
            return this.handleFunctionCalls(functions, debug);
        } else {
            return { status: "fail", raw: content, usage: response.usage, reason: `status 不是一个有效值: ${LLMResponse.status}`, adapterIndex: payload.adapterIndex };
        }
    }

    private async _executeStandardRequest(messages: Message[], debug: boolean): Promise<LLMResponse> {
        const payload = await this._prepareRequestPayload(this.intelligentRouterConfig.StandardModel.AdapterId, false, messages);
        return this._executeFinalRequest(this.intelligentRouterConfig.StandardModel.AdapterId, payload, debug);
    }

    private async _executeEnhancedRequest(messages: Message[], debug: boolean): Promise<LLMResponse> {
        const payload = await this._prepareRequestPayload(this.intelligentRouterConfig.EnhancedModel.AdapterId, true, messages);
        return this._executeFinalRequest(this.intelligentRouterConfig.EnhancedModel.AdapterId, payload, debug);
    }

    // 或许可以将这两个函数整合到一起
    // 递归调用
    // TODO: 指定最大调用深度
    // TODO: 上报函数调用信息
    private async handleToolCalls(toolCalls: ToolCall[], debug: boolean): Promise<LLMResponse | null> {
        if (debug) {
            this.ctx.logger.info(`Bot[${this.session.selfId}] 想要调用工具`)
            this.ctx.logger.info(toolCalls.map(toolCall => `Name: ${toolCall.function.name}\nArgs: ${JSON.stringify(toolCall.function.arguments)})}`).join('\n'));
        }
        let returns: ToolMessage[] = [];
        for (let toolCall of toolCalls) {
            try {
                let result = await this.callFunction(toolCall.function.name, toolCall.function.arguments);
                if (!isEmpty(result)) returns.push(ToolMessage(result, toolCall.id));
            } catch (e) {
                returns.push(ToolMessage(e.message, toolCall.id));
            }
        }
        if (returns.length > 0) {
            return this.generateResponse(returns, debug);
        }
        return null;
    }

    private async handleFunctionCalls(functions: Tool[], debug: boolean): Promise<LLMResponse | null> {
        if (debug) {
            this.ctx.logger.info(`Bot[${this.session.selfId}] 想要调用工具`)
            this.ctx.logger.info(toolsToString(functions));
        }
        let returns: Message[] = [];
        for (const func of functions) {
            const { name, params } = func;
            try {
                if (this.sendAssistantMessageAs === "USER") {
                  returns.push(UserMessage(this.addRoleTagBeforeContent ? "[assistant] " : ""  + `CALLING FUNCTION: ${name} PARAMS: ${JSON.stringify(params)}`));
                  let returnValue = await this.callFunction(name, params);
                  if (!isEmpty(returnValue)) returns.push(UserMessage(this.addRoleTagBeforeContent ? "[tool] " : ""  + `FUNCTION RESULT: ${returnValue}`));
                } else {
                  returns.push(AssistantMessage(this.addRoleTagBeforeContent ? "[assistant] " : ""  + `CALLING FUNCTION: ${name} PARAMS: ${JSON.stringify(params)}`));
                  let returnValue = await this.callFunction(name, params);
                  if (!isEmpty(returnValue)) returns.push(AssistantMessage(this.addRoleTagBeforeContent ? "[tool] " : ""  + `FUNCTION RESULT: ${returnValue}`));
                }
            } catch (e) {
                if (this.sendAssistantMessageAs === "USER") {
                  returns.push(UserMessage(this.addRoleTagBeforeContent ? "[tool] " : ""  + `FUNCTION ERROR: ${e.message}`));
                }
                else {
                  returns.push(AssistantMessage(this.addRoleTagBeforeContent ? "[tool] " : ""  + `FUNCTION ERROR: ${e.message}`));
                }
            }
        }
        if (returns.length > 0) {
            return this.generateResponse(returns, debug);
        }
        return null;
    }

    // 如果 replyTo 不是私聊会话，只保留数字部分
    private extractReplyTo(replyTo: string): string {
        try {
            replyTo = replyTo.toString().trim();
            if (isNotEmpty(replyTo) && !replyTo.startsWith("private:")) {
                const numericMatch = replyTo.match(/\d+/);
                if (numericMatch) {
                    replyTo = numericMatch[0].replace(/\s/g, "");
                }
                // 不合法的 channelId
                if (replyTo.match(/\{.+\}/)) {
                    replyTo = "";
                }
                if (replyTo.indexOf("sandbox") > -1) {
                    replyTo = "";
                }
            }
            return replyTo;
        } catch (e) {
            return "";
        }
    }

    // TODO: 规范化params
    // OpenAI和Ollama提供的参数不一致
    async callFunction(name: string, params: Record<string, any>): Promise<any> {
        let func = this.extensions[name] as Function;
        if (!func) {
            throw new Error(`Function not found: ${name}`);
        }

        return await func(params);
    }

    getMemory(selfId: string) {
        // @ts-ignore
        if (this.ctx.memory) return this.ctx.memory.MEMORY_PROMPT
        return "";
    }

    async unparseFaceMessage(message: string) {
        message = message.toString();
        // 反转义 <face> 消息
        const faceRegex = /\[表情[:：]\s*([^\]]+)\]/g;
        const matches = Array.from(message.matchAll(faceRegex));

        const replacements = await Promise.all(
            matches.map(async (match) => {
                const name = match[1];
                let id = await this.emojiManager.getIdByName(name);
                if (!id) {
                    id = (await this.emojiManager.getIdByName(await this.emojiManager.getNameByTextSimilarity(name))) || "500";
                }
                return {
                    match: match[0],
                    replacement: `<face id="${id}" name="${(await this.emojiManager.getNameById(id)) || undefined}"></face>`,
                };
            })
        );
        replacements.forEach(({ match, replacement }) => {
            message = message.replace(match, replacement);
        });
        return message;
    }

    private collectUserID() {
        const users: Map<string, string> = new Map();
        const stringTemplate = this.template;

        for (const history of this.context) {
            let content = history.content;
            switch (typeof content) {
                case "string":
                    break;
                case "object":
                    content = (history.content as (TextComponent | ImageComponent)[])
                        .filter((comp): comp is TextComponent => comp.type === 'text')
                        .map(comp => comp.text)
                        .join('');
                    break;
                default:
                    content = "";
                    break;
            }
            const result = stringTemplate.unrender(content);

            if (result.senderId && result.senderName) {
                users.set(result.senderId, result.senderName);
            }
        }

        return users;
    }
}

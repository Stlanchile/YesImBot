import { Config } from "../config";
import logger from "../utils/logger";
import { LLM } from "./config";
import { AssistantMessage, Message } from "./creators/component";
import { ToolSchema } from "./creators/schema";

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface Response {
  model: string;
  created: string;
  message: AssistantMessage;
  usage: Usage;
}

export abstract class BaseAdapter {
  protected url: string;
  protected readonly apiKey: string;
  protected readonly model: string;
  protected readonly otherParams: Record<string, any>;
  readonly ability: ("原生工具调用" | "识图功能" | "结构化输出" | "流式输出" | "深度思考" | "对话前缀续写")[];
  readonly reasoningStart?: string;
  readonly reasoningEnd?: string;
  readonly reasoningEffort?: "low" | "medium" | "high";
  readonly startWith?: string;

  protected history: Message[] = [];

  constructor(
    public adapterConfig: LLM,
    protected parameters?: Config["Parameters"]
  ) {
    const { APIKey, APIType, AIModel, Ability, ReasoningStart, ReasoningEnd, ReasoningEffort, StartWith } = adapterConfig;
    this.apiKey = APIKey;
    this.model = AIModel;
    this.ability = Ability || [];
    this.reasoningStart = ReasoningStart;
    this.reasoningEnd = ReasoningEnd;
    this.reasoningEffort = ReasoningEffort;
    this.startWith = StartWith;

    // 解析其他参数
    this.otherParams = {};
    if (this.parameters?.OtherParameters) {
      Object.entries(this.parameters?.OtherParameters).forEach(([key, value]) => {
        key = key.trim();
        if (typeof value === 'string') {
          value = value.trim();
        }
        // 尝试解析 JSON 字符串
        try {
          // 仅当值为字符串时才进行解析
          if (typeof value === 'string') {
            value = JSON.parse(value);
          }
        } catch (e) {
          // 如果解析失败，保持原值
        }
        // 转换 value 为适当的类型
        if (typeof value === 'boolean') {
          this.otherParams[key] = value;
        } else if (value === 'true') {
          this.otherParams[key] = true;
        } else if (value === 'false') {
          this.otherParams[key] = false;
          // @ts-ignore
        } else if (!isNaN(value)) {
          this.otherParams[key] = Number(value);
        } else {
          this.otherParams[key] = value;
        }
      });
    }

    logger.info(`Adapter: ${APIType} registered`);
  }

  abstract chat(messages: Message[], tools?: ToolSchema[], debug?: Boolean): Promise<Response>;
}

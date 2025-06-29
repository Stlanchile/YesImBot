import fs from "fs";
import { Context } from "koishi";
import { SchemaNode, ToolSchema } from "../adapters/creators/schema";
import { Bot } from "../bot";

export abstract class Extension {
  readonly name: string;
  readonly description: string;
  readonly params: Record<string, SchemaNode>;

  constructor(
    protected readonly ctx: Context,
    protected readonly bot: Bot
  ) {
    // 读取类的静态属性来初始化实例属性
    const funcName = (this.constructor as any)["funcName"];
    const description = (this.constructor as any)["description"];
    const params = (this.constructor as any)["params"];

    // 返回一个函数实例，使得类实例可以调用
    const callable = (keyword: string) => this.apply(keyword);
    Object.defineProperty(callable, "name", { value: funcName });
    Object.defineProperty(callable, "description", { value: description });
    Object.defineProperty(callable, "params", { value: params });
    return callable as any;
  }

  abstract apply(...args: any[]): any;

  get session() {
    return this.bot.session;
  }
}

export function getExtensions(ctx: Context, bot: Bot): Extension[] {
  let extensions: Extension[] = [];

  fs.readdirSync(__dirname)
    .filter((file) => file.startsWith("ext_") && !file.endsWith(".d.ts")) // 不指定 .js 是为了兼容dev模式
    .forEach((file) => {
      try {
        // 应该在 Metadata 中加入模块所需依赖
        // 并通过某种手段安装或加载这些依赖
        // @ts-ignore
        if (!ctx?.memory && file.startsWith("ext_memory")) {
          ctx.logger.warn(`Skip load extension: ${file}`)
          return
        }
        const extension = require(`./${file}`);
        for (const key in extension) {
          extensions.push(new extension[key](ctx, bot));
        }
        ctx.logger.info(`Loaded extension: ${file}`);
      } catch (e) {
        ctx.logger.error(`Failed to load extension: ${file}`);
        ctx.logger.error(e.stack);
      }
    });
  return extensions;
}

/**
 * 生成工具模板
 *
 * https://platform.openai.com/docs/guides/function-calling
 * https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
 * @param ext
 * @returns
 */
export function getToolSchema(ext: Extension): ToolSchema {
  return {
    type: "function",
    function: {
      name: ext.name,
      description: ext.description,
      parameters: {
        type: "object",
        properties: ext.params,
        // 如果有默认值则非必填
        required: Object.entries(ext.params).map(([key, value]) => value.default ? null : key).filter(Boolean),
      },
    },
  };
}

/**
 * 以文本形式给出的工具模板
 * @param ext
 */
export function getFunctionPrompt(ext: Extension): string {
  let lines = [];
  lines.push(`${ext.name}:`);
  lines.push(`  description: ${ext.description}`);
  lines.push(`  params:`);
  Object.entries(ext.params).forEach(([key, value]) => {
    lines.push(`    ${key}: ${value.description}`);
  })
  return lines.join("\n");
}

export function Name(funcName: string) {
  return function (target: Function) {
    target["funcName"] = funcName;
  };
}

export function Description(description: string) {
  return function (target: Function) {
    target["description"] = description;
  };
}


export function Param(param: string, schema: string | SchemaNode) {
  return function (target: Function) {
    if (!target["params"]) {
      target["params"] = {};
    }
    if (typeof schema === "string") {
      schema = SchemaNode.String(schema);
    }
    target["params"][param] = schema;
  };
}

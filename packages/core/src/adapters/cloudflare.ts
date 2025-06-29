import { Config } from "../config";
import { sendRequest } from "../utils/http";
import { BaseAdapter, Response } from "./base";
import { LLM } from "./config";
import { Message } from "./creators/component";
import { ToolSchema } from "./creators/schema";

export class CloudflareAdapter extends BaseAdapter {
  constructor(config: LLM, parameters?: Config["Parameters"]) {
    super(config, parameters);
    const { BaseURL, UID, AIModel } = config;
    this.url = `${BaseURL}/accounts/${UID}/ai/run/${AIModel}`;
  }

  async chat(messages: Message[], toolsSchema?: ToolSchema[], debug = false): Promise<Response> {
    const requestBody = {
      model: this.model,
      messages,
      tools: toolsSchema,
      temperature: this.parameters?.Temperature,
      max_tokens: this.parameters?.MaxTokens,
      frequency_penalty: this.parameters?.FrequencyPenalty,
      presence_penalty: this.parameters?.PresencePenalty,
      ...this.otherParams,
    };
    let response = await sendRequest(this.url, this.apiKey, requestBody, this.adapterConfig.Timeout, debug);
    try {
      return {
        model: this.model,
        created: new Date().toISOString(),
        message: {
          role: response.result.role,
          content: response.result.response,
          tool_calls: response.result.tool_calls,
        },
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      };
    } catch (error) {
      console.error("Error parsing response:", error);
      console.error("Response:", response);
    }
  }
}

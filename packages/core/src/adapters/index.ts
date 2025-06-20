import { Config } from "../config";
import { getAdapter } from "../utils/factory";
import { BaseAdapter } from "./base";
import { CloudflareAdapter } from "./cloudflare";
import { CustomAdapter } from "./custom";
import { GeminiAdapter } from "./gemini";
import { OllamaAdapter } from "./ollama";
import { OpenAIAdapter } from "./openai";

export { BaseAdapter, CloudflareAdapter, CustomAdapter, GeminiAdapter, OllamaAdapter, OpenAIAdapter };

export class AdapterSwitcher {
  private adapters: BaseAdapter[];
  private current = 0;
  constructor(
    adapterConfig: Config["API"]["APIList"],
    parameters: Config["Parameters"]
  ) {
    this.updateConfig(adapterConfig, parameters);
  }

  getAdapter() {
    try {
      if (this.current >= this.adapters.length) this.current = 0;
      return { current: this.current, adapter: this.adapters[this.current++] };
    } catch (error) {
      return;
    }
  }

  getAdapterById(id: string): { current: number, adapter: BaseAdapter } {
    const index = this.adapters.findIndex(a => a.adapterConfig.Id === id);
    if (index !== -1) {
        return { current: index, adapter: this.adapters[index] };
    }
    return { current: -1, adapter: null };
  }

  updateConfig(
    adapterConfig: Config["API"]["APIList"],
    parameters: Config["Parameters"]
  ) {
    this.adapters = [];
    for (const adapter of adapterConfig) {
      if (!adapter.Enabled) continue;
      this.adapters.push(getAdapter(adapter, parameters));
    }
  }
}

export { LLM as LLMConfig } from "./config";

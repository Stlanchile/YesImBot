import { Context } from 'koishi';
import { EmbeddingBase } from './base';
import { EnabledEmbeddingConfig } from './config';
import { sendRequest } from '../utils/http';
import { CacheManager } from '../managers/cacheManager';

export class GeminiEmbedding extends EmbeddingBase {
  private ctx: Context;
  private apiKey: string;
  private endpoint: string;
  private model: string;

  constructor(ctx: Context, config: EnabledEmbeddingConfig, manager?: CacheManager<number[]>) {
    super(config, manager);
    this.ctx = ctx;
    this.apiKey = config.APIKey;
    this.endpoint = `${config.BaseURL}?key=${this.apiKey}`;
    this.model = config.EmbeddingModel;
  }

  async _embed(text: string): Promise<number[]> {
    try {
      const requestBody = {
        model: `models/${this.model}`,
        content: {
          parts: [{ text }],
        },
      };

      // 使用 sendRequest 而不是 post
      const response = await sendRequest<any>(this.endpoint, '', requestBody);

      if (response && response.embedding && response.embedding.values) {
        return response.embedding.values;
      } else {
        throw new Error('Invalid response structure from Gemini API');
      }
    } catch (error) {
      this.ctx.logger.error('Failed to get embedding from Gemini:', error);
      throw error; // 向上抛出，让基类处理
    }
  }
}
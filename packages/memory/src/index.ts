import { Context, Schema, Service, Session } from 'koishi';
import axios from 'axios';
import { z } from 'zod';
import { LanceDBVectorStore } from './vectorStore';
import { getEmbedding, EmbeddingBase } from 'koishi-plugin-yesimbot';
import { Config } from './config';

const DecisionResponseSchema = z.object({
  should_remember: z.boolean(),
  reason: z.string().optional(),
});

declare module 'koishi' {
  interface Context {
    memory: Memory;
  }
}

export class Memory extends Service {
  private vectorStore: LanceDBVectorStore;
  private embedder: EmbeddingBase;
  private config: Config;

  constructor(ctx: Context, config: Config) {
    super(ctx, 'memory');
    this.config = config;
    if (config.embedding.Enabled) {
      this.embedder = getEmbedding(ctx, config.embedding);
      this.vectorStore = new LanceDBVectorStore(ctx, config.vectorStore, this.embedder);
    }
    this.ctx.middleware(this.postDialogueMiddleware.bind(this));
  }

  public async retrieve(userId: string, queryText: string): Promise<string[]> {
    if (!this.vectorStore || !queryText) return [];
    try {
      return await this.vectorStore.search(
        queryText,
        userId,
        this.config.retrieval.topK
      );
    } catch (error) {
      this.ctx.logger.warn('Memory retrieval failed:', error.message);
      return [];
    }
  }

  private async postDialogueMiddleware(session: Session, next: () => Promise<void>) {
    if (!this.vectorStore) return next();

    const originalSend = session.send.bind(session);
    let botResponse = '';

    session.send = (message: string) => {
      if (typeof message === 'string') {
        botResponse = message;
      }
      return originalSend(message);
    };

    await next();

    if (session.content && botResponse) {
      const conversation = `用户: ${session.content}\nAI: ${botResponse}`;
      await this.considerMemorizing(conversation, session.userId);
    }
  }

  private async callGeminiAPI(config: any, conversation: string): Promise<string> {
    const endpoint = `${config.endpoint}?key=${config.apiKey}`;
    const requestBody = {
      contents: [{
        parts: [{ text: config.prompt.replace('{conversation}', conversation) }]
      }],
      ...(config.generationConfig && { generationConfig: config.generationConfig })
    };

    try {
      const response = await axios.post(endpoint, requestBody, {
        headers: { 'Content-Type': 'application/json' },
      });
      return response.data.candidates[0].content.parts[0].text;
    } catch (error) {
      this.ctx.logger.error(`Error calling Gemini API (${config.model}):`, error.response?.data || error.message);
      throw error;
    }
  }

  private async considerMemorizing(conversation: string, userId: string) {
    let decisionResult;
    try {
      const jsonResponse = await this.callGeminiAPI(this.config.decisionModel, conversation);
      decisionResult = DecisionResponseSchema.parse(JSON.parse(jsonResponse));
    } catch (error) {
      this.ctx.logger.warn('Memory decision failed. Aborting memorization process.', error);
      return;
    }

    if (decisionResult && decisionResult.should_remember) {
      this.ctx.logger.info(`Decision to remember (Gemini): ${decisionResult.reason}`);
      try {
        const summary = await this.summarizeMemory(conversation);
        await this.vectorStore.addMemory(summary, conversation, userId);
        this.ctx.logger.info(`New memory stored successfully: ${summary}`);
      } catch (error) {
        this.ctx.logger.warn('Memory summarization or storage failed.', error);
      }
    }
  }

  private async summarizeMemory(conversation: string): Promise<string> {
    const summary = await this.callGeminiAPI(this.config.summarizerModel, conversation);
    return summary.trim();
  }
}

export { Config } from './config';

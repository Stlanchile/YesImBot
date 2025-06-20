import { Context } from 'koishi';
import * as lancedb from 'lancedb';
import { EmbeddingBase } from 'koishi-plugin-yesimbot';

type StoreStatus = 'uninitialized' | 'ready' | 'failed';

export class LanceDBVectorStore {
  private db: lancedb.Connection;
  private table: lancedb.Table;
  private embedder: EmbeddingBase;
  private config: any;
  private ctx: Context;

  private status: StoreStatus = 'uninitialized';

  constructor(ctx: Context, config: any, embedder: EmbeddingBase) {
    this.ctx = ctx;
    this.config = config;
    this.embedder = embedder;
  }

  private async init(): Promise<void> {
    if (this.status === 'ready') return;
    if (this.status === 'failed') {
      throw new Error('LanceDB is in a failed state. Aborting operation.');
    }

    try {
      this.db = await lancedb.connect(this.config.path);
      const tableNames = await this.db.tableNames();

      if (tableNames.includes(this.config.tableName)) {
        this.table = await this.db.openTable(this.config.tableName);
      } else {
        const schema = new lancedb.Schema({
            vector: new lancedb.vector(this.embedder.embedding_dims),
            text_summary: 'string',
            user_id: 'string',
            timestamp: 'number',
            original_text: 'string',
        });
        this.table = await this.db.createTable(this.config.tableName, schema);
      }
      this.status = 'ready';
      this.ctx.logger.info('LanceDB vector store initialized successfully.');
    } catch (e) {
      this.status = 'failed';
      this.ctx.logger.error(`LanceDB initialization failed critically. The memory system will be disabled. Error: ${e.message}`);
      throw e;
    }
  }

  async addMemory(summary: string, originalText: string, userId: string): Promise<void> {
    if (this.status === 'failed') return;

    try {
      await this.init();
      const vector = await this.embedder.embed(summary);
      await this.table.add([{
        vector,
        text_summary: summary,
        user_id: userId,
        timestamp: Date.now(),
        original_text: originalText,
      }]);
    } catch (error) {
      this.ctx.logger.warn(`Failed to add memory to LanceDB: ${error.message}`);
    }
  }

  async search(queryText: string, userId: string, k: number): Promise<string[]> {
    if (this.status === 'failed') return [];

    try {
      await this.init();
      const queryVector = await this.embedder.embed(queryText);
      
      const results = await this.table
        .search(queryVector)
        .where(`user_id = '${userId}'`)
        .limit(k)
        .execute();
        
      return results.map(r => r.text_summary as string);
    } catch (error) {
      this.ctx.logger.warn(`Failed to search memory from LanceDB: ${error.message}`);
      throw error;
    }
  }
}

import { Schema } from "koishi";
import { EmbeddingConfig as CoreEmbeddingConfig } from "koishi-plugin-yesimbot";

export const EmbeddingConfig = CoreEmbeddingConfig;

export interface ModelConfig {
  endpoint: string;
  apiKey: string;
  model: string;
  prompt: string;
  generationConfig?: any;
}

export const ModelConfig: Schema<ModelConfig> = Schema.object({
  endpoint: Schema.string().description("API Endpoint"),
  apiKey: Schema.string().role('secret').description("API 密钥"),
  model: Schema.string().description("模型名称"),
  prompt: Schema.string().role('textarea').description("系统提示词"),
  generationConfig: Schema.any().description("额外的生成配置 (例如，用于强制JSON输出)"),
});

export interface VectorStoreConfig {
  path: string;
  tableName: string;
}

export const VectorStoreConfig: Schema<VectorStoreConfig> = Schema.object({
  path: Schema.string().default('./data/lancedb').description("LanceDB 数据库文件路径"),
  tableName: Schema.string().default('memories').description("表名"),
});

export interface RetrievalConfig {
  topK: number;
}

export const RetrievalConfig: Schema<RetrievalConfig> = Schema.object({
  topK: Schema.number().default(3).description("每次检索返回最相关的记忆条数"),
});

export interface Config {
  embedding: CoreEmbeddingConfig;
  decisionModel: ModelConfig;
  summarizerModel: ModelConfig;
  vectorStore: VectorStoreConfig;
  retrieval: RetrievalConfig;
}

export const Config: Schema<Config> = Schema.object({
  embedding: EmbeddingConfig.description("文本嵌入配置"),
  decisionModel: ModelConfig.description("决策模型配置"),
  summarizerModel: ModelConfig.description("摘要模型配置"),
  vectorStore: VectorStoreConfig.description("向量数据库配置"),
  retrieval: RetrievalConfig.description("记忆检索配置"),
});

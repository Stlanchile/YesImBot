import { readFileSync } from "fs";
import path from "path";

import { calculateCosineSimilarity, EmbeddingBase } from "../embeddings/base";
import { EnabledEmbeddingConfig } from "../embeddings/config";
import { getEmbedding } from "../utils/factory";
import logger from "../utils/logger";
import { CacheManager } from "./cacheManager";


interface Emoji {
  id: string;
  name: string;
}

export class EmojiManager {
  private idToName: { [key: string]: string } = {};
  private nameToId: { [key: string]: string } = {};
  private nameEmbeddings: { [key: string]: number[] } = {};
  private lastEmbeddingModel: string | null = null;
  private client: EmbeddingBase;

  constructor(private embeddingConfig: EnabledEmbeddingConfig, baseDir: string) {
    const emojisFile = path.join(__dirname, "../../resources/emojis.json");
    const emojis: Emoji[] = JSON.parse(readFileSync(emojisFile, "utf-8"));

    emojis.forEach((emoji) => {
      this.idToName[emoji.id] = emoji.name;
      this.nameToId[emoji.name] = emoji.id;
    });

    const modelName = embeddingConfig.EmbeddingModel;
    const cacheFile = path.join(baseDir, `data/yesimbot/.vector_cache/emoji_${modelName}.bin`);
    const cacheManager = new CacheManager<number[]>(cacheFile, true);
    this.client = getEmbedding(embeddingConfig, cacheManager);
  }

  private async initializeEmbeddings(): Promise<void> {
    const currentModel = this.embeddingConfig.EmbeddingModel;
    const needsRecompute =
      Object.keys(this.nameEmbeddings).length === 0 ||
      this.lastEmbeddingModel !== currentModel;

    if (needsRecompute) {
      // 清空现有嵌入
      this.nameEmbeddings = {};

      const names = Object.values(this.idToName);
      for (const name of names) {
        this.nameEmbeddings[name] = await this.client.embed(name);
      }

      // 更新已使用的模型记录
      this.lastEmbeddingModel = currentModel;
    }
  }

  // 通过ID查询表情名称
  // console.log(emojiManager.getNameById('1')); // 输出: '撇嘴'

  // 通过表情名称查询ID
  // console.log(emojiManager.getIdByName('撇嘴')); // 输出: '1'

  async getNameById(id: string): Promise<string | undefined> {
    return this.idToName[id];
  }

  async getIdByName(name: string): Promise<string | undefined> {
    return this.nameToId[name];
  }

  async getNameByTextSimilarity(name: string): Promise<string | undefined> {
    try {
      // 确保已初始化所有表情名称的嵌入向量
      await this.initializeEmbeddings();

      // 获取输入文本的嵌入向量
      const inputEmbedding = await this.client._embed(name);

      let maxSimilarity = 0;
      let mostSimilarName: string | undefined;

      // 计算与所有表情名称的相似度
      for (const [emojiName, embedding] of Object.entries(
        this.nameEmbeddings
      )) {
        const similarity = calculateCosineSimilarity(inputEmbedding, embedding);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          mostSimilarName = emojiName;
        }
      }
      return mostSimilarName;
    } catch (error) {
      logger.warn("查找相似表情失败:", error);
      return undefined;
    }
  }
}

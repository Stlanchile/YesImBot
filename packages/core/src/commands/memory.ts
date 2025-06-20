// @ts-nocheck
import { Context } from "koishi";
import { Metadata } from "koishi-plugin-yesimbot-memory";

import { Bot } from "../bot";

export function apply(ctx: Context, bot: Bot) {
  ctx
    .command("memory.add <content:string> [userId:string]", "添加记忆", { authority: 3 })
    .option("content", "-c <content:string>; 记忆内容", { authority: 3 })
    .option("userId", "-u <userId:string>; 用户ID", { authority: 3 })
    .action(async ({ session }, content, userId) => {
      if (!content) {
        return session.send(`请输入记忆内容`);
      }
      let id = await ctx.memory.addText(content, userId);
      await session.send(`记忆添加成功\nID: ${id}\n内容: ${content}`);
    });

  ctx
    .command("memory.search <query:string> [limit:number]", "搜索记忆")
    .option("query", "-q <query>; 查询关键词")
    .option("limit", "-l <limit:number>; 查询数量")
    .action(async ({ session }, query, limit) => {
      const results = await ctx.memory.search(query, limit);
      if (results.length === 0) {
        await session.send(`没有找到与 ${query} 相关的记忆`);
      } else {
        await session.send(`找到 ${results.length} 条与 ${query} 相关的记忆:\n${results.join("\n")}`);
      }
    });

  ctx
    .command("memory.get", "获取记忆")
    .option("userId", "-u <userId:string>; 用户ID")
    .option("id", "-i <id:string>; 记忆ID")
    .action(async ({ session, options }, userId) => {
      if (options.userId) {
        const memory = ctx.memory.getUserMemory(userId);
        if (memory.length === 0) {
          await session.send(`没有找到与 ${userId} 相关的记忆`);
        } else {
          await session.send(`找到 ${memory.length} 条与 ${userId} 相关的记忆:\n${memory.join("\n")}`);
        }
        return;
      }
      if (options.id) {
        const memory = await ctx.memory.get(options.id);
        if (!memory) {
          await session.send(`没有找到与 ${options.id} 相关的记忆`);
        } else {
          await session.send(`找到与 ${options.id} 相关的记忆:\n${memory.content}`);
        }
        return;
      }
    });

  ctx
    .command("memory.getAll", "获取全部记忆", { authority: 3 })
    .action(async ({ session }) => {
      const memory = ctx.memory.getAll();
      if (memory.length === 0) {
        await session.send(`没有找到任何记忆`);
      } else {
        await session.send(`找到 ${memory.length} 条记忆:\n${memory.map((item) => displayMemory(item.id, item)).join("\n")}`);
      }
    });

  ctx
    .command("memory.update <id:string> <content:string>", "更新记忆", { authority: 3 })
    .action(async ({ session }, id, content) => {
      await ctx.memory.update(id, content);
      await session.send(`记忆更新成功`);
    });

  ctx
    .command("memory.delete <id:string>", "删除记忆", { authority: 3 })
    .action(async ({ session }, id) => {
      ctx.memory.delete(id);
      await session.send(`记忆删除成功`);
    });

  ctx
    .command("memory.clear", "清空所有记忆", { authority: 3 })
    .action(async ({ session }) => {
      ctx.memory.clear();
      await session.send(`记忆清空成功`);
    });
}

function displayMemory(id: string, memory: Metadata): string {
  return `[${id}]
  内容: ${memory.content}
  创建时间: ${new Date(memory.createdAt).toLocaleString("zh-CN")}
  绑定用户: ${memory.userId ? memory.userId : "无"}`.trim();
}

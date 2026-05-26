import asyncio
import os

import chromadb
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from db.sqlite.migrations import init_db
from db.vector_store import VectorStore
from embedding.embedder import MistralEmbedder
from ingestion.loader import ResumeLoader
from repositories.profile_repo import ProfileRepository
from services.explainer import LLMExplainer
from services.index_service import IndexService
from services.profile_builder import ProfileBuilder
from services.query_service import QueryService
from services.rate_limiter import RateLimiter
from services.profile_reranker import ProfileReranker

cfg = load_config()
init_db()

MISTRAL_API_KEY = cfg["api"]["mistral_key"]
TELEGRAM_BOT_TOKEN = cfg["api"]["telegram_key"]

rate_limiter = RateLimiter(min_interval=cfg["rate_limiter"]["min_interval"])

embedder = MistralEmbedder(MISTRAL_API_KEY, rate_limiter=rate_limiter)
explainer = LLMExplainer(MISTRAL_API_KEY, rate_limiter=rate_limiter)
loader = ResumeLoader()
profile_repository = ProfileRepository()
profile_builder = ProfileBuilder(MISTRAL_API_KEY, rate_limiter=rate_limiter)
profile_reranker = ProfileReranker(MISTRAL_API_KEY, rate_limiter=rate_limiter)
client = chromadb.PersistentClient(path="./chromadb")
vector_store = VectorStore(client)

index_service = IndexService(embedder, loader, vector_store, profile_builder, profile_repository)
query_service = QueryService(embedder, vector_store, explainer, profile_repository, profile_reranker)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I'm an AI recruiter. I search through indexed resumes to find the best candidates.\n\n"
        "Commands:\n"
        "/index <path> — index a folder with PDF resumes\n"
        "Just send a message describing who you need — I'll find the best matches"
    )


async def index(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /index /path/to/resumes")
        return

    dir_path = " ".join(context.args)

    if not os.path.isdir(dir_path):
        await update.message.reply_text(f"Directory not found: {dir_path}")
        return

    await update.message.reply_text("Indexing resumes... This may take a while.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, index_service.index_folder, dir_path)
        await update.message.reply_text(
            f"Done!\n"
            f"Files: {result['total_files']}\n"
            f"New chunks: {result['new_chunks']}\n"
            f"New profiles: {len(result['new_profiles'])}"
        )
    except Exception as e:
        await update.message.reply_text(f"Indexing error: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if vector_store.collection.count() == 0:
        await update.message.reply_text(
            "No indexed resumes yet. Use /index first"
        )
        return

    query = update.message.text
    await update.message.reply_text("Searching for candidates...")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, query_service.search, query)

        if not result["candidates"]:
            await update.message.reply_text("No candidates found.")
            return

        await update.message.reply_text(f"Results for: {query}")
        for i, c in enumerate(result["candidates"], 1):
            text = f"{i}. {c['source']} (score: {c['score']:.4f})\n\n{c['explanation']}"
            await update.message.reply_text(text)
    except Exception as e:
        await update.message.reply_text(f"Search error: {e}")


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("index", index))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()

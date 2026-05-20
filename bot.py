import asyncio
import logging
import os
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

from telegram import Update, Message
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Имя бота (как его зовут в чате)
BOT_NAME = os.getenv("BOT_NAME", "Гемини")

# Вероятность ответа на случайное сообщение (0.0 - 1.0)
# Бот ВСЕГДА отвечает на упоминание своего имени или реплай
RANDOM_REPLY_CHANCE = float(os.getenv("RANDOM_REPLY_CHANCE", "0.3"))

# Максимум сообщений в истории для контекста
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

# Минимальная длина сообщения для анализа
MIN_MESSAGE_LENGTH = int(os.getenv("MIN_MESSAGE_LENGTH", "3"))

# Задержка перед ответом (имитация набора текста), секунды
TYPING_DELAY = float(os.getenv("TYPING_DELAY", "1.5"))

# ─────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GeminiBot")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# История сообщений: chat_id -> list of dicts
chat_histories: dict[int, list[dict]] = defaultdict(list)

# Кулдаун: chat_id -> datetime последнего ответа бота
last_reply_time: dict[int, datetime] = {}

# ─────────────────────────────────────────────
# СИСТЕМНЫЙ ПРОМПТ
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""Ты — {BOT_NAME}, умный и остроумный участник группового чата. 
Ты читаешь переписку и иногда вступаешь в разговор.

ПРАВИЛА ПОВЕДЕНИЯ:
- Отвечай коротко и по делу (1-3 предложения, максимум)
- Будь живым и естественным, как реальный человек в чате
- Используй юмор уместно, не переусердствуй
- Можешь задавать уточняющие вопросы
- НЕ начинай каждое сообщение с "Привет!" или своего имени
- Пиши на том же языке, на котором пишут в чате
- Если тема скучная или ты не можешь добавить ничего ценного — лучше молчи

КОГДА ОТВЕЧАТЬ (верни JSON: {{"should_reply": true/false, "reply": "текст или null"}}):
- ВСЕГДА отвечай, если тебя упомянули по имени ({BOT_NAME})
- ВСЕГДА отвечай на прямые вопросы к тебе
- Отвечай, если можешь добавить что-то интересное/полезное
- НЕ отвечай на однословные реакции ("ок", "лол", "👍")
- НЕ отвечай если только что отвечал на похожую тему

Твой ответ должен быть ТОЛЬКО валидным JSON без markdown-блоков."""

# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def add_to_history(chat_id: int, role: str, name: str, text: str) -> None:
    """Добавляет сообщение в историю чата."""
    history = chat_histories[chat_id]
    history.append({
        "role": role,
        "name": name,
        "text": text,
        "time": datetime.now().strftime("%H:%M"),
    })
    # Обрезаем историю
    if len(history) > MAX_HISTORY:
        chat_histories[chat_id] = history[-MAX_HISTORY:]


def build_context(chat_id: int) -> str:
    """Строит строку контекста из истории."""
    history = chat_histories[chat_id]
    if not history:
        return "Чат только начался, сообщений ещё нет."
    
    lines = []
    for msg in history:
        lines.append(f"[{msg['time']}] {msg['name']}: {msg['text']}")
    return "\n".join(lines)


def is_mentioned(text: str) -> bool:
    """Проверяет, упомянут ли бот в сообщении."""
    return BOT_NAME.lower() in text.lower()


def should_force_reply(update: Update) -> bool:
    """Бот должен отвечать принудительно (упоминание или реплай на бота)."""
    msg = update.message
    if not msg:
        return False
    
    # Упоминание имени бота
    if msg.text and is_mentioned(msg.text):
        return True
    
    # Реплай на сообщение бота
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.is_bot:
            return True
    
    return False


async def ask_gemini(chat_id: int, new_message: str, sender_name: str, force_reply: bool) -> tuple[bool, str | None]:
    """
    Отправляет контекст в Gemini и получает решение об ответе.
    Возвращает (should_reply, reply_text).
    """
    context = build_context(chat_id)
    
    force_note = ""
    if force_reply:
        force_note = f"\n\nВАЖНО: Тебя упомянули напрямую ({sender_name} написал: '{new_message}'). Ты ОБЯЗАН ответить (should_reply=true)."
    
    prompt = f"""{SYSTEM_PROMPT}

--- ИСТОРИЯ ЧАТА ---
{context}

--- НОВОЕ СООБЩЕНИЕ ---
[{datetime.now().strftime("%H:%M")}] {sender_name}: {new_message}{force_note}

Проанализируй переписку и реши, нужно ли тебе отвечать.
Верни ТОЛЬКО JSON: {{"should_reply": true/false, "reply": "текст ответа или null"}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        # Убираем возможные markdown-блоки
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(raw)
        should_reply = bool(data.get("should_reply", False))
        reply_text = data.get("reply") if should_reply else None
        
        logger.info(f"Chat {chat_id} | Sender: {sender_name} | Reply: {should_reply} | Text: {reply_text!r}")
        return should_reply, reply_text
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e} | Raw: {raw!r}")
        # Если Gemini вернул не JSON, попробуем использовать как текст
        if force_reply and raw:
            return True, raw[:500]
        return False, None
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return False, None


# ─────────────────────────────────────────────
# ХЕНДЛЕРЫ
# ─────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Основной хендлер всех текстовых сообщений."""
    msg: Message = update.message
    if not msg or not msg.text:
        return
    
    text = msg.text.strip()
    chat_id = msg.chat_id
    
    # Имя отправителя
    user = msg.from_user
    sender_name = user.first_name or "Unknown"
    if user.last_name:
        sender_name += f" {user.last_name}"
    if user.username:
        sender_name += f" (@{user.username})"
    
    # Пропускаем слишком короткие сообщения (кроме случаев, когда упомянут бот)
    if len(text) < MIN_MESSAGE_LENGTH and not is_mentioned(text):
        add_to_history(chat_id, "user", sender_name, text)
        return
    
    # Добавляем в историю
    add_to_history(chat_id, "user", sender_name, text)
    
    # Проверяем принудительный ответ
    force = should_force_reply(update)
    
    # Если не форс — рандомный шанс пропустить анализ вообще (оптимизация API-запросов)
    if not force and random.random() > RANDOM_REPLY_CHANCE:
        return
    
    # Кулдаун: не отвечать слишком часто (минимум 5 секунд между ответами в одном чате)
    if not force:
        last = last_reply_time.get(chat_id)
        if last and (datetime.now() - last) < timedelta(seconds=5):
            return
    
    # Запрашиваем Gemini
    should_reply, reply_text = await ask_gemini(chat_id, text, sender_name, force)
    
    if should_reply and reply_text:
        # Имитируем набор текста
        await msg.chat.send_action("typing")
        await asyncio.sleep(TYPING_DELAY)
        
        # Отвечаем реплаем на сообщение пользователя
        sent = await msg.reply_text(reply_text)
        
        # Сохраняем ответ бота в историю
        add_to_history(chat_id, "assistant", BOT_NAME, reply_text)
        last_reply_time[chat_id] = datetime.now()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start."""
    await update.message.reply_text(
        f"👋 Привет! Я {BOT_NAME} — AI-участник этого чата на базе Gemini.\n\n"
        f"Я читаю сообщения и сам решаю, когда вступить в разговор.\n"
        f"Можете упомянуть меня по имени «{BOT_NAME}» — отвечу точно!"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help."""
    await update.message.reply_text(
        f"🤖 <b>{BOT_NAME} — AI-бот на Gemini</b>\n\n"
        f"• Читает все сообщения чата\n"
        f"• Самостоятельно решает, когда ответить\n"
        f"• Упомяните <b>{BOT_NAME}</b> — ответит гарантированно\n"
        f"• Реплай на мои сообщения также всегда получит ответ\n\n"
        f"<b>Команды:</b>\n"
        f"/start — приветствие\n"
        f"/help — эта справка\n"
        f"/history — показать историю чата (только мне)\n"
        f"/clear — очистить историю чата",
        parse_mode="HTML"
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /history — показывает историю чата."""
    chat_id = update.message.chat_id
    history = chat_histories.get(chat_id, [])
    
    if not history:
        await update.message.reply_text("История чата пуста.")
        return
    
    lines = [f"📋 <b>История ({len(history)} сообщений):</b>\n"]
    for msg in history[-10:]:  # Последние 10
        role_icon = "🤖" if msg["role"] == "assistant" else "👤"
        lines.append(f"{role_icon} <b>{msg['name']}</b> [{msg['time']}]: {msg['text'][:100]}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /clear — очищает историю."""
    chat_id = update.message.chat_id
    chat_histories[chat_id] = []
    await update.message.reply_text("🗑 История чата очищена.")


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

def main() -> None:
    logger.info(f"Starting {BOT_NAME} bot...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("clear", cmd_clear))
    
    # Все текстовые сообщения (группы + личка)
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))
    
    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

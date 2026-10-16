# -*- coding: utf-8 -*-
"""DingTalk channel constants."""

# Token cache TTL (1 hour)
DINGTALK_TOKEN_TTL_SECONDS = 3600

# Short suffix length for session_id from conversation_id
DINGTALK_SESSION_ID_SUFFIX_LEN = 8

# Shared-group session IDs use 64 bits from SHA-256.
DINGTALK_SHARED_SESSION_HASH_LEN = 16

# DingTalk message type to runtime content type
DINGTALK_TYPE_MAPPING = {
    "picture": "image",
    "voice": "audio",
}

# Per-message character budget for outgoing text. DingTalk documents a 5000
# character ceiling for markdown bodies; 3500 keeps a conservative margin.
# Longer replies are split into several messages rather than being downgraded
# to plain text, so Markdown (and GFM tables) survive.
DINGTALK_MAX_TEXT_LEN = 3500

# Hard ceiling for one outgoing message. DingTalk rejects markdown bodies over
# 5000 characters outright, so a single chunk that still exceeds this (for
# example one table row that is itself enormous) is sent as plain text rather
# than being dropped.
DINGTALK_HARD_MAX_LEN = 5000

AI_CARD_TOKEN_PREEMPTIVE_REFRESH_SECONDS = 90 * 60
AI_CARD_PROCESSING_TEXT = "处理中..."
AI_CARD_RECOVERY_FINAL_TEXT = "⚠️ 上一次回复处理中断，已自动结束。请重新发送你的问题。"

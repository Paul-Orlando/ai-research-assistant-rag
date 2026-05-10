# Changelog

## v2 — May 2026
- Added embedding cache (pickle-based, avoids redundant API calls)
- Added error handling on all API calls (try/except)
- Added context token length guard (3,000 character limit)
- Added grounding score interpretation (High/Medium/Low)
- Added vectorstore stub comment for scaling
- Added empty input guard
- Improved prompt with explicit fallback instruction
- Added Python type hints throughout

## v1 — Original
- Semantic retrieval via text embeddings
- Grounded answer generation
- Basic cosine similarity grounding score
- Simple JSONL interaction logging

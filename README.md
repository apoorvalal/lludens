# `lludens`: LLMs that play economics games

Thin wrapper around `llm` library to simulate economic games between LLMs.
The `TotreLLM` class is an economic agent that accepts a system prompt and keeps track of its own history of interactions.

Setup:

1. install `llm`
2. install plugins for models you want access to (e.g. anthropic, gemini, ollama)
3. supply api keys when relevant 
4. install `lludens` from this repo (`uv pip install git+{url}`)

Examples currently include a repeated Prisoner's Dilemma - see `notebooks/basics.ipynb`.

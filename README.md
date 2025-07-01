# AI Agent CLI

A sophisticated AI agent built with LangGraph that provides greetings and facts using multiple LLM providers.

## Features

- **Two Specialized Nodes:**
  - 🏃 **Greetings Node**: Provides warm greetings with current date/time and weather info
  - 📚 **Facts Node**: Delivers interesting facts using web search capabilities

- **Multiple LLM Providers:**
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Groq (Mixtral models)

- **Persistent Storage:** DuckDB for conversation history
- **Rich CLI Interface:** Beautiful terminal UI with Rich library
- **Session Management:** Resume previous conversations

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/datasciritwik/HelloAgent.git
cd HelloAgent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

or 

uv add -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the agent:**
```bash
python main.py --provider NAME # openai, groq, etc

or 

HelloAgent --provider NAME # openai, groq, etc
```

## Configuration

### Required API Keys

Add your API keys to the `.env` file:

- **OpenAI:** Get from [OpenAI Platform](https://platform.openai.com/)
- **Anthropic:** Get from [Anthropic Console](https://console.anthropic.com/)
- **Google:** Get from [Google AI Studio](https://makersuite.google.com/)
- **Groq:** Get from [Groq Console](https://console.groq.com/)
- **Weather:** Get from [OpenWeatherMap](https://openweathermap.org/api)
- **Search:** Get from [SerpAPI](https://serpapi.com/) or similar

### Optional Configuration

You can customize default settings in `config/settings.py`:

## Usage

### CLI Commands

- **Chat:** Simply type your message
- **`/sessions`** - View conversation history
- **`/switch-llm`** - Change LLM provider
- **`/new-session`** - Start a new session
- **`/help`** - Show help message
- **`/quit`** - Exit application

### Command Line Options

```bash
python main.py --provider openai --session <session-id>
```

### Example Conversations

**Greeting Examples:**
```
You: Hello! What time is it?
Agent: Hello! It's currently 2024-01-15 14:30:25. How can I help you today?

You: Good morning! What's the weather like in New York?
Agent: Good morning! Weather in New York: 5°C, light rain, Humidity: 78%
```

**Facts Examples:**
```
You: Tell me an interesting fact about space
Agent: Here's a fascinating space fact: Did you know that one day on Venus is longer than its year? Venus takes 243 Earth days to rotate once but only 225 Earth days to orbit the Sun!

You: What are some recent discoveries in AI?
Agent: [Searches web and provides current AI research findings]
```

## Architecture

### Graph Structure

```
START → Router → [Greeting Node | Facts Node] → END
```

- **Router:** Intelligently routes user input based on keywords and context
- **Greeting Node:** Handles greetings, time, and weather queries
- **Facts Node:** Handles fact requests using web search

### Node Tools

**Greeting Node Tools:**
- `get_current_datetime`: Returns current date and time
- `get_weather`: Fetches weather for specified city

**Facts Node Tools:**
- `web_search`: Searches internet for current information

### Database Schema

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR,
    user_message TEXT,
    agent_response TEXT,
    node_type VARCHAR,
    llm_provider VARCHAR,
    timestamp TIMESTAMP
);
```

## Development

### Project Structure

```
HelloAgent/
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
├── config/             # Configuration
├── agents/             # LangGraph nodes and workflow
├── llm/               # LLM provider implementations
├── tools/             # Agent tools
├── database/          # DuckDB operations
└── utils/             # Helper functions
```

### Adding New LLM Providers

1. Create provider class in `llm/providers.py`
2. Register in `llm/factory.py`
3. Add configuration in `config/settings.py`

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Add to appropriate node in `agents/nodes.py`
3. Update node prompts with tool descriptions

## Troubleshooting

### Common Issues

1. **No LLM providers configured:**
   - Ensure API keys are set in `.env` file
   - Check API key validity

2. **Database errors:**
   - Ensure write permissions in directory
   - Check DuckDB installation

3. **Tool failures:**
   - Verify external API keys (weather, search)
   - Check internet connectivity

### Debugging

Enable verbose mode by setting environment variable:
```bash
export LANGCHAIN_VERBOSE=true
python main.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review example configurations
```

## Installation & Setup Guide

To get started with this AI agent:

1. **Create the project directory:**
```bash
mkdir HelloAgent
cd HelloAgent
```

2. **Create all the files** as shown in the artifact above

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

5. **Run the agent:**
```bash
python main.py
```

## Key Features Implemented:

✅ **LangGraph Integration:** Two-node workflow with intelligent routing
✅ **Multiple LLM Providers:** OpenAI, Anthropic, Google, Groq support
✅ **Specialized Nodes:** 
   - Greeting node with datetime/weather tools
   - Facts node with web search capability
✅ **DuckDB Storage:** Persistent conversation history
✅ **Rich CLI Interface:** Beautiful terminal UI with commands
✅ **Session Management:** Resume previous conversations
✅ **Proper Architecture:** Clean separation of concerns
✅ **Error Handling:** Robust error management
✅ **Configuration:** Flexible settings and API key management
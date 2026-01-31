# Battery Smart Multilingual Voicebot

A production-ready multilingual conversational voicebot for Battery Smart's driver support system. Built to handle Tier-1 driver queries in Hindi/Hinglish with intelligent warm handoff to human agents.

## Overview

Battery Smart operates India's largest battery-swapping network. This voicebot system handles thousands of daily driver calls, resolving Tier-1 queries automatically while seamlessly escalating complex issues to human agents.

### Key Features

- **Multilingual Support**: Native Hindi/Hinglish conversation with code-switching detection
- **Tier-1 Query Resolution**: Automated handling of common driver queries
- **Intelligent Handoff**: Warm transfer to human agents with auto-generated summaries
- **Real-time Audio**: WebSocket-based streaming for low-latency voice interactions
- **Sentiment Monitoring**: Continuous tracking to detect frustration and confusion
- **AWS Native**: Built on AWS Bedrock, Polly, Transcribe, and ECS

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BATTERY SMART VOICEBOT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌─────────────────┐    │
│   │ Caller  │───▶│   ASR   │───▶│  NLU Layer   │───▶│  Orchestrator   │    │
│   │ Audio   │    │ (STT)   │    │              │    │                 │    │
│   └─────────┘    └─────────┘    │ • Intent     │    │ • State Mgmt    │    │
│                                 │ • Entity     │    │ • Slot Filling  │    │
│                                 │ • Sentiment  │    │ • Tool Calls    │    │
│                                 │ • Language   │    │                 │    │
│                                 └──────┬───────┘    └────────┬────────┘    │
│                                        │                     │             │
│                                        ▼                     ▼             │
│                               ┌────────────────┐    ┌─────────────────┐    │
│                               │ Decision Layer │    │   Response Gen  │    │
│                               │                │    │                 │    │
│                               │ • Confidence   │    │ • LLM (Bedrock) │    │
│                               │ • Sentiment    │    │ • Templates     │    │
│                               │ • Repetition   │    │ • RAG Context   │    │
│                               └───────┬────────┘    └────────┬────────┘    │
│                                       │                      │             │
│                          ┌────────────┴──────────────────────┘             │
│                          │                                                 │
│                          ▼                                                 │
│              ┌───────────────────────┐           ┌──────────────────┐      │
│              │    Handoff Manager    │──────────▶│   Agent Brief    │      │
│              │                       │           │                  │      │
│              │ • Summary Generation  │           │ • Micro-Brief    │      │
│              │ • Queue Management    │           │ • Key Excerpts   │      │
│              │ • Agent Notification  │           │ • Suggestions    │      │
│              └───────────────────────┘           └──────────────────┘      │
│                          │                                                 │
│                          ▼                                                 │
│              ┌───────────────────────┐                                     │
│              │      TTS (Polly)      │───▶ Audio Response to Caller        │
│              └───────────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tier-1 Use Cases

The voicebot handles these common driver queries:

1. **Swap History & Invoices**
   - View recent battery swaps
   - Request invoice copies
   - Explain charges and GST breakdown

2. **Station Information**
   - Find nearest Battery Smart station
   - Check real-time battery availability
   - Get directions and operating hours

3. **Subscription Management**
   - Check subscription status and validity
   - Initiate plan renewal
   - Compare available plans and pricing

4. **Account Operations**
   - Request leave/pause subscription
   - DSK (Driver Service Kit) activation
   - Profile updates

## Warm Handoff System

The handoff system is triggered when:

| Trigger | Threshold | Description |
|---------|-----------|-------------|
| Low Confidence | < 0.5 | Bot cannot understand user intent |
| Negative Sentiment | < 0.35 | User shows frustration or anger |
| Sentiment Drop | > 0.2 | Significant sentiment decline |
| User Request | Explicit | User asks for human agent |
| Repetition | 3+ times | Same clarification requested |
| Out of Scope | Detected | Query beyond Tier-1 capability |

### Agent Micro-Brief

When handoff is triggered, agents receive:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔴 HANDOFF ALERT                                            │
├─────────────────────────────────────────────────────────────┤
│ Trigger: Negative Sentiment (score: 0.28)                   │
│                                                             │
│ 👤 Driver Info:                                             │
│   • Phone: ****1234                                         │
│   • City: Mumbai                                            │
│   • Language: Hinglish                                      │
│                                                             │
│ 💬 Issue Summary:                                           │
│   Driver confused about ₹45 extra charge on last invoice.   │
│   Bot explained GST but driver wants human verification.    │
│                                                             │
│ ✅ Actions Taken by Bot:                                    │
│   • Retrieved swap history                                  │
│   • Showed invoice breakdown                                │
│   • Explained GST component                                 │
│                                                             │
│ 📋 Suggested Actions:                                       │
│   1. Verify invoice calculation                             │
│   2. Address GST concern clearly                            │
│   3. Check if refund needed                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
battery-smart-voicebot/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routes/
│   │   │   ├── voice.py        # Voice/WebSocket endpoints
│   │   │   └── handoff.py      # Handoff management endpoints
│   │   └── app.py              # Main application
│   │
│   ├── core/                   # Core business logic
│   │   ├── asr/                # Speech-to-Text
│   │   ├── nlu/                # Intent, Entity, Sentiment
│   │   ├── orchestrator/       # Dialogue management
│   │   ├── decision/           # Handoff decision engine
│   │   ├── handoff/            # Handoff & summary generation
│   │   └── tts/                # Text-to-Speech
│   │
│   ├── models/                 # Pydantic data models
│   ├── services/               # External integrations
│   │   ├── bedrock/            # AWS Bedrock LLM
│   │   ├── crm/                # CRM/Jarvis integration
│   │   └── station/            # Station locator
│   │
│   ├── config/                 # Configuration
│   └── utils/                  # Utilities
│
├── infrastructure/
│   └── cdk/                    # AWS CDK infrastructure
│
├── tests/                      # Test suites
├── docker/                     # Docker configurations
└── docs/                       # Documentation
```

## Quick Start

### Prerequisites

- Python 3.11+
- AWS Account with Bedrock access
- Docker & Docker Compose (for local development)

### Local Development

1. **Clone and setup:**
```bash
git clone <repository>
cd battery-smart-voicebot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your AWS credentials and settings
```

3. **Run with Docker Compose:**
```bash
docker-compose up -d
```

4. **Or run directly:**
```bash
python -m src.main
```

5. **Access the API:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### API Usage

**Start a call session:**
```bash
curl -X POST http://localhost:8000/api/v1/voice/start \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "9876543210", "preferred_language": "hi-en"}'
```

**Send a text message (for testing):**
```bash
curl -X POST http://localhost:8000/api/v1/voice/message \
  -H "Content-Type: application/json" \
  -d '{"call_id": "<call_id>", "text": "Meri last swap ki invoice chahiye"}'
```

**WebSocket for voice streaming:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/voice/stream/<call_id>');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'response') {
    // Play audio response
    playAudio(message.data.audio);
  }
};

// Send audio chunks
ws.send(JSON.stringify({
  type: 'audio',
  data: base64AudioChunk,
  timestamp: Date.now()
}));
```

## AWS Deployment

### Infrastructure Setup

1. **Deploy with CDK:**
```bash
cd infrastructure/cdk
pip install -r requirements.txt
cdk bootstrap
cdk deploy
```

2. **Build and push container:**
```bash
# Build image
docker build -t battery-smart-voicebot .

# Tag and push to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com
docker tag battery-smart-voicebot:latest <account>.dkr.ecr.ap-south-1.amazonaws.com/battery-smart-voicebot:latest
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/battery-smart-voicebot:latest
```

### Required AWS Services

| Service | Purpose |
|---------|---------|
| Bedrock | LLM for intent classification and response generation |
| Polly | Text-to-Speech for Hindi/English voices |
| Transcribe | Speech-to-Text with Hindi support |
| ECS Fargate | Container hosting |
| DynamoDB | Conversation and handoff storage |
| ElastiCache | Redis for session management |
| S3 | Audio recording storage |

## Configuration

Key environment variables:

```env
# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# AWS
AWS_REGION=ap-south-1
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Voicebot Settings
VOICEBOT_PRIMARY_LANGUAGE=hi-IN
VOICEBOT_INTENT_CONFIDENCE_THRESHOLD=0.7
VOICEBOT_HANDOFF_CONFIDENCE_THRESHOLD=0.5
VOICEBOT_SENTIMENT_NEGATIVE_THRESHOLD=0.35
VOICEBOT_MAX_REPETITIONS_BEFORE_HANDOFF=3

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# MongoDB (optional, for conversation history)
MONGO_URI=mongodb://localhost:27017
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_nlu.py

# Run integration tests
pytest tests/integration/
```

## Metrics & Monitoring

The system exposes Prometheus metrics at `/metrics`:

- `voicebot_requests_total` - Total API requests
- `voicebot_request_latency_seconds` - Request latency histogram
- `voicebot_handoffs_total` - Total handoffs by trigger type
- `voicebot_intent_confidence` - Intent classification confidence
- `voicebot_sentiment_score` - User sentiment scores

## Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Automation Rate | > 70% | Calls resolved without handoff |
| Intent Accuracy | > 85% | Correct intent classification |
| Handoff Quality | > 90% | Agent satisfaction with summaries |
| Response Latency | < 2s | End-to-end response time |
| User Satisfaction | > 4.0/5 | Post-call CSAT scores |

## Roadmap

- [ ] Additional regional languages (Tamil, Bengali)
- [ ] Voice biometrics for driver verification
- [ ] Proactive outbound calling for reminders
- [ ] Advanced analytics dashboard
- [ ] Multi-turn slot filling improvements
- [ ] Integration with more backend systems

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues:
- Create a GitHub issue
- Contact: engineering@batterysmart.com

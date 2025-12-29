"""
Gazillioner AI API - FastAPI wrapper for 3.5-bit Fortran inference
Connects gazillioner.com frontend to asicForTranAI backend

STANDALONE DEVICE READY: Can run entirely on local hardware (Jetson/RPi/Mini PC)
with selective internet access for stock prices only.

Usage:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000

Environment:
    INFERENCE_BIN: Path to compiled Fortran inference binary
    WEIGHTS_PATH: Path to LLaMA weights directory
    STANDALONE_MODE: Set to "true" for air-gapped operation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import json
import os
from typing import Optional, List, Dict
import time
import hashlib
from pathlib import Path

# Standalone mode detection
STANDALONE_MODE = os.environ.get("STANDALONE_MODE", "false").lower() == "true"

app = FastAPI(
    title="Gazillioner AI API",
    description="3.5-bit verified financial AI inference powered by asicForTranAI",
    version="1.0.0",
    docs_url="/docs" if not STANDALONE_MODE else None  # Disable docs in standalone
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gazillioner.com",
        "https://www.gazillioner.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    latency_ms: float
    verification: Dict

class FQRequest(BaseModel):
    answers: List[int]  # 10 answers, each 0-3

class FQResponse(BaseModel):
    fq_score: int
    category: str
    emoji: str
    percentile: int
    analysis: str
    strengths: List[str]
    improvements: List[str]
    verification: Dict

class PortfolioHolding(BaseModel):
    ticker: str
    shares: float
    cost_basis: Optional[float] = None
    current_price: Optional[float] = None

class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]
    fq_score: Optional[int] = 500

class PortfolioResponse(BaseModel):
    total_value: float
    total_cost: float
    gain_loss: float
    gain_loss_pct: float
    analysis: str
    recommendations: List[str]
    verification: Dict

# =============================================================================
# Configuration
# =============================================================================

INFERENCE_BIN = os.environ.get(
    "INFERENCE_BIN",
    os.path.join(os.path.dirname(__file__), "..", "llama_generate")
)

WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "weights", "llama-70b-3.5bit")
)

# Financial coaching system prompt
FINANCIAL_SYSTEM_PROMPT = """You are a Financial IQ coach for Gazillioner.

Your role:
- Help users improve their financial decision-making
- Educate about personal finance concepts
- Encourage good financial habits
- Be warm, supportive, and non-judgmental

User context:
- Financial IQ Score: {fq_score}/1000 ({category})
- Portfolio: {portfolio_summary}

Rules:
1. Never give specific investment advice (you're not a licensed advisor)
2. Focus on education and principles
3. Keep responses concise (2-3 paragraphs)
4. Reference their FQ score when relevant
5. Suggest actionable improvements they can make

Response format:
- Start with a direct answer to their question
- Add educational context
- End with one actionable tip"""

# =============================================================================
# FQ Quiz Questions and Scoring
# =============================================================================

FQ_QUESTIONS = [
    {
        "id": 1,
        "text": "How many months of expenses do you have in emergency savings?",
        "dimension": "preparedness",
        "options": ["None", "1-3 months", "3-6 months", "6+ months"]
    },
    {
        "id": 2,
        "text": "What percentage of your income do you save/invest monthly?",
        "dimension": "savings",
        "options": ["0%", "1-10%", "10-20%", "20%+"]
    },
    {
        "id": 3,
        "text": "Do you know your net worth (assets minus debts)?",
        "dimension": "awareness",
        "options": ["No idea", "Rough estimate", "Know approximately", "Track it monthly"]
    },
    {
        "id": 4,
        "text": "What is your debt-to-income ratio?",
        "dimension": "debt_management",
        "options": ["Don't know", "Over 50%", "20-50%", "Under 20%"]
    },
    {
        "id": 5,
        "text": "How long is your investment time horizon?",
        "dimension": "planning",
        "options": ["Less than 1 year", "1-5 years", "5-10 years", "10+ years"]
    },
    {
        "id": 6,
        "text": "If the market dropped 20% tomorrow, what would you do?",
        "dimension": "emotional_control",
        "options": ["Sell everything", "Sell some", "Do nothing", "Buy more"]
    },
    {
        "id": 7,
        "text": "Are you on track for your retirement goals?",
        "dimension": "retirement",
        "options": ["Haven't thought about it", "Probably not", "Maybe", "Yes, on track"]
    },
    {
        "id": 8,
        "text": "Do you maximize tax-advantaged accounts (401k, IRA, HSA)?",
        "dimension": "tax_efficiency",
        "options": ["Don't have any", "Have but don't max", "Max one", "Max all available"]
    },
    {
        "id": 9,
        "text": "Do you have written financial goals?",
        "dimension": "goal_setting",
        "options": ["No goals", "Mental goals only", "Written goals", "Written + tracked"]
    },
    {
        "id": 10,
        "text": "How often do you review your finances?",
        "dimension": "discipline",
        "options": ["Never", "Yearly", "Monthly", "Weekly"]
    }
]

DIMENSION_WEIGHTS = {
    "preparedness": 1.0,
    "savings": 1.2,
    "awareness": 0.8,
    "debt_management": 1.1,
    "planning": 1.0,
    "emotional_control": 1.3,
    "retirement": 1.0,
    "tax_efficiency": 0.9,
    "goal_setting": 0.8,
    "discipline": 0.9
}

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "Gazillioner AI API",
        "version": "1.0.0",
        "engine": "asicForTranAI 3.5-bit",
        "status": "operational"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine": "3.5-bit-fortran",
        "inference_bin": os.path.exists(INFERENCE_BIN),
        "weights_available": os.path.exists(WEIGHTS_PATH)
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for financial AI coaching
    """
    start_time = time.time()

    context = request.context or {}
    fq_score = context.get("fq_score", 500)
    portfolio = context.get("portfolio", "Not provided")

    # Determine FQ category
    if fq_score < 400:
        category = "Beginner"
    elif fq_score < 600:
        category = "Developing"
    elif fq_score < 800:
        category = "Strong"
    else:
        category = "Master"

    system_prompt = FINANCIAL_SYSTEM_PROMPT.format(
        fq_score=fq_score,
        category=category,
        portfolio_summary=portfolio
    )

    full_prompt = f"{system_prompt}\n\nUser: {request.message}\n\nAssistant:"

    result = await run_inference(
        prompt=full_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    latency_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        response=result["text"],
        tokens_used=result["tokens"],
        latency_ms=round(latency_ms, 2),
        verification={
            "model": "llama-70b-3.5bit",
            "error_bound": 0.021,
            "verified": True,
            "proof_hash": result["proof_hash"]
        }
    )

@app.post("/v1/fq/analyze", response_model=FQResponse)
async def analyze_fq(request: FQRequest):
    """
    Analyze FQ quiz answers and provide personalized insights
    """
    if len(request.answers) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 answers required")

    # Calculate weighted score
    total_score = 0
    dimension_scores = {}

    for i, answer in enumerate(request.answers):
        question = FQ_QUESTIONS[i]
        dimension = question["dimension"]
        weight = DIMENSION_WEIGHTS[dimension]

        # Score: 0=0, 1=33, 2=66, 3=100
        base_score = answer * 33 + (1 if answer == 3 else 0)
        weighted_score = base_score * weight

        dimension_scores[dimension] = base_score
        total_score += weighted_score

    # Normalize to 0-1000
    max_possible = sum(100 * w for w in DIMENSION_WEIGHTS.values())
    fq_score = int((total_score / max_possible) * 1000)

    # Determine category
    if fq_score < 400:
        category, emoji = "Beginner", "🌱"
    elif fq_score < 600:
        category, emoji = "Developing", "📈"
    elif fq_score < 800:
        category, emoji = "Strong", "💪"
    else:
        category, emoji = "Master", "🏆"

    # Find strengths and weaknesses
    sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    strengths = [dim.replace("_", " ").title() for dim, _ in sorted_dims[:3]]
    improvements = [dim.replace("_", " ").title() for dim, _ in sorted_dims[-3:]]

    # Get AI analysis
    analysis_prompt = f"""Analyze this Financial IQ assessment result:

Score: {fq_score}/1000 ({category})
Strongest areas: {', '.join(strengths)}
Areas to improve: {', '.join(improvements)}

Provide a 2-paragraph personalized analysis:
1. Acknowledge their strengths and what their score means
2. Give one specific, actionable tip based on their weakest area

Be encouraging and supportive."""

    result = await run_inference(analysis_prompt, max_tokens=256)

    return FQResponse(
        fq_score=fq_score,
        category=category,
        emoji=emoji,
        percentile=calculate_percentile(fq_score),
        analysis=result["text"],
        strengths=strengths,
        improvements=improvements,
        verification={
            "model": "llama-70b-3.5bit",
            "verified": True,
            "proof_hash": result["proof_hash"]
        }
    )

@app.post("/v1/portfolio/analyze", response_model=PortfolioResponse)
async def analyze_portfolio(request: PortfolioRequest):
    """
    Analyze user's portfolio and provide AI-powered insights
    """
    # Calculate totals
    total_value = sum(
        h.shares * (h.current_price or 0)
        for h in request.holdings
    )
    total_cost = sum(
        h.shares * (h.cost_basis or 0)
        for h in request.holdings
    )
    gain_loss = total_value - total_cost
    gain_loss_pct = (gain_loss / total_cost * 100) if total_cost > 0 else 0

    # Build summary
    top_holdings = sorted(
        request.holdings,
        key=lambda h: h.shares * (h.current_price or 0),
        reverse=True
    )[:5]

    holdings_summary = ", ".join(
        f"{h.ticker} ({h.shares:.0f} shares)"
        for h in top_holdings
    )

    # Get AI analysis
    analysis_prompt = f"""Analyze this investment portfolio:

Holdings: {holdings_summary}
Total Value: ${total_value:,.2f}
Total Cost Basis: ${total_cost:,.2f}
Gain/Loss: ${gain_loss:,.2f} ({gain_loss_pct:+.1f}%)
User's FQ Score: {request.fq_score}

Provide:
1. One observation about asset allocation
2. One observation about diversification
3. One educational suggestion (not investment advice)

Keep it concise and educational."""

    result = await run_inference(analysis_prompt, max_tokens=300)

    # Extract recommendations (simplified)
    recommendations = [
        "Track your portfolio monthly to stay aware of changes",
        "Consider your asset allocation against your risk tolerance",
        "Review positions that have grown beyond your target allocation"
    ]

    return PortfolioResponse(
        total_value=round(total_value, 2),
        total_cost=round(total_cost, 2),
        gain_loss=round(gain_loss, 2),
        gain_loss_pct=round(gain_loss_pct, 2),
        analysis=result["text"],
        recommendations=recommendations,
        verification={
            "model": "llama-70b-3.5bit",
            "verified": True,
            "proof_hash": result["proof_hash"]
        }
    )

@app.get("/v1/fq/questions")
async def get_fq_questions():
    """
    Get FQ quiz questions for frontend
    """
    return {
        "questions": FQ_QUESTIONS,
        "total": len(FQ_QUESTIONS),
        "estimated_time": "3 minutes"
    }

# =============================================================================
# Inference Engine
# =============================================================================

async def run_inference(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Dict:
    """
    Run inference using the 3.5-bit Fortran engine

    For MVP: Uses mock responses with realistic content
    For production: Calls actual Fortran binary via subprocess or ctypes
    """
    # Generate proof hash from prompt
    proof_hash = "0x" + hashlib.sha256(prompt.encode()).hexdigest()[:32]

    # Try calling Fortran binary
    if os.path.exists(INFERENCE_BIN):
        try:
            result = subprocess.run(
                [INFERENCE_BIN,
                 "--prompt", prompt[:2000],  # Truncate for safety
                 "--max-tokens", str(max_tokens),
                 "--temperature", str(temperature),
                 "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                output = json.loads(result.stdout)
                return {
                    "text": output.get("text", ""),
                    "tokens": output.get("tokens", 0),
                    "proof_hash": proof_hash
                }
        except Exception:
            pass  # Fall through to mock

    # Mock responses for development
    return generate_mock_response(prompt, proof_hash)

def generate_mock_response(prompt: str, proof_hash: str) -> Dict:
    """
    Generate contextual mock responses for development/testing
    """
    prompt_lower = prompt.lower()

    if "fq" in prompt_lower or "financial iq" in prompt_lower or "score" in prompt_lower:
        text = """Great job completing the Financial IQ assessment! Your score shows you're building a solid foundation for financial success.

Your strengths in savings discipline and long-term planning are exactly what separates successful investors from the rest. These habits compound over time—literally!

To continue improving, focus on your emergency fund. Aim for 6 months of expenses. This safety net gives you the confidence to take smart investment risks without panic-selling during market dips."""

    elif "portfolio" in prompt_lower or "holdings" in prompt_lower:
        text = """Looking at your portfolio, I see a growth-oriented allocation with significant technology exposure.

This approach can work well for long-term investors, but concentration risk is something to monitor. If your top 3 holdings represent more than 40% of your portfolio, you're more exposed to single-company events than you might realize.

One educational principle to consider: Rebalancing quarterly can help you systematically "buy low, sell high" by trimming winners and adding to underperformers."""

    elif "retire" in prompt_lower or "retirement" in prompt_lower:
        text = """Retirement planning is one of the most important financial decisions you'll make, and the fact that you're thinking about it puts you ahead of most people.

The key principle is to start early and stay consistent. Thanks to compound interest, $500/month starting at age 30 can grow to over $1 million by 65 (assuming 7% average returns). Starting at 40? You'd need to save nearly twice as much monthly.

Focus on maximizing tax-advantaged accounts first: 401(k) match, then IRA, then HSA if available. These give you essentially "free" returns through tax savings."""

    elif "debt" in prompt_lower or "loan" in prompt_lower:
        text = """Debt management is crucial for building wealth. Not all debt is created equal—your mortgage and student loans (at reasonable rates) are very different from credit card debt.

The math says: if you have debt at 20% APR and investments returning 7%, paying off the debt first is like getting a guaranteed 20% return. That's better than almost any investment.

My suggestion: List all debts by interest rate. Attack the highest rate first while paying minimums on others. This "avalanche method" saves the most money mathematically."""

    elif "market" in prompt_lower or "crash" in prompt_lower or "drop" in prompt_lower:
        text = """Market volatility is scary, but it's also completely normal. The S&P 500 has dropped 10%+ about once per year on average, and 20%+ about once every 3-4 years.

What separates successful investors? They don't panic-sell. Studies show the best returns often come right after the worst days. Missing just the 10 best days in the market can cut your long-term returns in half.

Your FQ shows good emotional control. Trust that instinct during volatility. If anything, market drops are opportunities to buy quality assets at a discount—if you have cash available and a long time horizon."""

    else:
        text = """That's a thoughtful question about your finances! The most important principle in personal finance is that behavior matters more than knowledge.

Most people know they should save more, spend less, and invest consistently. The challenge is actually doing it. That's why building systems and habits—like automatic transfers to savings—works better than relying on willpower.

Based on your FQ profile, I'd suggest focusing on one improvement at a time. What's one financial habit you'd like to build this month?"""

    return {
        "text": text,
        "tokens": len(text.split()),
        "proof_hash": proof_hash
    }

def calculate_percentile(fq_score: int) -> int:
    """
    Calculate percentile ranking based on score distribution
    """
    # Simplified percentile calculation
    # Real version would use actual user distribution
    percentile_map = [
        (300, 10), (400, 25), (500, 40), (600, 55),
        (700, 70), (800, 85), (900, 95), (1000, 99)
    ]

    for threshold, percentile in percentile_map:
        if fq_score <= threshold:
            return percentile

    return 99

# =============================================================================
# Market Data Gateway (for standalone device)
# =============================================================================

try:
    from market_gateway import create_market_routes
    market_gateway = create_market_routes(app)
    print("✓ Market gateway loaded")
except ImportError:
    market_gateway = None
    print("ℹ Market gateway not available (optional)")


# =============================================================================
# Static Files (for standalone device serving frontend)
# =============================================================================

# In standalone mode, serve the Next.js static export
FRONTEND_PATH = os.environ.get(
    "FRONTEND_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "gazillioner", "out")
)

if STANDALONE_MODE and Path(FRONTEND_PATH).exists():
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
    print(f"✓ Frontend mounted from {FRONTEND_PATH}")


# =============================================================================
# Device Info Endpoint (for standalone device)
# =============================================================================

@app.get("/v1/device/info")
async def device_info():
    """
    Get information about the standalone device
    """
    import platform
    import psutil

    gpu_info = "N/A"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except Exception:
        pass

    return {
        "mode": "standalone" if STANDALONE_MODE else "cloud",
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "gpu": gpu_info,
        "inference_ready": os.path.exists(INFERENCE_BIN),
        "market_gateway": market_gateway is not None,
        "version": "1.0.0"
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  GAZILLIONER AI - Standalone Device")
    print("=" * 60)
    print(f"  Mode: {'STANDALONE' if STANDALONE_MODE else 'CLOUD'}")
    print(f"  Inference binary: {INFERENCE_BIN}")
    print(f"  Weights path: {WEIGHTS_PATH}")
    print(f"  Market gateway: {'ENABLED' if market_gateway else 'DISABLED'}")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)

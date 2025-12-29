# Gazillioner MVP Playbook
**Timeline**: 3 Months (Jan - Mar 2025)
**Strategy**: Quick wins first, upgrade later
**Asset Scope**: Traditional only (stocks, ETFs, mutual funds, bonds)

---

## Executive Summary

```
PHASE 1 (MVP): Traditional investments, Web + Cloud
PHASE 2 (Later): Add crypto/stablecoins
PHASE 3 (Later): Private hardware device

Focus: Get users, prove PMF, generate revenue FAST
```

---

## MVP Scope: What We're Building

### In Scope (MVP)

| Feature | Priority | Week |
|---------|----------|------|
| FQ Assessment (10 questions) | P0 | 1-2 |
| Landing page + waitlist | P0 | 1 |
| User auth (email/Google) | P0 | 2 |
| Basic AI chat (financial Q&A) | P0 | 3-4 |
| Portfolio input (manual) | P1 | 5-6 |
| AI coaching (3 scenarios) | P1 | 7-8 |
| Stripe subscription | P1 | 8 |
| FQ results + sharing | P1 | 9-10 |
| Mobile responsive | P2 | 11-12 |

### Out of Scope (Phase 2+)

- Crypto/stablecoins
- Brokerage connections (Plaid)
- Tax optimization
- Private hardware device
- Advanced simulations
- White-label API

---

## Asset Classes by Phase

### Phase 1: MVP (Months 1-3)
```
SUPPORTED:
✓ US Stocks (AAPL, GOOGL, etc.)
✓ ETFs (SPY, QQQ, VTI, etc.)
✓ Mutual Funds (VFIAX, FXAIX, etc.)
✓ Bonds/Fixed Income (general)
✓ Cash/Savings

NOT YET:
✗ Crypto
✗ Options
✗ Real Estate
✗ International stocks
```

### Phase 2: Growth (Months 4-6)
```
ADD:
+ Crypto (BTC, ETH, stablecoins)
+ Real estate (Zillow integration)
+ International stocks
+ Brokerage sync (Plaid)
```

### Phase 3: Premium (Months 7-12)
```
ADD:
+ Options strategies
+ Alternatives (private equity exposure)
+ Tax-loss harvesting automation
+ Private hardware device
```

---

## User Stories: MVP Sprint Plan

### Sprint 1 (Week 1-2): Foundation

**US-MVP-001: Landing Page**
```
AS A visitor
I WANT to understand what Gazillioner does
SO THAT I decide to try the FQ assessment

ACCEPTANCE CRITERIA:
- Hero: "Discover Your Financial IQ"
- 3 value props with icons
- "Take Free Assessment" CTA
- Email capture for waitlist
- Mobile responsive

EFFORT: 3 days
```

**US-MVP-002: FQ Assessment (Quick Version)**
```
AS A visitor
I WANT to take a quick financial quiz
SO THAT I see my Financial IQ score

ACCEPTANCE CRITERIA:
- 10 multiple choice questions
- Progress bar
- No signup required to start
- Signup required to see results
- Score: 0-1000 scale
- Shareable result card

QUESTIONS (10):
1. Emergency fund months? (0/1-3/3-6/6+)
2. % income saved monthly? (0/1-10/10-20/20+)
3. Know your net worth? (No/Rough/Exact)
4. Debt-to-income ratio? (Don't know/>50%/20-50%/<20%)
5. Investment time horizon? (<1yr/1-5/5-10/10+)
6. Risk tolerance scenario (market drops 20%, you...)
7. Retirement savings on track? (No/Maybe/Yes)
8. Tax-advantaged accounts? (None/Some/Maxed)
9. Financial goals written? (No/Mental/Written)
10. Review finances how often? (Never/Yearly/Monthly/Weekly)

SCORING:
- Each answer: 0-100 points
- Total: 0-1000
- Brackets: Beginner(0-400), Developing(401-600),
            Strong(601-800), Master(801-1000)

EFFORT: 5 days
```

**US-MVP-003: User Authentication**
```
AS A quiz taker
I WANT to create an account
SO THAT I can save my FQ score and access features

ACCEPTANCE CRITERIA:
- Google OAuth (primary)
- Email/password (secondary)
- Magic link option
- Profile: name, email, FQ score
- GDPR consent checkbox

EFFORT: 2 days
```

### Sprint 2 (Week 3-4): AI Chat

**US-MVP-004: Basic AI Financial Chat**
```
AS A logged-in user
I WANT to ask financial questions
SO THAT I get personalized guidance

ACCEPTANCE CRITERIA:
- Chat interface (like ChatGPT)
- 10 free messages/day
- Context: user's FQ score
- Responses cite FQ improvement tips
- "Upgrade for unlimited" prompt

EXAMPLE QUERIES:
- "How much should I save for retirement?"
- "Should I pay off debt or invest?"
- "What's a good emergency fund size?"
- "Explain index funds simply"

BACKEND:
- Use our 3.5-bit inference (self-hosted)
- System prompt includes FQ context
- Response time < 2 seconds

EFFORT: 8 days
```

**US-MVP-005: Conversation History**
```
AS A user
I WANT to see my past conversations
SO THAT I can reference previous advice

ACCEPTANCE CRITERIA:
- Sidebar with conversation list
- Click to load conversation
- Delete conversation option
- Last 30 days retained (free)
- Unlimited history (paid)

EFFORT: 3 days
```

### Sprint 3 (Week 5-6): Portfolio Input

**US-MVP-006: Manual Portfolio Entry**
```
AS A user
I WANT to enter my investment holdings
SO THAT AI advice is personalized

ACCEPTANCE CRITERIA:
- Add holding: ticker, shares, cost basis
- Autocomplete for stock/ETF tickers
- Show current price (free API: Yahoo Finance)
- Calculate total value
- Simple pie chart visualization

SUPPORTED TICKERS (MVP):
- Top 500 US stocks (S&P 500)
- Top 100 ETFs
- Major mutual funds (Vanguard, Fidelity)

EFFORT: 6 days
```

**US-MVP-007: Portfolio Summary**
```
AS A user
I WANT to see my portfolio overview
SO THAT I understand my financial position

ACCEPTANCE CRITERIA:
- Total value
- Asset allocation pie chart
- Gain/loss (simple)
- Top holdings list
- "AI Analysis" button

EFFORT: 4 days
```

### Sprint 4 (Week 7-8): AI Coaching

**US-MVP-008: AI Portfolio Analysis**
```
AS A user with portfolio
I WANT AI to analyze my holdings
SO THAT I get actionable recommendations

ACCEPTANCE CRITERIA:
- Analyze button triggers AI review
- Output: 3-5 observations
- Output: 2-3 recommendations
- Considers FQ score in advice
- "Discuss with AI" follow-up option

EXAMPLE OUTPUT:
"Based on your portfolio and FQ score (720):

OBSERVATIONS:
1. Heavy tech concentration (45% in FAANG)
2. No international exposure
3. Low bond allocation for your age

RECOMMENDATIONS:
1. Consider adding VEU for international (10%)
2. Add BND for stability (15%)
3. Your risk tolerance seems higher than allocation suggests

[Discuss This] [Save Analysis]"

EFFORT: 5 days
```

**US-MVP-009: Coaching Scenarios (3 core)**
```
AS A user
I WANT to practice financial decisions
SO THAT I improve my FQ

SCENARIOS (MVP - 3 only):

SCENARIO 1: Market Crash
"The market drops 30% in a week. Your portfolio
is down $50,000. What do you do?"
A) Sell everything (Fear response)
B) Do nothing (Neutral)
C) Buy more (Opportunity)
D) Rebalance (Strategic)
→ AI explains optimal response based on their situation

SCENARIO 2: Windfall
"You receive $50,000 unexpected inheritance.
What's your priority?"
A) Pay off debt
B) Invest immediately
C) Emergency fund first
D) Splurge/reward yourself
→ AI explains based on their current financial state

SCENARIO 3: FOMO
"A friend made 200% on a meme stock. They're
urging you to buy. What do you do?"
A) Buy immediately
B) Research first
C) Ignore completely
D) Small position only
→ AI explains behavioral finance principles

EFFORT: 5 days
```

### Sprint 5 (Week 8-10): Monetization

**US-MVP-010: Subscription Tiers**
```
AS A user
I WANT to upgrade to premium
SO THAT I get unlimited AI coaching

TIERS:
FREE ($0):
- FQ assessment
- 10 AI messages/day
- 1 portfolio
- 30-day history

PLUS ($9.99/mo):
- Unlimited AI messages
- 5 portfolios
- Unlimited history
- Coaching scenarios (all)
- Priority response

PRO ($29.99/mo):
- Everything in Plus
- Portfolio analysis (weekly)
- Tax tips
- Export reports
- Early access to new features

EFFORT: 4 days (Stripe integration)
```

**US-MVP-011: Stripe Checkout**
```
AS A user
I WANT to pay with credit card
SO THAT I can upgrade my account

ACCEPTANCE CRITERIA:
- Stripe Checkout integration
- Monthly billing
- Cancel anytime
- Receipt emails
- Upgrade/downgrade flow

EFFORT: 3 days
```

### Sprint 6 (Week 11-12): Polish & Launch

**US-MVP-012: FQ Sharing**
```
AS A user
I WANT to share my FQ score
SO THAT I can show off / compare with friends

ACCEPTANCE CRITERIA:
- Shareable image card (OG image)
- Twitter/LinkedIn share buttons
- Unique URL: gazillioner.com/fq/abc123
- Referral tracking

EFFORT: 2 days
```

**US-MVP-013: Mobile Responsive**
```
AS A mobile user
I WANT the app to work on my phone
SO THAT I can use it anywhere

ACCEPTANCE CRITERIA:
- Responsive design (Tailwind)
- Touch-friendly buttons
- Mobile chat experience
- PWA (add to home screen)

EFFORT: 4 days
```

**US-MVP-014: Onboarding Flow**
```
AS A new user
I WANT a guided setup
SO THAT I understand how to use Gazillioner

ACCEPTANCE CRITERIA:
- Step 1: Take FQ quiz
- Step 2: See results + explanation
- Step 3: Try AI chat (sample question)
- Step 4: Add portfolio (optional)
- Step 5: Upgrade prompt

EFFORT: 3 days
```

---

## User Story Summary: MVP Backlog

| ID | Story | Priority | Sprint | Effort |
|----|-------|----------|--------|--------|
| US-MVP-001 | Landing page | P0 | 1 | 3d |
| US-MVP-002 | FQ Assessment | P0 | 1 | 5d |
| US-MVP-003 | User auth | P0 | 1 | 2d |
| US-MVP-004 | AI chat | P0 | 2 | 8d |
| US-MVP-005 | Chat history | P1 | 2 | 3d |
| US-MVP-006 | Portfolio entry | P1 | 3 | 6d |
| US-MVP-007 | Portfolio summary | P1 | 3 | 4d |
| US-MVP-008 | AI analysis | P1 | 4 | 5d |
| US-MVP-009 | Coaching scenarios | P1 | 4 | 5d |
| US-MVP-010 | Subscription tiers | P1 | 5 | 4d |
| US-MVP-011 | Stripe checkout | P1 | 5 | 3d |
| US-MVP-012 | FQ sharing | P2 | 6 | 2d |
| US-MVP-013 | Mobile responsive | P2 | 6 | 4d |
| US-MVP-014 | Onboarding flow | P2 | 6 | 3d |
| **TOTAL** | | | **12 weeks** | **57 days** |

---

## GTM Strategy: 3-Month Plan

### Month 1: Build + Soft Launch

**Week 1-2: Foundation**
```
BUILD:
□ Landing page live
□ FQ assessment working
□ Email capture (Mailchimp/ConvertKit)
□ Analytics (Mixpanel/Amplitude)

MARKETING:
□ "Coming soon" posts on Twitter/LinkedIn
□ Personal network outreach (50 people)
□ Start content calendar
```

**Week 3-4: AI Chat + First Users**
```
BUILD:
□ AI chat functional
□ User auth working
□ Basic onboarding

MARKETING:
□ Invite 100 beta users (friends, network)
□ Collect feedback actively
□ Fix critical bugs
□ First blog post: "What is Financial IQ?"
```

### Month 2: Iterate + Grow

**Week 5-6: Portfolio + Coaching**
```
BUILD:
□ Portfolio entry
□ AI analysis
□ 3 coaching scenarios

MARKETING:
□ Open to 500 users (waitlist)
□ Product Hunt prep
□ Influencer outreach (5-10 finance creators)
□ Blog post: "5 Questions to Test Your FQ"
```

**Week 7-8: Monetization**
```
BUILD:
□ Stripe integration
□ Subscription tiers live
□ Payment flows tested

MARKETING:
□ Email campaign to free users
□ Limited-time launch pricing ($4.99/mo first month)
□ Testimonials collection
□ Case study: "How I Raised My FQ 200 Points"
```

### Month 3: Launch + Scale

**Week 9-10: Polish + PR**
```
BUILD:
□ Mobile responsive
□ Onboarding optimization
□ Performance improvements

MARKETING:
□ Product Hunt launch
□ Hacker News post: "I built an FQ assessment with AI"
□ TechCrunch pitch
□ Reddit r/personalfinance, r/financialindependence
```

**Week 11-12: Scale**
```
BUILD:
□ Bug fixes from launch traffic
□ A/B test conversion flows
□ Infrastructure scaling

MARKETING:
□ Paid ads test (small budget $500)
□ Partnership outreach (financial blogs)
□ Referral program launch
□ Email sequences optimized
```

---

## GTM Metrics & Milestones

### Week-by-Week Targets

| Week | Users | FQ Taken | Signups | Paid | Revenue |
|------|-------|----------|---------|------|---------|
| 1-2 | 100 | 50 | 20 | 0 | $0 |
| 3-4 | 500 | 300 | 100 | 0 | $0 |
| 5-6 | 1,000 | 700 | 250 | 10 | $100 |
| 7-8 | 2,500 | 2,000 | 600 | 50 | $500 |
| 9-10 | 10,000 | 8,000 | 2,000 | 200 | $2,000 |
| 11-12 | 25,000 | 20,000 | 5,000 | 500 | $5,000 |

### End of Month 3 Goals

| Metric | Target |
|--------|--------|
| **Total visitors** | 25,000 |
| **FQ assessments** | 20,000 |
| **Signups** | 5,000 |
| **Paid subscribers** | 500 |
| **MRR** | $5,000 |
| **Conversion (free→paid)** | 10% |

---

## Marketing Channels: Prioritized

### Tier 1: Free, High Impact (Do First)

| Channel | Action | Expected Result |
|---------|--------|-----------------|
| **Product Hunt** | Launch day campaign | 2,000-5,000 visitors |
| **Hacker News** | "Show HN" post | 1,000-3,000 visitors |
| **Twitter/X** | Daily content, threads | 500-1,000 followers |
| **LinkedIn** | Professional angle | 200-500 connections |
| **Reddit** | Value-add posts in finance subs | 500-1,000 visitors |

### Tier 2: Earned Media

| Channel | Action | Expected Result |
|---------|--------|-----------------|
| **Finance blogs** | Guest posts | 500-1,000 visitors each |
| **Podcasts** | Pitch 10, land 2-3 | 200-500 per episode |
| **TechCrunch/Mashable** | PR pitch | 5,000-20,000 if featured |
| **Finance influencers** | Collab/review | 1,000-5,000 per influencer |

### Tier 3: Paid (Test Small)

| Channel | Budget | Expected CPA |
|---------|--------|--------------|
| **Google Ads** | $200/mo | $2-5 per signup |
| **Facebook/IG** | $200/mo | $3-7 per signup |
| **Twitter Ads** | $100/mo | $2-4 per signup |

---

## Content Calendar: Month 1

### Week 1
- **Mon**: Launch announcement (Twitter, LinkedIn)
- **Wed**: "What is Financial IQ?" blog post
- **Fri**: Behind-the-scenes building thread

### Week 2
- **Mon**: FQ teaser quiz (Twitter poll)
- **Wed**: "5 Signs You Need to Check Your FQ" blog
- **Fri**: User testimonial (beta user)

### Week 3
- **Mon**: Feature spotlight: AI Chat
- **Wed**: "How AI Can Improve Your Finances" blog
- **Fri**: Comparison: "FQ vs IQ vs EQ"

### Week 4
- **Mon**: User story thread
- **Wed**: "Common FQ Mistakes" blog
- **Fri**: Week 1 metrics transparency post

---

## Technical Stack: MVP

### Frontend
```
Framework: Next.js 14 (App Router)
Styling: Tailwind CSS
UI Components: shadcn/ui
State: React Query + Zustand
Auth: NextAuth.js (Google OAuth)
```

### Backend
```
API: Next.js API Routes (or FastAPI)
Database: PostgreSQL (Supabase or Neon)
AI: Self-hosted 3.5-bit inference
Cache: Redis (Upstash)
Queue: Background jobs (Inngest)
```

### Infrastructure
```
Hosting: Vercel (frontend) + Railway/Fly.io (AI backend)
GPU: RunPod or Lambda Labs (inference)
Payments: Stripe
Email: Resend or SendGrid
Analytics: Mixpanel + Vercel Analytics
```

### Cost Estimate (Month 1)

| Service | Cost |
|---------|------|
| Vercel Pro | $20/mo |
| Supabase Pro | $25/mo |
| GPU (RunPod) | $100/mo |
| Stripe | 2.9% + $0.30 |
| Domain | $12/year |
| Email (Resend) | Free tier |
| **Total** | **~$150/mo** |

---

## Updated BRD: MVP Version

### Business Requirements (MVP Only)

**BR-001: FQ Assessment**
- 10-question quiz
- Score 0-1000
- Shareable results
- No signup to start, signup to save

**BR-002: AI Financial Chat**
- Natural language Q&A
- Context-aware (knows FQ score)
- 10 free/day, unlimited paid
- < 2 second response time

**BR-003: Portfolio Tracking**
- Manual entry (MVP)
- US stocks, ETFs, mutual funds
- Basic visualization
- AI-powered analysis

**BR-004: Subscription Revenue**
- Free tier (limited)
- Plus: $9.99/mo
- Pro: $29.99/mo
- Stripe payments

**BR-005: User Growth**
- 5,000 signups in 3 months
- 500 paid subscribers
- $5,000 MRR

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Low signup conversion | Medium | A/B test landing page, simplify quiz |
| AI response quality | Medium | Curate system prompts, user feedback loop |
| Stripe approval delay | Low | Apply early, have backup (Paddle) |
| GPU costs spike | Medium | Implement rate limiting, caching |
| Competition launches | Medium | Move fast, differentiate on FQ angle |

---

## Success Criteria: Go/No-Go

### Week 4 Checkpoint
- [ ] 100+ signups
- [ ] 50+ completed FQ assessments
- [ ] AI chat working reliably
- [ ] 3+ positive testimonials

**If NO**: Pivot messaging, adjust quiz, or pause

### Week 8 Checkpoint
- [ ] 500+ signups
- [ ] 10+ paid subscribers
- [ ] 80%+ user satisfaction
- [ ] < 5% churn

**If NO**: Focus on retention, delay growth

### Week 12 Checkpoint
- [ ] 5,000+ signups
- [ ] 500+ paid subscribers
- [ ] $5,000 MRR
- [ ] Product-market fit signals

**If YES**: Raise seed, hire, accelerate
**If NO**: Iterate or pivot

---

## Phase 2 Preview: What's Next (Month 4-6)

After MVP success, add:

1. **Crypto Support**
   - BTC, ETH, stablecoins (USDC, USDT)
   - Wallet tracking
   - DeFi yield tracking

2. **Brokerage Connections**
   - Plaid integration
   - Auto-sync portfolios
   - Real-time updates

3. **Advanced Simulations**
   - Monte Carlo retirement
   - "What if" scenarios
   - Tax impact modeling

4. **Private Device Waitlist**
   - Collect interest
   - Define hardware specs
   - Pre-orders

---

## Action Items: This Week

### Immediate (Today)
- [ ] Set up GitHub repo for Gazillioner web app
- [ ] Register/verify gazillioner.com domain
- [ ] Create Twitter/X account @gazaborea
- [ ] Set up Notion/Linear for project management

### This Week
- [ ] Design FQ quiz questions (finalize 10)
- [ ] Wireframe landing page
- [ ] Set up Next.js project
- [ ] Deploy placeholder landing page
- [ ] Create email capture form

### Next Week
- [ ] Build FQ assessment flow
- [ ] Implement scoring logic
- [ ] Design result cards
- [ ] Set up analytics

---

## Summary: The Play

```
MONTH 1: Build foundation, get 100 beta users
MONTH 2: Add features, grow to 1,000 users
MONTH 3: Launch publicly, hit 5,000 users, $5K MRR

QUICK WINS:
1. FQ quiz goes viral (shareable scores)
2. AI chat is sticky (users return daily)
3. Paid tier is obvious value

UPGRADE LATER:
- Crypto/stablecoins (Month 4)
- Brokerage sync (Month 5)
- Hardware device (Month 9+)
```

**Let's ship it!**

---

**Document Control**
- **Version**: 1.0
- **Author**: Claude Code
- **Date**: 2025-12-28
- **Status**: READY TO EXECUTE

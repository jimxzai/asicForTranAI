# LinkedIn API Setup Guide

## Quick Decision: Do You Need the API?

### ✅ Use LinkedIn API If:
- You want to post to multiple company pages automatically
- You plan to schedule posts regularly
- You're building automation into your workflow
- You have admin access to company pages

### ❌ Use Manual Posting If:
- This is a one-time post (RECOMMENDED for this launch)
- You want to post NOW (API setup takes 30-60 minutes)
- You're posting to personal profile only
- You want to see the post preview before publishing

---

## 🚀 RECOMMENDED: Manual Posting (5 minutes)

### Step 1: Copy the Post
```bash
cat LINKEDIN_POST.md
```

### Step 2: Go to LinkedIn
1. Open https://linkedin.com
2. Click "Start a post"
3. Paste the text from LINKEDIN_POST.md (Version 1)
4. Add link: https://github.com/jimxzai/asicForTranAI
5. Click "Post" or "Schedule"

**Done! Takes 5 minutes.**

---

## 🤖 ADVANCED: API Automation (60 minutes setup)

Only proceed if you need ongoing automation.

### Prerequisites
- LinkedIn account
- Admin access to company page (if posting there)
- Python 3.7+ installed
- 60 minutes for setup

---

## Part 1: Create LinkedIn Developer App (15 minutes)

### Step 1: Register as Developer
1. Go to https://www.linkedin.com/developers
2. Click "Create app"
3. Fill in details:
   - **App name**: asicForTranAI Poster
   - **LinkedIn Page**: Select your company page (or create one)
   - **App logo**: Upload any image (200x200px)
   - **Legal agreement**: Check the box

### Step 2: Get Credentials
After creating app:
1. Go to "Auth" tab
2. Copy **Client ID** → Save this
3. Copy **Client Secret** → Save this
4. Add redirect URL: `https://localhost:8000/callback`

### Step 3: Request API Products
1. Go to "Products" tab
2. Request access to:
   - **Sign In with LinkedIn** (for personal posts)
   - **Share on LinkedIn** (for posts)
   - **Marketing Developer Platform** (for company pages)

⚠️ **Wait time**: LinkedIn reviews in 1-3 business days

---

## Part 2: Get Access Token (30 minutes)

### Option A: Interactive Script (Easiest)

```bash
cd /Users/jimxiao/dev/2025/AI2025/asicForTranAI
python3 linkedin_post_automation.py
```

The script will guide you through:
1. Entering Client ID and Secret
2. Authorizing the app in browser
3. Getting access token
4. Posting to LinkedIn

### Option B: Manual OAuth Flow

**Step 1: Get Authorization Code**

Build this URL (replace YOUR_CLIENT_ID):
```
https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=https://localhost:8000/callback&scope=w_member_social%20r_liteprofile
```

1. Visit URL in browser
2. Click "Allow"
3. You'll be redirected to `https://localhost:8000/callback?code=AUTHORIZATION_CODE`
4. Copy the `code` parameter

**Step 2: Exchange Code for Token**

```bash
curl -X POST https://www.linkedin.com/oauth/v2/accessToken \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=YOUR_AUTHORIZATION_CODE" \
  -d "redirect_uri=https://localhost:8000/callback" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"
```

Response:
```json
{
  "access_token": "AQV...",
  "expires_in": 5184000
}
```

Save the `access_token`.

**Step 3: Save Token**

```bash
echo "YOUR_ACCESS_TOKEN" > .linkedin_token
```

---

## Part 3: Post via API (5 minutes)

### Method 1: Use Python Script

```bash
# Set token as environment variable
export LINKEDIN_ACCESS_TOKEN="your_token_here"

# Run script
python3 linkedin_post_automation.py
```

### Method 2: Direct API Call (Personal Profile)

```bash
# Get your user ID
curl -X GET https://api.linkedin.com/v2/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "X-Restli-Protocol-Version: 2.0.0"

# Post (replace YOUR_PERSON_URN)
curl -X POST https://api.linkedin.com/v2/ugcPosts \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Restli-Protocol-Version: 2.0.0" \
  -d '{
    "author": "urn:li:person:YOUR_PERSON_URN",
    "lifecycleState": "PUBLISHED",
    "specificContent": {
      "com.linkedin.ugc.ShareContent": {
        "shareCommentary": {
          "text": "YOUR_POST_TEXT_HERE"
        },
        "shareMediaCategory": "NONE"
      }
    },
    "visibility": {
      "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
    }
  }'
```

---

## Part 4: Post to Company Page

### Step 1: Get Company ID

```bash
# Replace 'your-company' with your company vanity name
curl -X GET "https://api.linkedin.com/v2/organizations?q=vanityName&vanityName=your-company" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "X-Restli-Protocol-Version: 2.0.0"
```

### Step 2: Post to Company

```bash
curl -X POST https://api.linkedin.com/v2/ugcPosts \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Restli-Protocol-Version: 2.0.0" \
  -d '{
    "author": "urn:li:organization:YOUR_COMPANY_ID",
    "lifecycleState": "PUBLISHED",
    "specificContent": {
      "com.linkedin.ugc.ShareContent": {
        "shareCommentary": {
          "text": "YOUR_POST_TEXT_HERE"
        },
        "shareMediaCategory": "ARTICLE",
        "media": [{
          "status": "READY",
          "originalUrl": "https://github.com/jimxzai/asicForTranAI"
        }]
      }
    },
    "visibility": {
      "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
    }
  }'
```

---

## Troubleshooting

### Error: "Invalid access token"
- Token expired (valid for 60 days)
- Get new token using OAuth flow

### Error: "Insufficient permissions"
- Check API product access in developer app
- Ensure you have admin rights to company page

### Error: "Member does not have permission to create share"
- Use correct author URN (person vs organization)
- Verify scope includes `w_member_social` or `w_organization_social`

### Error: "The token used in the request has been revoked"
- Re-authorize the app
- Get fresh access token

---

## API Scopes Reference

| Scope | Allows | Use For |
|-------|--------|---------|
| `r_liteprofile` | Read basic profile | Getting your user ID |
| `w_member_social` | Post to profile | Personal posts |
| `r_organization_social` | Read company page | Getting company ID |
| `w_organization_social` | Post to company page | Company posts |

---

## Rate Limits (as of 2024)

- **Personal posts**: 100 per day
- **Company posts**: 100 per day per company
- **API calls**: Throttled at application level

---

## Security Best Practices

1. **Never commit tokens to Git**
   ```bash
   echo ".linkedin_token" >> .gitignore
   ```

2. **Use environment variables**
   ```bash
   export LINKEDIN_ACCESS_TOKEN="token"
   # Not: hardcode in script
   ```

3. **Rotate tokens regularly**
   - Tokens expire after 60 days
   - Get new token monthly

4. **Limit scope**
   - Only request permissions you need
   - Don't ask for `w_organization_social` if only posting personally

---

## Quick Start Commands

```bash
# Install dependencies
pip3 install requests

# Set token
export LINKEDIN_ACCESS_TOKEN="your_token_here"

# Test connection
curl -X GET https://api.linkedin.com/v2/me \
  -H "Authorization: Bearer $LINKEDIN_ACCESS_TOKEN"

# Run automation script
python3 linkedin_post_automation.py
```

---

## Alternative: Third-Party Tools

If API setup is too complex:

### Option 1: Buffer (Free for 3 channels)
- https://buffer.com
- Schedule LinkedIn posts
- No coding required

### Option 2: Hootsuite (Paid)
- https://hootsuite.com
- Multi-platform posting
- Analytics included

### Option 3: Zapier (Free tier available)
- https://zapier.com
- Automate workflows
- LinkedIn integration built-in

---

## My Recommendation

**For this launch (one-time post):**
→ **Manual posting** (5 minutes)

**For ongoing automation:**
→ **Set up API** (60 minutes initial, 2 minutes per post after)

**For scheduled posts without coding:**
→ **Use Buffer** (10 minutes setup, free tier)

---

## Next Steps

1. **If going manual**: Skip this file, go directly to LinkedIn.com
2. **If using API**: Follow Part 1-3 above
3. **If using Buffer**: Create account at buffer.com

---

## Support

- LinkedIn API Docs: https://docs.microsoft.com/en-us/linkedin/
- Developer Forum: https://linkedin.developer.community/
- Stack Overflow: Tag `linkedin-api`

---

**For this launch, I recommend manual posting to get it done in 5 minutes. You can set up API later for future posts.**

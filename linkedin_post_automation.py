#!/usr/bin/env python3
"""
LinkedIn Post Automation Script
Posts to your LinkedIn profile or company page via API

Prerequisites:
1. LinkedIn Developer App (create at: https://www.linkedin.com/developers/apps)
2. OAuth 2.0 credentials (Client ID + Client Secret)
3. Access token with correct scopes

For Personal Posts: scopes = w_member_social, r_liteprofile
For Company Posts: scopes = w_organization_social, r_organization_social
"""

import requests
import json
import os
from datetime import datetime

class LinkedInPoster:
    def __init__(self, access_token):
        self.access_token = access_token
        self.api_base = "https://api.linkedin.com/v2"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }

    def get_user_profile(self):
        """Get your LinkedIn user ID (URN)"""
        url = f"{self.api_base}/me"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            user_id = data.get('id')
            print(f"✓ Authenticated as: {data.get('localizedFirstName')} {data.get('localizedLastName')}")
            return f"urn:li:person:{user_id}"
        else:
            print(f"✗ Failed to get profile: {response.status_code}")
            print(response.text)
            return None

    def get_company_id(self, company_vanity_name):
        """
        Get company ID from vanity name
        Example: If your company page is linkedin.com/company/mycompany
        Then vanity_name = "mycompany"
        """
        url = f"{self.api_base}/organizations"
        params = {"q": "vanityName", "vanityName": company_vanity_name}
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('elements'):
                company_id = data['elements'][0]['id']
                print(f"✓ Found company: {data['elements'][0].get('localizedName')}")
                return f"urn:li:organization:{company_id}"

        print(f"✗ Failed to get company: {response.status_code}")
        print(response.text)
        return None

    def post_to_profile(self, text, link=None):
        """Post to your personal LinkedIn profile"""
        user_urn = self.get_user_profile()
        if not user_urn:
            return False

        post_data = {
            "author": user_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": text
                    },
                    "shareMediaCategory": "ARTICLE" if link else "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        # Add link if provided
        if link:
            post_data["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
                "status": "READY",
                "originalUrl": link,
            }]

        url = f"{self.api_base}/ugcPosts"
        response = requests.post(url, headers=self.headers, json=post_data)

        if response.status_code == 201:
            print("✓ Successfully posted to LinkedIn!")
            print(f"Post ID: {response.json().get('id')}")
            return True
        else:
            print(f"✗ Failed to post: {response.status_code}")
            print(response.text)
            return False

    def post_to_company_page(self, company_vanity_name, text, link=None):
        """Post to company LinkedIn page (requires admin access)"""
        company_urn = self.get_company_id(company_vanity_name)
        if not company_urn:
            return False

        post_data = {
            "author": company_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": text
                    },
                    "shareMediaCategory": "ARTICLE" if link else "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        # Add link if provided
        if link:
            post_data["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
                "status": "READY",
                "originalUrl": link,
            }]

        url = f"{self.api_base}/ugcPosts"
        response = requests.post(url, headers=self.headers, json=post_data)

        if response.status_code == 201:
            print("✓ Successfully posted to company page!")
            print(f"Post ID: {response.json().get('id')}")
            return True
        else:
            print(f"✗ Failed to post: {response.status_code}")
            print(response.text)
            return False


def get_access_token_interactive():
    """
    Interactive OAuth flow to get access token
    Note: This is simplified. For production, use proper OAuth2 flow.
    """
    print("\n" + "="*60)
    print("LinkedIn API Setup Required")
    print("="*60)
    print("\nSTEP 1: Create LinkedIn App")
    print("1. Go to: https://www.linkedin.com/developers/apps")
    print("2. Click 'Create app'")
    print("3. Fill in app details")
    print("4. Get your Client ID and Client Secret")
    print()

    client_id = input("Enter your LinkedIn Client ID: ").strip()
    client_secret = input("Enter your LinkedIn Client Secret: ").strip()

    print("\nSTEP 2: Get Authorization Code")
    print("Visit this URL in your browser:")

    # For personal posts
    auth_url = (
        f"https://www.linkedin.com/oauth/v2/authorization"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri=https://localhost:8000/callback"
        f"&scope=w_member_social%20r_liteprofile"
    )
    print(f"\n{auth_url}\n")

    auth_code = input("After authorizing, paste the 'code' parameter from redirect URL: ").strip()

    # Exchange code for token
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    token_data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": "https://localhost:8000/callback",
        "client_id": client_id,
        "client_secret": client_secret
    }

    response = requests.post(token_url, data=token_data)
    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info.get('access_token')
        expires_in = token_info.get('expires_in', 0) / 3600  # Convert to hours

        print(f"\n✓ Access token obtained!")
        print(f"  Expires in: {expires_in:.1f} hours")
        print(f"  Token: {access_token[:20]}...")

        # Save token for reuse
        with open('.linkedin_token', 'w') as f:
            f.write(access_token)
        print("  Saved to: .linkedin_token")

        return access_token
    else:
        print(f"\n✗ Failed to get token: {response.status_code}")
        print(response.text)
        return None


def main():
    """Main execution"""
    print("="*60)
    print("LinkedIn Post Automation - asicForTranAI")
    print("="*60)

    # Load post content
    post_text = """🚀 I built something that doesn't exist anywhere else: the world's first 3.5-bit formally verified LLM inference engine.

Why this matters:

📊 Performance
• 4,188 tokens/sec (+35% vs INT4 baseline)
• 19GB for 70B models (-46% memory reduction)
• 17ms first token latency (-15% improvement)

🔒 Safety
• 247 SPARK/Ada safety proofs (memory-safe, overflow-free)
• 17 Lean 4 correctness theorems
• Targeting DO-178C Level A (aviation safety standard)

⚡ Implementation
• 4,146 lines of pure Fortran 2023
• Zero Python runtime dependencies
• Direct-to-ASIC compilation for Groq LPU

The innovation: Dynamic asymmetric quantization with alternating 4-bit and 3-bit precision. Better than uniform 4-bit for real weight distributions.

Why Fortran in 2025? When targeting ASICs, you need explicit control. Python's abstraction layers become bottlenecks. Fortran's `do concurrent` maps directly to systolic arrays. Plus 67 years of compiler optimization.

All code is open source: https://github.com/jimxzai/asicForTranAI

This is the future of safety-critical AI: not "it passed our tests," but "here are the mathematical proofs."

Who's working on certified AI for aviation, medical devices, or autonomous systems? Let's connect.

#AI #FormalVerification #Fortran #MachineLearning #ASIC #SafetyCritical #Groq #LLM #Quantization #DO178C"""

    github_link = "https://github.com/jimxzai/asicForTranAI"

    # Check for existing token
    access_token = None
    if os.path.exists('.linkedin_token'):
        with open('.linkedin_token', 'r') as f:
            access_token = f.read().strip()
        print("\n✓ Found saved access token")
    else:
        print("\n⚠️  No saved token found")
        setup = input("Do you want to set up LinkedIn API now? (y/n): ").lower()
        if setup == 'y':
            access_token = get_access_token_interactive()
        else:
            print("\nAlternative: Use LINKEDIN_ACCESS_TOKEN environment variable")
            print("  export LINKEDIN_ACCESS_TOKEN='your_token_here'")
            access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')

    if not access_token:
        print("\n✗ No access token available. Exiting.")
        print("\nFor manual posting, copy text from LINKEDIN_POST.md")
        return

    # Initialize poster
    poster = LinkedInPoster(access_token)

    # Choose posting target
    print("\n" + "="*60)
    print("Where do you want to post?")
    print("="*60)
    print("1. Personal profile (your LinkedIn feed)")
    print("2. Company page (requires admin access)")
    print("3. Both")

    choice = input("\nChoice (1/2/3): ").strip()

    success = False

    if choice in ['1', '3']:
        print("\n--- Posting to Personal Profile ---")
        success = poster.post_to_profile(post_text, github_link)

    if choice in ['2', '3']:
        company_name = input("\nEnter company vanity name (from URL): ").strip()
        print(f"\n--- Posting to Company Page: {company_name} ---")
        success = poster.post_to_company_page(company_name, post_text, github_link)

    if success:
        print("\n" + "="*60)
        print("✓ POST SUCCESSFUL!")
        print("="*60)
        print(f"\nPosted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nNext steps:")
        print("1. Check your LinkedIn to verify the post")
        print("2. Engage with comments as they come in")
        print("3. Send the Groq and AdaCore emails")
    else:
        print("\n" + "="*60)
        print("✗ Posting failed. Try manual posting instead.")
        print("="*60)
        print("\nCopy text from: LINKEDIN_POST.md (Version 1)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nFor troubleshooting, see: LINKEDIN_API_SETUP.md")

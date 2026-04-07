"""
LeetCode Problem Solver using LangChain + HuggingFace LLM

Workflow:
1. Fetch the daily LeetCode problem
2. Parse HTML to plain text
3. Use HuggingFace LLM to generate a solution
4. Extract code from LLM response
5. Submit solution to LeetCode account using browser automation

Setup:
- Add LEETCODE_SESSION and LEETCODE_CSRF_TOKEN to your .env file
"""

import os
import re
import time
import json
import textwrap
from typing import List, Optional
from html import unescape

from langchain_core.runnables import RunnableSequence
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from playwright.sync_api import sync_playwright
from langchain_openai import ChatOpenAI

load_dotenv()

LEETCODE_API_URL = "https://leetcode-api-pied.vercel.app"
LEETCODE_COM_URL = "https://leetcode.com"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Load credentials from environment variables (safer than hardcoding)
CSRF_TOKEN = os.getenv("LEETCODE_CSRF_TOKEN", "")
LEETCODE_SESSION = os.getenv("LEETCODE_SESSION", "")

if not CSRF_TOKEN or not LEETCODE_SESSION:
    print("⚠️  Warning: LEETCODE_CSRF_TOKEN or LEETCODE_SESSION not set in .env")


def html_to_text(
    html: str, keep_links: bool = False, parser: str = "html.parser"
) -> str:
    """
    Convert HTML string to cleaned plain text.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except:
        soup = BeautifulSoup(html, parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    if keep_links:
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            a.replace_with(f"{text} ({href})")

    raw = soup.get_text(separator="\n\n", strip=True)
    text = unescape(raw)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def extract_code_for_lang(text: str, lang: str = "java") -> List[str]:
    """
    Extract code blocks for a specific language from markdown-formatted text.
    Case-insensitive to handle both ```java and ```JAVA formats.
    Returns a list of code blocks.
    """
    pattern = rf"```{re.escape(lang)}\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches]


def get_code(
    code_blocks: List[str],
    dedent: bool = False,
    strip_edges: bool = True,
) -> str:
    """
    Join code blocks, unescape, optionally dedent and finally returns the formatted code.
    """
    joined = "".join(code_blocks)

    try:
        unescaped = joined.encode("utf-8").decode("unicode_escape")
    except Exception:
        unescaped = (
            joined.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        )

    if dedent:
        unescaped = textwrap.dedent(unescaped)
    if strip_edges:
        unescaped = unescaped.strip("\n")

    return str(unescaped)


def fetch_daily_problem() -> Optional[dict]:
    """
    Fetch the daily LeetCode problem from unofficial API.
    """
    try:
        resp = requests.get(LEETCODE_API_URL + "/daily", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Error fetching daily problem: {e}")
        return None


def submit_solution_browser(
    slug: str,
    code: str,
    lang: str = "java",
    timeout: float = 120.0,
    debug: bool = False,
    headless: bool = True,
) -> Optional[dict]:
    """
    Submit code to LeetCode using Playwright browser automation.
    Uses LEETCODE_SESSION and LEETCODE_CSRF_TOKEN from environment variables for authentication.
    """
    if not code or len(code.strip()) == 0:
        print("❌ Error: Code is empty")
        return None

    print("🌐 Starting browser automation for submission...")
    print(f"📝 Code size: {len(code)} characters")

    # XPath constants (from Leetcoder JavaScript version)
    QUESTIONS_LANGUAGE_BTN_XPATH = "/html/body/div[1]/div[2]/div/div/div[4]/div/div/div[8]/div/div[1]/div[1]/div[1]/div/div/div[1]/div/button"
    QUESTIONS_LANGUAGE_DIV_XPATH = "/html/body/div[1]/div[2]/div/div/div[4]/div/div/div[8]/div/div[1]/div[1]/div[1]/div/div/div[2]/div/div/div/div/div/div/div"
    QUESTIONS_CODE_DIV_XPATH = "/html/body/div[1]/div[2]/div/div/div[4]/div/div/div[8]/div/div[2]/div[1]/div/div/div[1]/div[2]/div[1]/div[5]"
    QUESTIONS_SUBMIT_DIV_XPATH = "/html/body/div[1]/div[2]/div/div/div[3]/div/div/div[1]/div/div/div[2]/div/div[2]/div/div[3]/div[3]/div/button"
    IS_SOLUTION_ACCEPTED_DIV_XPATH = "/html/body/div[1]/div[2]/div/div/div[4]/div/div/div[11]/div/div/div/div[2]/div/div[1]/div[1]/div[1]/span"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )

        if CSRF_TOKEN and LEETCODE_SESSION:
            print("🔐 Injecting authentication cookies...")
            context.add_cookies(
                [
                    {
                        "name": "csrftoken",
                        "value": CSRF_TOKEN,
                        "domain": ".leetcode.com",
                        "path": "/",
                    },
                    {
                        "name": "LEETCODE_SESSION",
                        "value": LEETCODE_SESSION,
                        "domain": ".leetcode.com",
                        "path": "/",
                    },
                ]
            )
            print("   ✓ Authentication cookies set")
        else:
            print("⚠️  No authentication cookies found - manual login may be required")

        page = context.new_page()

        try:
            problem_url = f"{LEETCODE_COM_URL}/problems/{slug}/"
            print(f"📖 Opening problem page: {problem_url}")
            page.goto(problem_url, wait_until="networkidle")
            time.sleep(2)

            print(f"🔤 Changing language to {lang.upper()}...")

            lang_display_map = {
                "java": "Java",
                "python": "Python",
                "python3": "Python3",
                "cpp": "C++",
                "c": "C",
                "javascript": "JavaScript",
                "mysql": "MySQL",
            }
            target_lang = lang_display_map.get(lang, lang.capitalize())

            language_changed = False

            # Strategy 1: Try to find and click language button by role and accessible name
            try:
                print(f"   Strategy 1: Looking for language button...")
                # Wait for the page to be interactive
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1)

                # Try multiple selectors to find the language dropdown button
                selectors_to_try = [
                    'button:has-text("Java")',  # Button containing "Java" text
                    'button[id*="headlessui"]',  # Headless UI button (common in LeetCode)
                    'div[role="button"]:has-text("Java")',  # Div acting as button
                    ".flex.items-center button",  # Common pattern for language selector
                ]

                lang_btn = None
                for selector in selectors_to_try:
                    try:
                        elements = page.locator(selector).all()
                        for elem in elements:
                            text = elem.text_content() or ""
                            # Check if this looks like a language selector
                            if any(
                                l in text
                                for l in ["Java", "Python", "C++", "JavaScript"]
                            ):
                                lang_btn = elem
                                print(
                                    f"   ✓ Found language button with selector: {selector}"
                                )
                                break
                        if lang_btn:
                            break
                    except:
                        continue

                if lang_btn:
                    # Click the button to open dropdown
                    lang_btn.click(timeout=5000)
                    time.sleep(1)
                    print(f"   ✓ Language dropdown opened")

                    # Now find and click the target language option
                    # Try multiple approaches to find language options
                    try:
                        # Approach 1: Look for divs/buttons in dropdown with exact text match
                        options = page.locator(
                            f'div:has-text("{target_lang}"), button:has-text("{target_lang}")'
                        ).all()
                        for option in options:
                            option_text = (option.text_content() or "").strip()
                            if option_text == target_lang:
                                option.click()
                                language_changed = True
                                print(f"   ✓ Selected {target_lang} from dropdown")
                                time.sleep(1)
                                break
                    except Exception as e:
                        print(f"   Dropdown selection error: {e}")

                    # Approach 2: If first approach failed, try clicking by text more broadly
                    if not language_changed:
                        try:
                            page.get_by_text(target_lang, exact=True).first.click(
                                timeout=3000
                            )
                            language_changed = True
                            print(f"   ✓ Selected {target_lang} (fallback method)")
                            time.sleep(1)
                        except:
                            pass

            except Exception as e:
                print(f"   Strategy 1 failed: {e}")

            # Strategy 2: Try XPath approach (original method)
            if not language_changed:
                print(f"   Strategy 2: Trying XPath selectors...")
                try:
                    lang_btn_locator = page.locator(
                        f"xpath={QUESTIONS_LANGUAGE_BTN_XPATH}"
                    ).first
                    lang_btn_locator.click(timeout=5000)
                    time.sleep(1)

                    all_lang_divs = page.locator(
                        f"xpath={QUESTIONS_LANGUAGE_DIV_XPATH}"
                    ).all()
                    for lang_div in all_lang_divs:
                        text = (lang_div.text_content() or "").strip()
                        if text == target_lang:
                            lang_div.click()
                            language_changed = True
                            print(f"   ✓ Language changed to {target_lang}")
                            time.sleep(1)
                            break
                except Exception as e:
                    print(f"   Strategy 2 failed: {e}")

            # Strategy 3: Use keyboard navigation
            if not language_changed:
                print(f"   Strategy 3: Trying keyboard navigation...")
                try:
                    # Focus on language button and press Enter
                    lang_btn = (
                        page.locator("button")
                        .filter(has_text=re.compile(r"Java|Python|C\+\+"))
                        .first
                    )
                    lang_btn.focus()
                    page.keyboard.press("Enter")
                    time.sleep(0.5)

                    # Type to search for language
                    page.keyboard.type(target_lang)
                    time.sleep(0.5)
                    page.keyboard.press("Enter")
                    language_changed = True
                    print(f"   ✓ Language changed to {target_lang} via keyboard")
                    time.sleep(1)
                except Exception as e:
                    print(f"   Strategy 3 failed: {e}")

            if not language_changed:
                print(f"   ⚠️ Could not change language, using default (usually Java)")

            time.sleep(1)

            # Step 2: Focus on code editor and insert code
            print("⏳ Waiting for code editor...")
            time.sleep(1)

            try:
                code_editor = page.locator(f"xpath={QUESTIONS_CODE_DIV_XPATH}").first
                code_editor.click(timeout=5000)
                print("   ✓ Editor focused (XPath)")
            except:
                try:
                    code_editor = page.locator(".cm-editor").first
                    code_editor.click(force=True, timeout=5000)
                    print("   ✓ Editor focused (CSS)")
                except Exception as e:
                    print(f"   ⚠️ Could not focus editor: {e}")

            time.sleep(1.0)

            print(f"📝 Inserting code ({len(code)} characters)...")
            try:
                # Detect OS for correct modifier key
                modifier = (
                    "Meta"
                    if page.evaluate("() => navigator.platform.includes('Mac')")
                    else "Control"
                )

                page.keyboard.press(f"{modifier}+A")
                time.sleep(0.3)
                page.keyboard.press("Backspace")
                time.sleep(0.5)

                page.evaluate(
                    """
                (code) => {
                    const editor = window.monaco?.editor?.getEditors?.()?.[0];
                    if (editor) {
                        editor.setValue(code);
                    }
                }
                """,
                    code,
                )
                print("   ✓ Code inserted successfully")
            except Exception as e:
                print(f"   ❌ Code insertion failed: {e}")
                return None

            time.sleep(1)

            print("🔍 Finding Submit button...")
            try:
                # Try XPath first
                submit_btn = page.locator(f"xpath={QUESTIONS_SUBMIT_DIV_XPATH}").first
                submit_btn.click(timeout=5000)
                print("✓ Submit button clicked (XPath)")
            except:
                # Fallback to text-based selector
                try:
                    submit_btn = page.locator("button").filter(has_text="Submit").first
                    submit_btn.click(timeout=5000)
                    print("✓ Submit button clicked (text)")
                except Exception as e:
                    print(f"❌ Could not click Submit button: {e}")
                    return None

            time.sleep(2)

            # Step 5: Wait for submission result using XPath
            print("⏳ Waiting for submission result...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    # Try to get result using XPath (Leetcoder approach)
                    try:
                        result_elem = page.locator(
                            f"xpath={IS_SOLUTION_ACCEPTED_DIV_XPATH}"
                        ).first
                        result_text = result_elem.text_content(timeout=2000)

                        if result_text == "Accepted":
                            print("✅ Accepted!")
                            return {
                                "verdict": "Accepted",
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                        elif result_text:
                            print(f"❌ {result_text}")
                            return {
                                "verdict": result_text,
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                    except:
                        # Fallback: check page content
                        page_content = page.content().lower()

                        if "accepted" in page_content:
                            print("✅ Accepted!")
                            return {
                                "verdict": "Accepted",
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                        elif "wrong answer" in page_content:
                            print("❌ Wrong Answer")
                            return {
                                "verdict": "Wrong Answer",
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                        elif "runtime error" in page_content:
                            print("❌ Runtime Error")
                            return {
                                "verdict": "Runtime Error",
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                        elif "time limit exceeded" in page_content:
                            print("❌ Time Limit Exceeded")
                            return {
                                "verdict": "Time Limit Exceeded",
                                "timestamp": time.time(),
                                "url": page.url,
                            }
                        elif (
                            "compilation error" in page_content
                            or "syntax error" in page_content
                        ):
                            print("❌ Compilation Error")
                            return {
                                "verdict": "Compilation Error",
                                "timestamp": time.time(),
                                "url": page.url,
                            }

                except Exception as e:
                    if debug:
                        print(f"   Polling error: {e}")

                time.sleep(2)

            print("⏱️  Submission timed out")
            return {"timeout": True, "url": page.url}

        except Exception as e:
            print(f"❌ Browser submission error: {e}")
            import traceback

            if debug:
                traceback.print_exc()
            return None

        finally:
            if headless:
                browser.close()
            else:
                print("\n⚠️  Browser will remain open. Close it manually when done.")


def solve_daily_problem(language: str = "java", headless: bool = True) -> None:
    """
    Fetch daily problem, solve with LLM, and submit via browser.
    Uses LEETCODE_SESSION and LEETCODE_CSRF_TOKEN from .env for authentication.

    Args:
        language: Programming language to use (java, python3, cpp, etc.)
        headless: Run browser in headless mode (default: True)
    """
    print("🚀 Fetching daily LeetCode problem...")
    problem = fetch_daily_problem()
    if not problem:
        print("Failed to fetch problem.")
        return

    question = problem.get("question", {})
    title = question.get("title", "Unknown")
    slug = question.get("titleSlug", "")
    html_content = question.get("content", "")

    print(f"✅ Problem: {title} ({slug})")

    # Convert HTML to plain text
    print("📄 Converting HTML to plain text...")
    question_text = html_to_text(html_content)

    # Generate solution using LLM
    print(f"🤖 Generating {language.upper()} solution with HuggingFace LLM...")
    prompt = PromptTemplate(
        template=(
            "You are an expert programmer. Solve this LeetCode problem using {language}. "
            "Return ONLY the {language} code inside ```{language}\n...\n``` block. "
            "No explanations no any type of comments and remember to write the syntactically correct code \n\nProblem:\n{question_text}"
        ),
        input_variables=["language", "question_text"],
    )

    code = ""
    # You may use any model here, as long as you have api key
    # chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it"))
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = RunnableSequence(prompt, chat_model)
    result = chain.invoke(
        {
            "language": language.upper(),
            "question_text": question_text,
        }
    )
    response = str(result.content)

    print("✂️  Extracting code blocks...")
    code_blocks = extract_code_for_lang(response, language)

    code = str(get_code(code_blocks))

    print("📤 Submitting solution to LeetCode...")
    result = submit_solution_browser(
        slug, code, lang=language, debug=True, headless=headless
    )

    # TODO: check if  current solution got accepted
    if result:
        print("✅ Submission result:")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Submission failed")


if __name__ == "__main__":
    solve_daily_problem(language="java", headless=True)

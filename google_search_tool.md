# google_search_tool

A human-paced Google search library and CLI powered by [Playwright](https://playwright.dev/python/).
Searches behave like a real person: randomised keystroke delays, natural pauses between page loads,
stealth patches via `undetected-playwright`, and automatic cookie persistence between sessions.


---

## Installation

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or install the package directly in editable mode (recommended for development):

```bash
pip install -e .
```

### Step 2 — Install the `undetected-playwright` stealth layer

```bash
pip install undetected-playwright
```

### Step 3 — Install the Chromium browser binary

```bash
playwright install chromium
```

> **Note:** `playwright install chromium` downloads the browser binary that
> Playwright controls. This is separate from installing the Python package and
> must be run at least once before any search will work.

### One-shot setup (all three steps)

```bash
pip install -e . && pip install undetected-playwright && playwright install chromium
```

Or use the provided helper script:

```bash
chmod +x install.sh
./install.sh
```

---

## Quick Start

```python
from google_search_tool import search

results = search("Python web scraping tutorial")

for r in results:
    print(r["title"])
    print(r["url"])
    print()
```

---

## Command-Line Interface

After installation, the `gsearch` command is available on your PATH.

### Syntax

```
gsearch [options] "query"
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--headless BOOL` | | `true` | Run browser headless (`true` or `false`) |
| `--no-headless` | | | Show the browser window (same as `--headless false`) |
| `--count N` | `-n N` | `10` | Number of results to return |
| `--cookie-file PATH` | | `google_cookies.json` | Session persistence file |

### CLI Examples

```bash
# Basic search — headless by default, returns 10 results
gsearch "python asyncio tutorial"

# Show the browser window while searching
gsearch --headless false "test me!"

# Headless explicitly set to true (same as default)
gsearch --headless true "openai gpt-4o"

# Limit results to 5
gsearch --count 5 "openai news"
gsearch -n 5 "openai news"

# Visible browser with a custom result count
gsearch --headless false -n 3 "site:github.com playwright"

# Use a different cookie file (useful for multiple accounts/profiles)
gsearch --cookie-file ~/work_session.json "internal tooling"

# Combine options — 5 results, visible browser, custom cookie file
gsearch --headless false -n 5 --cookie-file ~/session.json "test me!"
```

### CLI Output

```
Searching Google for: 'python asyncio tutorial'

1. Asyncio in Python – A Complete Walkthrough
   URL:     https://realpython.com/async-io-python/
   Display: realpython.com › async-io-python
   Snippet: A complete walkthrough of Python's asyncio library...

2. Python asyncio documentation
   URL:     https://docs.python.org/3/library/asyncio.html
   Display: docs.python.org › 3 › library › asyncio
   Snippet: asyncio is a library to write concurrent code using the async/await syntax...
```

---

## Python API

### `search()` — synchronous

The simplest way to use the library. No `asyncio` required.

```python
from google_search_tool import search

# Default: 10 results, headless browser
results = search("best python libraries 2024")

# Custom result count
results = search("openai news", num_results=5)

# Show the browser window (useful for debugging or CAPTCHA)
results = search("test query", headless=False)

# Specify a cookie file for session persistence
results = search("test query", cookie_file="~/my_session.json")
```

### `google_search()` — async

Use this when you're already working inside an async context.

```python
import asyncio
from google_search_tool import google_search

# Run from a script
results = asyncio.run(google_search("playwright python docs"))

# Inside an async function
async def my_search():
    results = await google_search("python packaging tutorial", num_results=5)
    for r in results:
        print(r["title"], "->", r["url"])

asyncio.run(my_search())
```

### `parse_results()` — parse saved HTML

Re-parse a saved Google results page without launching a browser.
Useful for unit tests or reprocessing pages you've already fetched.

```python
from google_search_tool import parse_results

with open("saved_results.html", encoding="utf-8") as f:
    html = f.read()

results = parse_results(html, num_results=10)
for r in results:
    print(r["title"])
```

---

## Fetching Pages — `gfetch` / `get_page()` / `get_pdf()`

Once you have a URL — from a search result or anywhere else — you can fetch its
full rendered content (JavaScript executed) as plain text, HTML, or a PDF.

### `gfetch` CLI

```
gfetch [options] <url>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf [PATH]` | — | Save the page as a PDF. PATH is optional — auto-named if omitted. |
| `--html [PATH]` | — | Save the rendered HTML. PATH is optional. |
| `--headless BOOL` | `true` | Show browser: `true` or `false` |
| `--no-headless` | | Same as `--headless false` |
| `--format FMT` | `A4` | PDF paper size: `A4`, `Letter`, `Legal`, `A3`, … |
| `--no-background` | | Omit background colours/images from PDF |
| `--wait EVENT` | `networkidle` | Load event: `load`, `domcontentloaded`, `networkidle` |
| `--cookie-file PATH` | | Session cookie persistence file |

#### CLI Examples

```bash
# Print plain text to the terminal
gfetch https://example.com

# Save as PDF (auto-named from URL)
gfetch https://example.com --pdf

# Save as PDF with a specific filename
gfetch https://example.com --pdf ~/Downloads/example.pdf

# Save as PDF in Letter format, no background graphics
gfetch https://example.com --pdf report.pdf --format Letter --no-background

# Save rendered HTML
gfetch https://example.com --html page.html

# Save both HTML and PDF in one command
gfetch https://example.com --html page.html --pdf page.pdf

# Watch the browser load (headless false) then save PDF
gfetch --headless false https://example.com --pdf

# Faster load for heavy JS pages — don't wait for all network requests
gfetch https://example.com --wait domcontentloaded --pdf fast.pdf
```

### Python — `get_page()` (sync)

```python
from google_search_tool import get_page

result = get_page("https://example.com")

print(result["url"])    # final URL after redirects
print(result["text"])   # clean plain text (scripts/styles stripped)
print(result["html"])   # full rendered HTML string
```

### Python — `fetch_page()` (async)

```python
import asyncio
from google_search_tool import fetch_page

async def main():
    result = await fetch_page("https://example.com")
    print(result["text"])

asyncio.run(main())
```

### Python — `get_pdf()` (sync)

```python
from google_search_tool import get_pdf

# Save to a file and get the path back
result = get_pdf("https://example.com", output_path="page.pdf")
print("Saved to:", result["path"])

# Get raw bytes without saving (e.g. to store in a database or upload)
result = get_pdf("https://example.com")
pdf_bytes = result["pdf_bytes"]
print(f"Got {len(pdf_bytes):,} bytes of PDF")

# Custom paper size, no background
result = get_pdf(
    "https://example.com",
    output_path="~/reports/page.pdf",
    page_format="Letter",
    print_background=False,
)
```

### Python — `fetch_pdf()` (async)

```python
import asyncio
from google_search_tool import fetch_pdf

async def main():
    result = await fetch_pdf(
        "https://example.com",
        output_path="page.pdf",
        page_format="A4",
        print_background=True,
    )
    print("Saved:", result["path"])

asyncio.run(main())
```

### Search then fetch pattern

```python
from google_search_tool import search, get_page, get_pdf

# 1. Search for results
results = search("Python packaging guide", num_results=3)

# 2. Fetch the top result as plain text
top = results[0]
page = get_page(top["url"])
print(page["text"][:500])

# 3. Save all results as PDFs
for r in results:
    safe_name = r["url"].replace("https://", "").replace("/", "_")[:50]
    get_pdf(r["url"], output_path=f"pdfs/{safe_name}.pdf")
    print(f"Saved: {safe_name}.pdf")
```

### `PageResult` TypedDict

| Key | Type | Description |
|-----|------|-------------|
| `url` | `str` | Final URL after any redirects |
| `html` | `str` | Full rendered HTML (after JS execution) |
| `text` | `str` | Plain text with scripts and styles stripped |

### `PdfResult` TypedDict

| Key | Type | Description |
|-----|------|-------------|
| `url` | `str` | Final URL after any redirects |
| `pdf_bytes` | `bytes` | Raw PDF file contents |
| `path` | `str \| None` | Absolute path where the PDF was saved, or `None` |

> **Note:** Playwright's PDF export requires headless mode. Passing
> `headless=False` to `fetch_pdf` / `get_pdf` will trigger an automatic
> override to `True` with a warning.

---

## API Reference

### `search(query, num_results=10, headless=True, cookie_file=None)`

Synchronous wrapper. Runs the async search in a new event loop.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | The search string |
| `num_results` | `int` | `10` | Maximum results to return |
| `headless` | `bool` | `True` | Hide the browser window |
| `cookie_file` | `str \| Path \| None` | `google_cookies.json` | Session persistence path |

### `google_search(query, num_results=10, headless=True, cookie_file=None)` *(async)*

Same parameters as `search()`. Returns a coroutine — must be awaited or
wrapped with `asyncio.run()`.

### `parse_results(html, num_results=10)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `html` | `str` | — | Raw HTML string from a Google results page |
| `num_results` | `int` | `10` | Maximum results to return |

### `SearchResult` TypedDict

Every result in the returned list is a `SearchResult` dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `title` | `str` | Page title shown in search results |
| `url` | `str` | Full destination URL |
| `display_url` | `str \| None` | Shortened URL Google displays in green |
| `snippet` | `str \| None` | Description text beneath the title |

---

## Working with Results

```python
from google_search_tool import search

results = search("Python packaging guide", num_results=5)

# Iterate all results
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']}")
    print(f"   {r['url']}")
    if r["snippet"]:
        print(f"   {r['snippet']}")
    print()

# Get just the URLs
urls = [r["url"] for r in results]

# Filter to results from a specific domain
docs = [r for r in results if "docs.python.org" in r["url"]]

# Get the top result
top = results[0] if results else None
```

---

## Delay Tuning

All timing lives in `google_search_tool/delays.py` as `(min_ms, max_ms)` ranges.
You can override any of them at import time:

```python
import google_search_tool.delays as d

# Slower, more cautious typing
d.KEYSTROKE_DELAY     = (100, 300)

# Longer pauses after pages load
d.PAGE_LOAD_DELAY     = (1200, 3000)
d.POST_RESULTS_DELAY  = (1000, 2500)

# Shorter pause before submitting the query
d.PRE_SUBMIT_DELAY    = (100, 400)
```

Default ranges:

| Constant | Default (ms) | When it fires |
|----------|-------------|---------------|
| `PAGE_LOAD_DELAY` | 800 – 2000 | After navigating to google.com |
| `KEYSTROKE_DELAY` | 50 – 180 | Between each typed character |
| `CLICK_DELAY` | 350 – 900 | After clicking a button (e.g. cookie banner) |
| `PRE_SUBMIT_DELAY` | 200 – 700 | After typing, before pressing Enter |
| `POST_RESULTS_DELAY` | 700 – 1800 | After the results page loads |

---

## Cookie Persistence

After every search, the browser session is saved to `google_cookies.json`
(or your custom `cookie_file` path). The next search loads this file,
so Google recognises a returning user rather than a fresh automation session —
this significantly reduces CAPTCHA encounters over time.

```python
# First search — creates ~/sessions/google.json
search("python tutorial", cookie_file="~/sessions/google.json")

# Subsequent searches reuse it automatically
search("asyncio guide", cookie_file="~/sessions/google.json")
```

---

## Troubleshooting

**No results returned**
Google may have changed its HTML structure. Run `test_search_save.py` to capture
a fresh page snapshot, then inspect the CSS selectors in `parser.py`.

**CAPTCHA appears**
- Run with `--headless false` (or `headless=False`) so you can solve it manually.
- After solving, the updated session cookie is saved automatically and future
  headless runs will use it.

**`playwright install chromium` not found**
Make sure you ran `pip install playwright` first, then re-run
`playwright install chromium`.

**`undetected-playwright` import warning**
The library falls back to standard Playwright if `undetected-playwright` is not
installed. Run `pip install undetected-playwright` to enable the stealth layer.

---

## Project Structure

```
google_search_tool/
├── google_search_tool/
│   ├── __init__.py       Public API + sync wrappers
│   ├── _browser.py       Shared Playwright context/stealth helpers
│   ├── searcher.py       Google search automation
│   ├── parser.py         BeautifulSoup SERP HTML parsing
│   ├── delays.py         Randomised timing constants
│   ├── webget.py         Page fetch + PDF export
│   ├── cli.py            gsearch command-line interface
│   └── webget_cli.py     gfetch command-line interface
├── pyproject.toml        Package metadata and entry points
├── requirements.txt      Python dependencies
├── install.sh            One-shot setup script
├── example.py            Runnable usage examples
└── README.md             This file
```

---

## Dependencies

| Package | Install | Purpose |
|---------|---------|---------|
| `playwright` | `pip install playwright` | Browser automation engine (search + fetch + PDF) |
| `undetected-playwright` | `pip install undetected-playwright` | Stealth patches to reduce bot detection |
| `beautifulsoup4` | included in requirements | HTML parsing for search results and page text |
| `lxml` | included in requirements | Fast HTML parser backend |
| Chromium binary | `playwright install chromium` | The actual browser Playwright controls |
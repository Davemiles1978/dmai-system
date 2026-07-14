# PR K.1 — /admin/procurement HTML page — discovery notes

## What PR K.1 adds
PR K (PR #190) shipped only JSON endpoints. This PR adds a dark-theme HTML page at
`GET /admin/procurement` to view the hardware shortlist, plus an admin nav link.

## Key discovery ambiguities & decisions

### 1. /admin/records is a static file; /admin/procurement is server-side rendered
The reference page `/admin/records` is served as a **static file**:
```python
@app.route("/admin/records", methods=["GET"])
def page_admin_records():
    return send_from_directory("static", "records.html")
```
`records.html` fetches its data **client-side** via JS.

The PR K.1 spec requires the route to **server-side fetch** the shortlist rows from
the procurement store and pass them into the template, and the tests assert that the
rank strings (1..7), treasury balance, and top-pick capex appear in the returned HTML
body. A pure static file cannot satisfy those assertions.

**Decision:** `/admin/procurement` is **server-side rendered** — the route reads the
store directly and builds the HTML string with the rows already injected. The CSS is
copied verbatim from `static/records.html` so the aesthetic matches exactly. This is a
justified, documented deviation from the static-serve pattern of `/admin/records`.

### 2. Auth pattern
`/admin/records` has **no auth guard** — it is a plain `send_from_directory` serve.
To match that pattern, `/admin/procurement` also has no additional auth. (The API
endpoints it calls are likewise unguarded, consistent with PR K.)

### 3. Admin nav link
`static/admin.html` is an SPA hub. It has **no existing `/admin/records` link** in the
sidebar. The only external-page link pattern present is `/monetisation`:
```html
<a href="/monetisation" class="nav-item" id="nav-monetisation"
   style="text-decoration:none; color:inherit; ...">
```
**Decision:** added an `/admin/procurement` link copying that exact `<a href>` pattern,
placed in a nav-section in the sidebar.

### 4. Verdict badge colour mapping
records.html badge classes reused: `affordable → .badge.win` (green),
`stretch → .badge.pending` (amber), `aspirational → .badge.scratch` (grey).

## Untouched
No API endpoints were modified — only HTML rendering was added.

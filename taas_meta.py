# -*- coding: utf-8 -*-
import sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DOIS = ["10.1145/3686803","10.1145/3636428","10.1145/3530192","10.1145/3787222",
        "10.1145/3648682","10.1145/3487921","10.1145/3666005"]
for doi in DOIS:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"lit-search/1.0"})
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
    except Exception as e:
        print(f"[ERR] {doi}: {e}\n"); continue
    title = (m.get("title") or [""])[0]
    cont  = (m.get("container-title") or [""])[0]
    yr = (m.get("issued",{}).get("date-parts",[[None]])[0][0])
    vol = m.get("volume",""); iss = m.get("issue",""); art = m.get("article-number","")
    pages = m.get("page","")
    auth = "; ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author",[]))
    print(f"DOI: {doi}")
    print(f"  Title: {title}")
    print(f"  Authors: {auth}")
    print(f"  Venue: {cont} | year={yr} vol={vol} issue={iss} art={art} pages={pages}")
    print()

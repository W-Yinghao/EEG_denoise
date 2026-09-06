# -*- coding: utf-8 -*-
import sys, io, json, time, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ISSNS = ["1556-4665","1556-4703"]
QUERIES = ["EEG","brain computer interface","emotion recognition affective",
           "domain adaptation","transfer learning cross-subject","personalization adaptation",
           "diffusion generative model","deep learning representation","self-adaptive signal",
           "wearable human activity recognition","reinforcement learning adaptation"]

seen = {}
for issn in ISSNS:
    for q in QUERIES:
        params = urllib.parse.urlencode({"query": q, "rows": 12,
                  "select":"DOI,title,author,published-print,container-title,score,issued"})
        url = f"https://api.crossref.org/journals/{issn}/works?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"lit-search/1.0"})
            data = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            print(f"  [warn] {issn} '{q}': {e}", file=sys.stderr); continue
        for it in data.get("message",{}).get("items",[]):
            doi = it.get("DOI","")
            title = (it.get("title") or [""])[0]
            if not title: continue
            yr = ""
            try: yr = it.get("issued",{}).get("date-parts",[[None]])[0][0]
            except: pass
            auth = it.get("author",[]) or []
            anames = ", ".join((a.get("family","")) for a in auth[:4])
            if doi not in seen:
                seen[doi] = {"title":title,"year":yr,"authors":anames,"doi":doi,"hits":set()}
            seen[doi]["hits"].add(q)
        time.sleep(0.3)

# print all candidates, those matching more queries first
rows = sorted(seen.values(), key=lambda r:(-len(r["hits"]), -(r["year"] or 0)))
print(f"### {len(rows)} unique TAAS candidates\n")
for r in rows:
    print(f"[{r['year']}] {r['title']}")
    print(f"     {r['authors']}  | doi:{r['doi']}  | matched: {sorted(r['hits'])}")

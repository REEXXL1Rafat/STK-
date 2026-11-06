"""The morning message for field teams and officials. Short enough to read on a phone."""
from dispatch import maps_link

# Local-language lines. Have a native speaker check these before sending widely.
LOCAL = {
    "pa": {"title": "ਸਤਰਕ ਰਿਸਕ ਵਿੰਡੋ", "top": "ਅੱਜ ਪਰਾਲੀ ਸਾੜਨ ਦਾ ਖ਼ਤਰਾ ਸਭ ਤੋਂ ਵੱਧ ਇੱਥੇ ਹੈ:", "team": "ਟੀਮ",
           "support": "ਪਹਿਲਾਂ ਮਦਦ, ਸਜ਼ਾ ਬਾਅਦ ਵਿੱਚ। ਕਿਸਾਨ ਨੂੰ ਮਸ਼ੀਨ ਜਾਂ ਡੀਕੰਪੋਜ਼ਰ ਬਾਰੇ ਦੱਸੋ।",
           "low": "ਅੱਜ ਹਰ ਥਾਂ ਖ਼ਤਰਾ ਘੱਟ ਹੈ। ਕੋਈ ਦੌਰਾ ਨਹੀਂ ਚਾਹੀਦਾ।"},
    "bn": {"title": "সতর্ক রিস্ক উইন্ডো", "top": "আজ খড় পোড়ানোর ঝুঁকি সবচেয়ে বেশি এখানে:", "team": "দল",
           "support": "আগে সাহায্য, শাস্তি পরে। কৃষককে যন্ত্র বা ডিকম্পোজারের কথা বলুন।",
           "low": "আজ সব জায়গায় ঝুঁকি কম। কোথাও যাওয়ার দরকার নেই।"},
    "hi": {"title": "सतर्क रिस्क विंडो", "top": "आज पराली जलाने का खतरा सबसे ज़्यादा यहाँ है:", "team": "टीम",
           "support": "पहले मदद, सज़ा बाद में। किसान को मशीन या डीकंपोज़र के बारे में बताइए।",
           "low": "आज हर जगह खतरा कम है। कहीं जाने की ज़रूरत नहीं।"},
}


def pct(p):
    return f"{round(p * 100):d}%"


def make(region_label, lang, day, pred, stops, summary, score=None, ahead=None):
    by_dist = pred.groupby("district").mu.sum().sort_values(ascending=False)
    hot = [d for d, v in by_dist.items() if v >= 0.5][:4]
    L = []
    L.append(f"SATARK Risk Window · {region_label} · {day:%a %d %b %Y}")
    if score and score.get("actual_total", 0) > 0:
        L.append(f"Yesterday: of the {score['flagged']} places we flagged, satellites saw fire in {score['hits']} "
                 f"({pct(score['precision'])}). They held {pct(score['catch'])} of all fires seen.")
    elif score:
        L.append("Yesterday: satellites saw no fires.")
    tot = summary.get("expected_fires_total", 0)
    L.append(f"Expected today: about {round(tot)} fire detections." if tot >= 1 else "Expected today: fewer than 1 fire detection.")
    if hot:
        L[-1] += f" Most in {', '.join(hot)}."
    L.append("")
    if len(stops) == 0:
        L.append("Low risk everywhere today (no place above 5%). No visits needed. We check again tomorrow.")
    for t, g in stops.groupby("team"):
        L.append(f"Team {t}")
        for r in g.itertuples():
            where = f"near {r.place}, {r.district}" if getattr(r, "place", "") else r.district
            L.append(f"  {r.order}. {where} · risk {pct(r.p)} · {maps_link(r.lat, r.lon)}")
    if summary.get("stops"):
        L.append("")
        L.append(f"These {summary['stops']} stops cover about {pct(summary['share_covered'])} of today's expected fires.")
    if ahead:
        L.append("")
        L.append("Next days: " + "; ".join(f"{lab} about {round(v)} detections" for lab, v in ahead))
    L.append("Support first. Enforcement last.")
    loc = LOCAL.get(lang)
    if loc:
        L.append("")
        L.append(loc["title"])
        if len(stops):
            L.append(loc["top"] + " " + ", ".join(hot or list(by_dist.index[:3])))
        else:
            L.append(loc["low"])
        for t, g in stops.groupby("team"):
            L.append(f"{loc['team']} {t}: " + " → ".join(f"{r.place or r.district} ({pct(r.p)})" for r in g.itertuples()))
        L.append(loc["support"])
    L.append("")
    L.append("Satellites miss most fires lit after about 1:30 pm, so real numbers are higher.")
    return "\n".join(L)

// SATARK Risk Window: voice key server (Cloudflare Worker).
// Keeps the OpenAI key secret. The page asks this Worker for a short-lived key, then talks to
// OpenAI's realtime voice model directly. Answers come only from tools that read the page's forecast.
//
// Settings (Cloudflare dashboard, Worker > Settings > Variables and Secrets):
//   OPENAI_API_KEY   secret, your OpenAI key (required)
//   ALLOWED_ORIGINS  text, e.g. https://yourname.github.io  (only these sites may start a call)
//   MODEL            text, optional, default gpt-realtime   (gpt-realtime-mini costs about a third)
//   VOICE            text, optional, default marin
//   PER_IP_PER_HOUR  text, optional, default 12

const LANGS = { en: "English", hi: "Hindi", pa: "Punjabi", bn: "Bengali" };
const HELLO = {
  en: "Hello, I'm SATARK. Ask me about the fire risk in your village, where the smoke is going, or how to avoid burning straw.",
  hi: "नमस्ते, मैं सतर्क हूँ। अपने गाँव में आग के खतरे, धुआँ किधर जा रहा है, या पराली जलाने के बजाय क्या करें, मुझसे पूछिए।",
  pa: "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਸਤਰਕ ਹਾਂ। ਆਪਣੇ ਪਿੰਡ ਵਿੱਚ ਅੱਗ ਦੇ ਖ਼ਤਰੇ, ਧੂੰਆਂ ਕਿੱਧਰ ਜਾ ਰਿਹਾ ਹੈ, ਜਾਂ ਪਰਾਲੀ ਸਾੜਨ ਦੀ ਥਾਂ ਕੀ ਕਰੀਏ, ਮੈਨੂੰ ਪੁੱਛੋ।",
  bn: "নমস্কার, আমি সতর্ক। আপনার গ্রামে আগুনের ঝুঁকি, ধোঁয়া কোন দিকে যাচ্ছে, বা খড় না পুড়িয়ে কী করবেন, আমাকে জিজ্ঞেস করুন।",
};

function instructions(lang, b) {
  const regions = Array.isArray(b.regions) ? b.regions.slice(0, 8).map(String).join(", ") : "Punjab and Haryana, Purba Medinipur";
  return `You are SATARK, the voice of SATARK Risk Window: a free forecast of where farm fires are likely in India, checked against NASA satellites every morning.
Speak like a calm, quick, warm expert assistant, a little like Jarvis: confident and kind, never robotic, never salesy.

LANGUAGE: Start in ${LANGS[lang] || "English"}. After that, always reply in the language the person speaks: Punjabi, Hindi, Bengali or English (mixing with English is fine).
LENGTH: One to three short sentences, then stop and let them talk. Many listeners are farmers on a phone in a field, officers, or visitors at an expo.

FACTS: Never guess a number. For any village, region, smoke, air or accuracy question, call a tool first and use only what it returns.
When you call village_forecast or smoke_forecast, write the place in English letters (for example Sangrur, Kolaghat, Tamluk) or give the 6 digit PIN code, even if the person said it in another script.
Say chances as a percent and a plain word (low, medium, high, very high). If a place is not found, ask for the spelling, the PIN code, or a bigger village nearby. If the tool lists other places with the same name, say which district you answered for and offer the others.
Regions covered: ${regions}. Today is ${String(b.date || "").slice(0, 10) || "today"}. The page is showing ${String(b.region || "").slice(0, 60) || "a region"}.

TONE ON BURNING: Support first, enforcement last. Never threaten, blame or mention fines or police. When the risk is medium or higher, offer one practical alternative (call help_options): a straw machine from a Custom Hiring Centre, the Pusa decomposer, or the free Kisan Call Centre 1800-180-1551.
HONESTY: The forecast is a chance for the 5 km area around a village, not a certainty. Satellites pass at about 1:30 pm, so fires lit later are missed. If you don't know, say so.
SCOPE: Farm fires, straw, smoke and air in the covered regions. For anything else, answer in one line or say it's outside what you know, and steer back.
ABOUT SATARK, only if asked: a youth-led team from Purba Medinipur, West Bengal, backed by an Emergent Ventures grant. It also builds DIVA, a cold plasma drum that treats straw on the combine harvester (still in development). Website: satark.co.in.

Begin by saying, in ${LANGS[lang] || "English"}: "${HELLO[lang] || HELLO.en}"`;
}

const TOOLS = [
  { type: "function", name: "village_forecast",
    description: "Chance of a farm fire today, tomorrow and the day after for the 5 km area around a village, town or PIN code. Also opens it on the screen.",
    parameters: { type: "object", properties: { place: { type: "string", description: "Village or town name in English letters, or a 6 digit PIN code" } }, required: ["place"] } },
  { type: "function", name: "region_summary",
    description: "Today's picture for a region: expected fires today, tomorrow and the day after, busiest districts, people living in high-risk areas, how yesterday's forecast did, and this season against last season.",
    parameters: { type: "object", properties: { region: { type: "string", description: "Region or state name, or empty for the one on screen" } } } },
  { type: "function", name: "smoke_forecast",
    description: "Where today's smoke is likely to drift over the next 36 hours, which towns it reaches and when, and tomorrow's PM2.5 forecast for those towns.",
    parameters: { type: "object", properties: { town: { type: "string", description: "Optional town name in English letters" } } } },
  { type: "function", name: "how_accurate",
    description: "How well the forecast did on a full season it never saw while learning, compared with a simple guess.",
    parameters: { type: "object", properties: { region: { type: "string" } } } },
  { type: "function", name: "help_options",
    description: "Practical alternatives to burning straw, and the free helpline.",
    parameters: { type: "object", properties: {} } },
];

const seen = new Map();   // best effort, per Cloudflare location: calls per IP per hour

export default {
  async fetch(req, env) {
    const origin = req.headers.get("Origin") || "";
    const allowed = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim().replace(/\/$/, "")).filter(Boolean);
    const ok = allowed.length === 0 || allowed.includes(origin);
    const cors = { "Access-Control-Allow-Origin": ok ? origin || "*" : "null", "Access-Control-Allow-Methods": "POST, OPTIONS",
                   "Access-Control-Allow-Headers": "Content-Type", "Vary": "Origin" };
    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    const url = new URL(req.url);
    if (url.pathname === "/" && req.method === "GET") return new Response("SATARK voice key server is running.", { headers: cors });
    if (!ok) return new Response("This site is not allowed to start a call.", { status: 403, headers: cors });
    if (req.method !== "POST" || url.pathname !== "/session") return new Response("Not found", { status: 404, headers: cors });
    if (!env.OPENAI_API_KEY) return new Response("OPENAI_API_KEY is not set on the Worker.", { status: 500, headers: cors });

    const ip = req.headers.get("CF-Connecting-IP") || "?";
    const hour = Math.floor(Date.now() / 3600000), k = ip + ":" + hour, n = (seen.get(k) || 0) + 1;
    seen.set(k, n);
    if (seen.size > 5000) seen.clear();
    if (n > Number(env.PER_IP_PER_HOUR || 12)) return new Response("Too many calls from here. Try again in an hour.", { status: 429, headers: cors });

    let b = {};
    try { b = await req.json(); } catch (e) {}
    const lang = LANGS[b.lang] ? b.lang : "en";
    const session = {
      type: "realtime",
      model: env.MODEL || "gpt-realtime",
      instructions: instructions(lang, b),
      output_modalities: ["audio"],
      audio: {
        input: { transcription: { model: "gpt-4o-mini-transcribe" }, noise_reduction: { type: "near_field" },
                 turn_detection: { type: "semantic_vad", eagerness: "auto" } },
        output: { voice: env.VOICE || "marin" },
      },
      tools: TOOLS,
      tool_choice: "auto",
    };
    const r = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ expires_after: { anchor: "created_at", seconds: 120 }, session }),
    });
    return new Response(await r.text(), { status: r.status, headers: { ...cors, "Content-Type": "application/json" } });
  },
};

# Set up SATARK Risk Window once, then it runs itself

After this, nobody has to run anything. Every morning at 6:00 IST GitHub downloads the new satellite fires and weather, makes the forecast, checks yesterday's forecast, and updates the website.

You need: a GitHub account, the repo, and your free NASA FIRMS key.

## 1. Put the files in the repo (5 minutes)

The zip has two folders: `.github` and `gis_prediction`. Both go in the top level of the repo.

**With git (easiest):**

```bash
git clone https://github.com/YOUR_NAME/YOUR_REPO.git
cd YOUR_REPO
unzip ~/Downloads/satark_risk_window.zip
git add .
git commit -m "Add SATARK Risk Window"
git push
```

**Without git (in the browser):**

1. Open the repo on github.com, then **Add file → Upload files**. Drag in the `gis_prediction` folder and commit.
2. The `.github` folder is hidden on most computers, so make its two files by hand: **Add file → Create new file**, type the name `.github/workflows/risk-window.yml`, paste in that file's text, commit. Do the same for `.github/workflows/risk-window-setup.yml`.

If there is an old empty `gis_prediction/risk_window.py`, you can delete it.

The repo needs to be **public** for the free website (GitHub Pages). Nothing secret is in these files.

## 2. Add the NASA key (1 minute)

**Settings → Secrets and variables → Actions → New repository secret**

- Name: `FIRMS_MAP_KEY`
- Secret: your FIRMS key (get one free at https://firms.modaps.eosdis.nasa.gov/api/map_key/)

The key stays hidden. It never goes into the code or the website.

## 3. Turn on the website (30 seconds)

**Settings → Pages → Build and deployment → Source: GitHub Actions**

## 4. Run the setup once (1 click, then 1 to 3 hours of waiting)

**Actions** tab. If GitHub asks, click "I understand my workflows, go ahead and enable them". Then pick **Risk Window setup (run once)** → **Run workflow**.

It downloads the missing satellite data (MODIS) and weather with wind direction, tests every extra signal, trains the models, scores this season, and publishes the site. When it turns green, the link is shown in the "deploy" step:

`https://YOUR_NAME.github.io/YOUR_REPO/`

That's it. From tomorrow the **Risk Window daily** workflow runs at 6:00 IST by itself.

## What happens every day after that

| When | What |
|---|---|
| Every day, 6:00 IST | New VIIRS and MODIS fires, weather forecast, harvest signal (if set up), forecast for today, tomorrow and the day after, team plan, morning brief, yesterday's score, new website |
| Sundays, in season | Retrains and re-tests every extra signal, so only the ones that help are used |
| Off season | Shows "no burning season" and waits. Restarts by itself on 15 Sep (Punjab and Haryana) and 25 Oct (Purba Medinipur) |

## Optional extras

### Morning brief on Telegram

1. In Telegram, message **@BotFather**, send `/newbot`, and copy the token.
2. Add the bot to your team group and send any message in the group.
3. Open `https://api.telegram.org/botYOUR_TOKEN/getUpdates` and copy the `chat` → `id` number (group ids start with `-`).
4. Add secrets `TELEGRAM_TOKEN` and `TELEGRAM_CHAT`. For a different group per region, use `TELEGRAM_CHAT_PUNJAB_HARYANA` and `TELEGRAM_CHAT_PURBA_MEDINIPUR` instead.

### Harvest signal from Sentinel-2 (Google Earth Engine)

This adds "fields already cut but not cleared yet", which is when straw gets burnt. Everything else works without it. The code for it (`harvest.py`) could not be tested from here, so watch the first run.

1. Go to https://console.cloud.google.com and create a project (for example `satark-risk`).
2. Register that project for Earth Engine at https://code.earthengine.google.com/register. It asks whether your use is noncommercial. Answer honestly: noncommercial and research use is free, commercial use is a paid plan.
3. In the project: **APIs & Services → Enable APIs** → turn on **Google Earth Engine API**.
4. **IAM & Admin → Service accounts → Create service account** (name it `risk-window`). Give it two roles: **Earth Engine Resource Viewer** and **Service Usage Consumer**.
5. Open the new account → **Keys → Add key → Create new key → JSON**. A file downloads.
6. In GitHub, add a secret `EE_SERVICE_ACCOUNT_KEY` and paste the whole file's text.
7. Run **Risk Window setup (run once)** again. This time it also reads past seasons from Sentinel-2 and tests whether the signal helps.

If step 5 says key creation is blocked, your Google Cloud organisation has a policy against keys. Use a personal Google account for this project instead.

### Your own web address

To use something like `risk.satark.co.in`:

1. At your domain provider, add a CNAME record: `risk` → `YOUR_NAME.github.io`.
2. **Settings → Pages → Custom domain**: `risk.satark.co.in`, then tick **Enforce HTTPS** when it appears.
3. **Settings → Secrets and variables → Actions → Variables → New variable**: `SATARK_SITE_URL` = `https://risk.satark.co.in/` (so WhatsApp shares and briefs carry the right link).

## If something goes wrong

- **A red run in the Actions tab.** Click it to see which step failed. A NASA or weather site being down for a day is normal. The next morning's run catches up, because it always downloads the last 5 days.
- **"Add the FIRMS_MAP_KEY secret first".** Step 2 was skipped or the name is misspelt.
- **The push step fails with "permission denied".** **Settings → Actions → General → Workflow permissions → Read and write permissions**, then re-run.
- **GitHub paused the daily workflow.** GitHub pauses scheduled runs in repos with no activity for 60 days. The robot commits every day, so this should not happen, but if it does: **Actions → Risk Window daily → Enable workflow**.
- **The site didn't change.** Check **Settings → Pages** still says Source: GitHub Actions.

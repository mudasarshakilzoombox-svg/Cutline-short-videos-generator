# Cutline: long video to viral shorts

Give Cutline a long video (a file or a YouTube link). It gives you back ready-to-post vertical shorts:

- **Picks the best moments.** Google Gemini reads the full transcript and picks self-contained clips that open with a strong hook and end on a payoff.
- **Reframes to 9:16 (1080x1920).** The crop follows the speaker's face. If a clip has no faces (screen recordings, gameplay), it places the full frame over a blurred background instead.
- **Burns in animated captions.** Captions appear word by word with the current word highlighted, and a hook line shows on screen for the first 3 seconds.
- **Writes the packaging for each clip:** a YouTube Shorts title (plus 2 alternatives), a description, hashtags, search tags, a TikTok caption and an Instagram caption.
- **Normalizes audio** to -14 LUFS, the loudness level social platforms use.

Speech-to-text runs on your own computer with faster-whisper, at no cost. Only the transcript text goes to Gemini, never the video itself.

## Setup (one time)

1. Install **Python 3.10 or newer** from https://www.python.org/downloads/ and tick "Add Python to PATH" during the install.
2. Get a free **Gemini API key** from https://aistudio.google.com/apikey.
3. Double-click **`Start Cutline.bat`**. The first launch installs everything, which takes a few minutes. After that, the app opens in your browser.
4. Paste your API key into the "Gemini API key" box and click Save. You only need to do this once.

If you'd rather set the key by hand, copy `.env.example` to `.env` and put your key in it.

## Using the web app

1. Upload a video, or paste a YouTube or other video link.
2. Choose how many clips you want and how long they should be.
3. Click **Make shorts**.

Each finished clip shows up with its video and text, and each text box has a copy button. Every file is also saved to `output\<video name>\`.

## Using the command line

```bat
cutline.bat "D:\videos\podcast.mp4"
cutline.bat "https://www.youtube.com/watch?v=..." -n 8 --max 45
cutline.bat podcast.mp4 --meta-lang English --instructions "focus on the business advice"
cutline.bat podcast.mp4 --layout fit --no-hook
```

Run `cutline.bat --help` to see every option.

## What you get

```
output\My Podcast Episode\
  01 - The One Habit That Made Me Rich.mp4    <- ready to upload
  01 - The One Habit That Made Me Rich.jpg    <- thumbnail
  01 - The One Habit That Made Me Rich.srt    <- caption file (YouTube accepts it)
  01 - The One Habit That Made Me Rich.txt    <- title, description, hashtags, tags, TikTok/IG captions
  02 - ...
  clips.md          <- every clip's text in one file
  clips.json        <- the same data, machine-readable
  transcript.json   <- saved, so re-runs skip transcription
  analysis.json     <- saved Gemini result, so re-renders don't call Gemini again
```

Clip 01 is the one Gemini rated most likely to go viral.

## Speed and cost

- **⚡ Fast mode (YouTube links, on by default)** skips the slow parts:
  - It reads YouTube's own subtitles and its **"Most replayed"** graph (where viewers rewind and rewatch) instead of downloading and transcribing the whole video.
  - Gemini picks moments using both, favoring the most re-watched parts.
  - Cutline then downloads **only those ~1-minute parts** and transcribes just them, for exact caption timing.

  A 90-minute video takes minutes instead of an hour. If a video has no subtitles, Cutline falls back to the full process automatically. To force the full process, untick Fast mode or pass `--full`.

- **Transcription** runs on your CPU, or on an NVIDIA GPU if you have one. On a typical laptop, the default `small` model takes about 1 minute per 3 to 5 minutes of video. Use `--whisper base` for roughly double the speed, or `--whisper medium` for better accuracy, especially for non-English speech. The model downloads once on first use (about 500 MB for `small`).
- **Gemini analysis** uses `gemini-4-hydrogen`, which is on Gemini's free tier. If that model isn't available to your key, Cutline automatically tries `gemini-3.8-flash`, then `gemini-3.6-flash`, then `gemini-3.5-flash-lite`, all also free. It takes about 1 to 3 minutes per video. The free tier has daily request limits, but one video uses one request. Results are saved. If you only change caption or layout settings, run again and it re-renders without calling Gemini. Add `--reanalyze` (or tick "Re-run Gemini analysis" in the web app) when you want new picks.
- **Privacy:** on the free tier, Google may use what you send (here, your video transcripts) to improve its products. If that matters for a video, enable billing on your Google AI Studio account; paid usage isn't used this way.
- **Rendering** takes about 20 to 60 seconds per clip.

## Tips

- If the speech isn't in English, set `--lang` (for example `hi`, `ur`, `es` or `ar`) and consider `--whisper medium`.
- Use `--meta-lang English` to get English titles for a video in another language.
- Use `--instructions` to steer the picks, for example `"only the funny moments"` or `"clips about fitness for beginners"`.
- Use `--layout fit` for videos with two people side by side, slides or screen content. It always shows the whole frame.

## Troubleshooting

- **YouTube says "Sign in to confirm you're not a bot":** YouTube is limiting downloads from your internet connection. Pick one fix:
  - Download the video another way and use **Upload a video** instead of the link.
  - Give Cutline a YouTube login:
    1. In Chrome, install the **Get cookies.txt LOCALLY** extension, and allow it in Incognito (open chrome://extensions, then the extension's **Details**).
    2. Open an **Incognito** window, sign in to YouTube, then go to youtube.com/robots.txt.
    3. Click the extension, then **Export**. **Close the Incognito window** straight away. If you skip this, YouTube replaces the cookies within hours and the block comes back.
    4. Drop the file into **Advanced > YouTube cookies file** in the web app (or save it as `cookies.txt` in the Cutline folder, or pass `--cookies file.txt` on the command line).

    Treat `cookies.txt` like a password and don't share it.
  - Wait a few hours (the block is often temporary) and turn off any VPN.
- **A YouTube download fails another way:** update the downloader with `.venv\Scripts\python.exe -m pip install -U yt-dlp`. YouTube changes things often.
- **"No Gemini API key found":** see step 4 of Setup.
- **"None of the Gemini models could take this request" with 429 / RESOURCE_EXHAUSTED:** you've hit the free-tier limit. Wait a while (limits reset daily) and run again. Your transcript is saved, so it won't redo that step.
- **Captions use the wrong font:** they use Arial Black, which comes with Windows. Make sure it's installed.
